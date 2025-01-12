import os
import base64
import io
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import argparse
import yaml
import mlflow
import optuna
import matplotlib.pyplot as plt
from optuna.storages import RDBStorage
from optuna import create_study
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler

from .vision_language_model import VisionLanguageModel

current_dir = os.path.dirname(__file__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Vision Language Model Trainer")
    parser.add_argument('--config', type=str, required=True, help="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_text_data(config):
    """Reads text data from the file specified in config['input_path']."""
    input_path = config['input_path']
    filename = os.path.join(current_dir, input_path)
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    return text


# ----------------- Initialize Config and Text ----------------- #

config = parse_args()
text = load_text_data(config)

# def custom_trace_handler(base_dir):
#     def handler(prof):
#         # Generate the proper directory structure
#         plugin_dir = os.path.join(base_dir, "plugins/profile")
#         timestamp = torch.profiler._utils._format_time(prof.start_time)
#         output_dir = os.path.join(plugin_dir, timestamp)
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Save the trace file in the correct directory
#         tensorboard_trace_handler(output_dir)(prof)
#         print(f"Trace saved to: {output_dir}")
#     return handler

# Configure the PyTorch Profiler
profiler = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    if torch.cuda.is_available()
    else [ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('/Users/shamhithreddy/Desktop/Enigma/seemore/profiling_logs'),
    record_shapes=True,
    with_stack=True,
    with_flops=True
)

# Pull hyperparameters from config
batch_size = config['batch_size']
block_size = config['block_size']
max_iters = config['max_iters']
eval_interval = config['eval_interval']
learning_rate = config['learning_rate']
epochs = config['epochs']
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = config['eval_iters']
num_blks = config['num_blks']
head_size = config['head_size']
n_embd = config['n_embd']
num_head = config['num_head']
n_layer = config['n_layer']
dropout = config['dropout']
img_size = config['img_size']
patch_size = config['patch_size']
image_embed_dim = config['image_embed_dim']
emb_dropout = config['emb_dropout']
blk_dropout = config['blk_dropout']

# Build vocabulary
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
stoi["<pad>"] = len(stoi)
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l if i in itos])
vocab_size = len(stoi)


def base64_to_tensor(base64_str, img_size=96):
    image = Image.open(io.BytesIO(base64.b64decode(base64_str)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_pipeline(image).unsqueeze(0)


def get_batch(df, batch_size, split="train", img_size=96, val_batch_size=8):
    """Samples a batch from train or val."""
    n = int(0.9 * len(df))
    df_train = df.iloc[:n]
    df_val = df.iloc[n:]
    data = df_train if split == "train" else df_val

    batch_size = batch_size if split == "train" else val_batch_size
    replace = False if split == "train" else True
    batch = data.sample(n=batch_size, replace=replace)

    images = torch.cat([
        base64_to_tensor(img, img_size) for img in batch["b64string_images"]
    ], dim=0).to(device)

    text_indices = [
        torch.tensor(encode(desc), dtype=torch.long) for desc in batch["caption"]
    ]
    # Ensure no index exceeds vocab size
    for idx_tensor in text_indices:
        if idx_tensor.numel() > 0 and idx_tensor.max().item() >= vocab_size:
            raise ValueError(f"Index out of range: {idx_tensor.max().item()} for vocab size {vocab_size}")

    max_length = max(len(t) for t in text_indices)
    padded_text = torch.full(
        (batch_size, max_length), fill_value=stoi["<pad>"], dtype=torch.long
    ).to(device)
    for i, t in enumerate(text_indices):
        padded_text[i, : len(t)] = t

    targets = torch.cat([
        padded_text[:, 1:],
        torch.full((batch_size, 1), fill_value=stoi["<pad>"], dtype=torch.long, device=device),
    ], dim=1)

    return images, padded_text, targets


def train_model(model, df, config):
    """Trains the model (with the PyTorch profiler) and logs metrics to MLflow."""
    mlflow.set_experiment("vision_language_training")
    train_losses = []
    final_val_loss = None

    with mlflow.start_run():
        mlflow.log_params(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        model.to(device)

        # Use the profiler context
        with profiler:
            for epoch in range(config['epochs']):
                model.train()
                for step_idx in range(config['max_iters']):
                    with torch.profiler.record_function("data_loading"):
                        images, idx, targets = get_batch(df, config['batch_size'], "train", config['img_size'])
                    optimizer.zero_grad()
                    profiler.step()

                    with torch.profiler.record_function("forward_pass"):
                        logits, loss = model(images, idx, targets)
                    profiler.step()

                    with torch.profiler.record_function("backward_pass"):
                        loss.backward()
                        optimizer.step()
                    profiler.step()

                    mlflow.log_metric("loss", loss.item())
                    train_losses.append(loss.item())

                    if step_idx % config['eval_interval'] == 0:
                        print(f"Loss at iteration {step_idx}: {loss.item()}")

                val_loss = estimate_loss(model, df, "val", config['img_size'])
                final_val_loss = val_loss
                mlflow.log_metric("val_loss", val_loss)
                print(f"Validation Loss after epoch {epoch}: {val_loss}")
        profiler.stop()

        # Plot & save training loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.savefig("training_loss.png")
        mlflow.log_artifact("training_loss.png")

    return final_val_loss


def estimate_loss(model, df, split, img_size=96, val_batch_size=8):
    """Computes average loss over 40 random samples."""
    losses = []
    model.eval()
    for _ in range(40):
        images, idx, targets = get_batch(
            df, batch_size, split, img_size, val_batch_size=val_batch_size
        )
        _, loss = model(images, idx, targets)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def objective(trial, df, config):
    """Objective function for Optuna hyperparameter tuning."""
    config['learning_rate'] = trial.suggest_float("learning_rate", 1e-3, 6e-3, log=True)
    config['batch_size'] = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    config['n_embd'] = trial.suggest_categorical("n_embd", [128, 256, 512])

    model = VisionLanguageModel(
        config['n_embd'],
        config['image_embed_dim'],
        config['vocab_size'],
        config['n_layer'],
        config['img_size'],
        config['patch_size'],
        config['num_head'],
        config['num_blks'],
        config['emb_dropout'],
        config['blk_dropout'],
    )
    final_val_loss = train_model(model, df, config)
    return final_val_loss


def main():
    input_path = config['input_path']
    df = pd.read_csv(os.path.join(current_dir, input_path))
    df = pd.concat([df] * 30)[["b64string_images", "caption"]]

    storage_name = "sqlite:///optuna_study.db"
    storage = RDBStorage(url=storage_name)
    study = optuna.create_study(direction="minimize", storage=storage_name, load_if_exists=True)

    # Run exactly 1 trial for demonstration. Increase as needed.
    study.optimize(lambda t: objective(t, df, config), n_trials=1)
    print(f"\n✅ Best Hyperparameters: {study.best_params}")
    print(f"✅ Best Validation Loss: {study.best_value}")

    # No second training call – code ends here


if __name__ == "__main__":
    main()