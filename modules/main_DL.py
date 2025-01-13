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
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
from torch.utils.data import DataLoader, Dataset
import time


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


config = parse_args()
text = load_text_data(config)


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

scaler = GradScaler() if device == "cuda" else None


def base64_to_tensor(base64_str, img_size=96):
    image = Image.open(io.BytesIO(base64.b64decode(base64_str)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_pipeline(image)



def collate_batch(batch):
    """Custom collate function to handle variable length sequences"""
    images, texts, targets = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Find max length in the batch
    max_len = max(text.size(0) for text in texts)
    
    # Pad sequences
    padded_texts = torch.full((len(texts), max_len), fill_value=stoi["<pad>"], dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), fill_value=stoi["<pad>"], dtype=torch.long)
    
    for i, (text, target) in enumerate(zip(texts, targets)):
        padded_texts[i, :len(text)] = text
        padded_targets[i, :len(target)] = target
    
    return images, padded_texts, padded_targets

class VisionLanguageDataset(Dataset):
    def __init__(self, df, img_size=96):
        self.df = df
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = base64_to_tensor(row["b64string_images"], self.img_size)
        text_indices = torch.tensor(encode(row["caption"]), dtype=torch.long)
        
        # Pad the text to the maximum length within the batch
        max_length = text_indices.size(0)
        padded_text = torch.full((max_length,), fill_value=stoi["<pad>"], dtype=torch.long)
        padded_text[:len(text_indices)] = text_indices
        
        # Create targets by shifting
        targets = torch.cat([padded_text[1:], torch.tensor([stoi["<pad>"]], dtype=torch.long)])

        return image, padded_text, targets


def train_model(model, df, config):
    mlflow.set_experiment("vision_language_training")
    
    # Create train/val splits
    n = int(0.9 * len(df))
    train_df = df.iloc[:n]
    val_df = df.iloc[n:]
    
    # Create datasets and dataloaders
    train_dataset = VisionLanguageDataset(train_df, config['img_size'])
    val_dataset = VisionLanguageDataset(val_df, config['img_size'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_batch)

    train_losses = []
    final_val_loss = None
    start_time = time.time()

    with mlflow.start_run():
        mlflow.log_params(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        model.to(device)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=tensorboard_trace_handler('./profiling_logs'),
            record_shapes=True,
            with_stack=True,
            with_flops=True
        ) as prof:
            for epoch in range(config['epochs']):
                epoch_start_time = time.time()
                model.train()
                for step_idx, (images, idx, targets) in enumerate(train_loader):
                    images, idx, targets = images.to(device), idx.to(device), targets.to(device)
                    optimizer.zero_grad()

                    with autocast(device_type=device) if device == 'cuda' else nullcontext():
                        logits, loss = model(images, idx, targets)

                    if scaler: scaler.scale(loss).backward()
                    else: loss.backward()
                    if scaler: scaler.step(optimizer)
                    else: optimizer.step()
                    if scaler: scaler.update()
                    prof.step()

                    mlflow.log_metric("loss", loss.item())
                    train_losses.append(loss.item())

                    if step_idx % 10 == 0:
                        print(f"Loss at iteration {step_idx}: {loss.item()}")
                    if step_idx == 0:
                        print(f"Total batches per epoch: {len(train_loader)}")

                val_loss = estimate_loss(model, val_loader)
                final_val_loss = val_loss
                mlflow.log_metric("val_loss", val_loss)
                print(f"Validation Loss after epoch {epoch}: {val_loss}")

                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")


    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    return final_val_loss

def estimate_loss(model, dataloader):
    """Computes average loss over validation dataset."""
    losses = []
    model.eval()
    with torch.no_grad():
        for images, idx, targets in dataloader:
            images, idx, targets = images.to(device), idx.to(device), targets.to(device)
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

    study.optimize(lambda t: objective(t, df, config), n_trials=1)
    print(f"\n✅ Best Hyperparameters: {study.best_params}")
    print(f"✅ Best Validation Loss: {study.best_value}")


if __name__ == "__main__":
    main()
    
