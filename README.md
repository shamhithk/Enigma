# Enigma ML Engineer Take Home Evaluation

 Most changes are done in ```modules``` folder ```main.py```

## Frameworks / Libraries explored for the evaluation:
- PyYAML
- PyTorch
- torchvision
- mlflow
- optuna
- matplotlib
- torch.profiler
- tensorboard
- DataLoader
- DistributedSampler
- torch.distributed
- torch.multiprocessing
- DistributedDataParallel (DDP)

 ## Files 
 ```

├── config.yaml                     # contains all my hyperparameters
├── profiling_logs/                 # Traces of model every time each batch is processed capturing duration, FLOPS, memory utilization
├── optuna_study.db                 # Optuna database for hyperparameter tuning helps in visualizing Training loss as well
├── mlruns/                         # MLFlow experiment logs and artifacts
├── requirements.txt                # Python dependencies
└── training_loss.png               # Graph of training loss
```

Execution:

```  python -m modules.main --config config.yaml```

# Task 1: Configuring Hyperparameters through CLI

### Configuring Hyperparameters via config.yaml

Specified the model and training hyperparameters in a config.yaml file. This YAML file allows you to modify the key hyperparameters without altering the code.



```
![][]
# Dataset and file paths
input_path: "../images/inputs.csv"

# Precision settings
precision_mode: 'mixed'  # default precision mode ('fp32', 'fp16', or 'mixed')

# For mixed/half precision specific settings
amp_enabled: true  # enable automatic mixed precision
grad_scaler_enabled: true  # enable gradient scaling for mixed precision
init_scale: 65536  # initial scale factor for mixed precision training
growth_interval: 2000  # how many steps between growing the scale factor

# Hardware optimization
cuda_enabled: true  # enable CUDA if available
num_workers: 4  # number of data loading workers
pin_memory: true  # pin memory for faster data transfer

# Model hyperparameters
n_embd: 128
image_embed_dim: 512
vocab_size: 6666  # Adjusted based on your character encoding
n_layer: 8
img_size: 96
patch_size: 16
num_head: 8
head_size: 16ml
dropout: 0.1
num_blks: 3
emb_dropout: 0.1
blk_dropout: 0.1

# Training hyperparameters
batch_size: 16
val_batch_size: 8
block_size: 32
max_iters: 100
eval_interval: 10
learning_rate: 0.001
epochs: 1
eval_iters: 40
hyperparameter_tuning: true  # Set to false to skip tuning
n_trials: 10
optimization_timeout: 3600  # in seconds
weight_decay: 0.01

```

### Parsing Parameters Using ``` argparse ```
The main.py uses argparse to load the YAML file dynamically via the command line.

```
def parse_args():
    parser = argparse.ArgumentParser(description="Vision Language Model Trainer")
    parser.add_argument('--config', type=str, required=True, help="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
```

### Loading and Applying the Parameters in Code
Once the configuration is loaded, the values are applied directly to set the hyperparameters for training:

```
batch_size = config['batch_size']
learning_rate = config['learning_rate']
```


# Task 2: Simple solution  for hyperparameter management and tracking

Optuna is an open-source framework specifically designed for hyperparameter optimization in machine learning


### Optuna: ##

![Screenshot 2025-01-13 at 8 07 01 PM](https://github.com/user-attachments/assets/f053d22d-4bdb-4c20-8a70-fe236e5c2883)

- Optuna is used for automated hyperparameter tuning and run ``` optuna-dashboard sqlite:///optuna_study.db``` this command to see dashboard
- The study results are stored in an SQLite database (optuna_study.db).
- Trials adjust learning rates, batch sizes, and embedding dimensions.




# Task 3: Storing and Visualizing training loss():

MLflow is a open-source platform for managing the entire machine learning lifecycle, including experiment tracking, model management, and deployment.

### mlflow: ##

![mlflow runs](https://github.com/user-attachments/assets/70cff627-c37a-434e-9ea6-ee94f6dc9fbc)

- run this command ``` mlflow ui``` after training the model to check out mlflow dashboard
- All hyperparameters are logged using MLFlow during training runs.
- MLFlow records metrics like loss and validation loss for performance analysis.

Optuna is primarily used to find the best hyperparameters for a model, while MLflow tracks and compares the results of different model configurations including those tuned with Optuna.

### matplotlib.pyplot:
```
# Plot & save training loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.savefig("training_loss.png")
        mlflow.log_artifact("training_loss.png")
```

![training_loss](https://github.com/user-attachments/assets/ce0e3416-6cc8-451b-a86c-201fa7f59e72)

The plot provided shows the training loss decreasing over time during model training.

# Task 4: Simple solution for profiling the training performance to identify bottlenecks in the model configuration

torch.profiler is used to capture traces of model, which helps to optimize structure, compute and architecture.

The profiling is done using PyTorch's built-in profiler functionality during the model training phase.

### First, the code imports the necessary profiling components from PyTorch:
```
pythonCopyfrom torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler
```
### In the ```train_model``` function, profiling is set up using a context manager with the following configuration:
```
pythonCopywith profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./profiling_logs'),
    record_shapes=True,
    with_stack=True,
    with_flops=True
) as prof:
```
#### Let's break down each profiling parameter:

```activities:``` Specifies which hardware to profile

- Profiles both CPU and CUDA (GPU) if CUDA is available
- Otherwise, only profiles CPU activities


```schedule:``` Defines the profiling schedule with four phases:

- ```wait=1:``` Skip first step
- ```warmup=1:``` Warmup for 1 step
- ```active=3:``` Actively profile for 3 steps
- ```repeat=1:``` Repeat this cycle once


```on_trace_ready:``` Uses TensorBoard trace handler to save profiling results to './profiling_logs'
```record_shapes:``` Enables recording of tensor shapes
```with_stack:``` Records Python stack traces
```with_flops:``` Enables calculation of floating point operations (FLOPS)


### Inside the training loop, the profiler steps forward after each batch:

```prof.step()```

The profiling data collected includes:

CPU and GPU utilization, Memory usage, Tensor shapes, Stack traces, Timeline of operations

#### Trace of model:
Trace is a ```.json``` that contains all the information about model profiles


We tried ```tensorboard``` or you can load the saved trace file to this site ```chrome://tracing ``` for visualization of these files

### Chrome://tracing for a run on single thread
<img width="1113" alt="Screenshot 2025-01-12 at 12 26 21 PM" src="https://github.com/user-attachments/assets/f17d39ff-0f5a-4410-a4d4-bae57a4ac790" />

### Chrome://tracing for a run on multiple threads
<img width="676" alt="Screenshot 2025-01-13 at 8 42 43 PM" src="https://github.com/user-attachments/assets/d59e3959-9b22-4699-b8f4-d9cdae03188b" />

I've also tried to write a simple python script to get top 100 events given a trace file - ```top100_trace.ipynb```

Looking at top 100 is easier and can tell us imp information to find bottlenecks


#### Taking all traces I've run into consideration, below are the potential bottlenecks in model configuration:

- Data Loading and Threading Bottlenecks
- Backward Pass Inefficiency
- Optimizer Performance
- Can optimize forward pass by pruning or quantization

#### Improvements:
- I observed taking too many profiles isn't ideal for model performance
- Can add profiles for particular layers in model to identify more in depth what functions are taking more computing, time, etc.


# Task 5: Include  best practices into the training to ensure the fastest performance possible (i.e. half precision, ...)

Tried  Data Loader, Mixed Precision with CPU and GPU handling

### Data Loading Implementation:

This code implements a custom dataset and data loader setup using PyTorch's utilities:
```
pythonCopyclass VisionLanguageDataset(Dataset):
    def __init__(self, df, img_size=96):
        self.df = df
        self.img_size = img_size

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = base64_to_tensor(row["b64string_images"], self.img_size)
        text_indices = torch.tensor(encode(row["caption"]), dtype=torch.long)
        
        # Padding implementation for variable length sequences
        max_length = text_indices.size(0)
        padded_text = torch.full((max_length,), fill_value=stoi["<pad>"], dtype=torch.long)
        padded_text[:len(text_indices)] = text_indices
        targets = torch.cat([padded_text[1:], torch.tensor([stoi["<pad>"]], dtype=torch.long)])

        return image, padded_text, targets
```

#### Key Data Loading features:

- Custom collate function for handling variable length sequences
```
pythonCopydef collate_batch(batch):
    images, texts, targets = zip(*batch)
    images = torch.stack(images)
    max_len = max(text.size(0) for text in texts)
    padded_texts = torch.full((len(texts), max_len), fill_value=stoi["<pad>"], dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), fill_value=stoi["<pad>"], dtype=torch.long)
```
- DataLoader configuration

```
pythonCopytrain_loader = DataLoader(
    train_dataset, 
    batch_size=config['batch_size'], 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True, 
    collate_fn=collate_batch
)
```
### Precision Techniques:

Tried implementing mixed precision training using PyTorch's automatic mixed precision (AMP)
```
pythonCopyscaler = GradScaler() if device == "cuda" else None
```
The training loop uses precision optimization through
```
pythonCopywith autocast(device_type=device) if device == 'cuda' else nullcontext():
    logits, loss = model(images, idx, targets)

if scaler: 
    scaler.scale(loss).backward()
else: 
    loss.backward()

if scaler: 
    scaler.step(optimizer)
else: 
    optimizer.step()

if scaler: 
    scaler.update()
```

#### Key precision features:

- Conditional AMP implementation based on device availability
- GradScaler for managing gradient scaling in mixed precision
- Proper handling of both CUDA and CPU scenarios
- Integration with the optimizer step


# Task 6: Extension of this training function in order to be scaleable to a multi-GPU or multi-node setting.

For efficient GPU / CPU scaling, I tried  pytorch ``` DDP  Distributed (Data Parallel) ```, Since I don't have access to multi GPU at the moment I was not able to run the code to check results, but my plan for the code is as follows

###To enable multi-GPU utilization

#### Setup and cleanup functions:

- ``` setup_ddp()```: Initializes the distributed process group
 ``` 
def setup_ddp(rank, world_size):
    """Initialize DDP process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

```
- ``` cleanup_ddp():``` Cleans up the process group after training

  ```
  def cleanup_ddp():
    """Clean up DDP process group"""
    dist.destroy_process_group()
 
  ```

- Modify the ```train_model``` function to accomodate GPU utility
```
  def train_model(rank, world_size, model, df, config):
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Modify model for DDP
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
-----
------

    train_sampler = DistributedSampler(train_dataset, 
                                      num_replicas=world_size,
                                      rank=rank)
    val_sampler = DistributedSampler(val_dataset,
                                    num_replicas=world_size,
                                    rank=rank)

# Create dataloaders with distributed samplers
    train_loader = DataLoader(train_dataset, 
                            batch_size=config['batch_size'],
                            sampler=train_sampler,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=collate_batch)
    
    val_loader = DataLoader(val_dataset,
                          batch_size=config['batch_size'],
                          sampler=val_sampler,
                          num_workers=4,
                          pin_memory=True,
                          collate_fn=collate_batch)

--------
---------

    # Clean up
    cleanup_ddp()
    if rank == 0:
        mlflow.end_run()

    
```

- Modify ```main()``` and use ``mp.spawn`` to create multiple processes
```
  ------
---------

# Get world size from available GPUs
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        # Spawn processes for DDP
        import torch.multiprocessing as mp
        mp.spawn(
            train_model,
            args=(world_size, model, df, config),
            nprocs=world_size,
            join=True
        )
    else:
        print("No multiple GPUs found. Running on single GPU or CPU.")
        train_model(0, 1, model, df, config)
------
------

```
- Modify ```estimate_loss ()``` to calculate avg_loss which is how GPU's learn and can paralelize

Ambigous about the code, but I'm thinking something like below
```
 def estimate_loss(model, val_loader, rank):
    """ 
    Args:
        model: DDP-wrapped model
        val_loader: DataLoader with validation data
        rank: Current device rank
    """
    model.eval()
    total_loss = torch.zeros(1).to(rank)
    total_samples = torch.zeros(1).to(rank)
    
    with torch.no_grad():
        for images, idx, targets in val_loader:
            images = images.to(rank)
            idx = idx.to(rank)
            targets = targets.to(rank)
            
            _, loss = model(images, idx, targets)
            batch_loss = loss.item() * len(images)
            total_loss += batch_loss
            total_samples += len(images)
    
    # Synchronize statistics across all GPUs
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    
    # Calculate average loss
    avg_loss = total_loss.item() / total_samples.item()
    
    return avg_loss

```




