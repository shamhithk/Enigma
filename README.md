# Enigma ML Engineer Take Home Evaluation

## Frameworks / Libraries explored for the evaluation:
- PyYAML
- PyTorch
- torchvision
- mlflow
- optuna
- matplotlib
- torch.profiler
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


## Task 1: Configuring Hyperparameters through CLI

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

Task 2: Simple solution  for hyperparameter management and tracking

MLflow and Optuna tools are for hyperparameter tuning and management as shown below



