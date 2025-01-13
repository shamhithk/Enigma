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



