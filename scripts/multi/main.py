import os
import random
import numpy as np
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from data_module import BurnedAreasDataModule
from lightning_module import LightningSegmentation
from config import config

def run_experiment(config, start_from_checkpoint=False, run_id=None, ckpt_path=None):
    # Ensure any previous run is finished
    wandb.finish()
     
    # Login to W&B
    wandb.login(key=config['wandb_api_key'])

    # Setup W&B logger
    wandb_logger = WandbLogger(project="Burned Areas Delineation", name=config['experiment_name'], resume="allow" if start_from_checkpoint else None)
    
    # Log .py files as artifacts
    code_files = [f for f in os.listdir('.') if f.endswith('.py') or f.endswith('.ipynb')]
    code_artifact = wandb.Artifact('code_files', type='code')
    for file in code_files:
        code_artifact.add_file(file)
    wandb_logger.experiment.log_artifact(code_artifact)

    # Fix the seed for reproducibility
    pl.seed_everything(config['random_state'], workers=True)
    np.random.seed(config['random_state'])
    random.seed(config['random_state'])

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Check if mixed precision is enabled and configure accordingly
    if config['use_mixed_precision']:
        torch.set_float32_matmul_precision('high')

    # Initialize Data Module
    data_module = BurnedAreasDataModule(
        data_root_dir=config['data_root_dir'],
        batch_size=config['batch_size'],
        additional_features=config['additional_features'],
        stats=config['all_stats'],
        random_state=config['random_state']
    )

    # Initialize Lightning Model
    lightning_model = LightningSegmentation(
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        model_name=config['model_name'],
        optimizer_name=config['optimizer_name'],
        scheduler_name=config['scheduler_name'],
        loss_func_name_ba=config['loss_func_name_ba'],
        loss_func_name_lc=config['loss_func_name_lc'],
        loss_factor_lc=config['loss_factor_lc'],
        data_root_dir=config['data_root_dir'],
        additional_features=config['additional_features'],
        all_stats=config['all_stats'],
        use_tta=config['use_tta'],
        outputs_dir=config['outputs_dir'],
        export_preds=config['export_preds'],
        model_kwargs=config['model_kwargs'],
        optimizer_kwargs=config['optimizer_kwargs'],
        scheduler_kwargs=config['scheduler_kwargs'],
        loss_func_kwargs_ba=config['loss_func_kwargs_ba'],
        loss_func_kwargs_lc=config['loss_func_kwargs_lc']
    )

    # Print the device being used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # If CUDA is available, print more detailed information about the GPU
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
    # Trainer Setup
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        precision="16-mixed" if config['use_mixed_precision'] else "32-true",
        accumulate_grad_batches=config['gradient_accumulation_steps'],
        callbacks=[
            ModelSummary(),
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(dirpath='checkpoints', filename=f"{config['experiment_name']}-{{epoch}}-{{val_loss:.2f}}", monitor='val_loss', mode='min', save_top_k=1),
            EarlyStopping(monitor='val_loss', patience=20, verbose=True)
        ],
        #limit_train_batches=0.2,
        #limit_val_batches=0.2,
        #limit_test_batches=0.2,
        #fast_dev_run=True,
        logger=wandb_logger,
    )

    # Fit the Model
    trainer.fit(lightning_model, datamodule=data_module, ckpt_path=ckpt_path if start_from_checkpoint else None)

    # After training is complete, log best checkpoint
    best_checkpoint_path = trainer.checkpoint_callback.best_model_path
    model_artifact = wandb.Artifact(name='model', type='model', description='Best model checkpoint')
    model_artifact.add_file(best_checkpoint_path)
    wandb_logger.experiment.log_artifact(model_artifact)

    # Test the Model
    trainer.test(datamodule=data_module, ckpt_path='best')

    # Finish the W&B run
    wandb.finish()

run_experiment(config, start_from_checkpoint=False)
#run_experiment(config, start_from_checkpoint=True, run_id='s6q2yvng', ckpt_path=r"/kaggle/working/checkpoints/Testing v1-epoch=0-val_loss=0.97.ckpt")