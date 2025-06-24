"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
Enhanced for better architecture

This script implements a modular architecture for training the CMNeXt model with confidence.
It separates concerns into different modules:
- Trainer: Handles the training and validation process
- DataModule: Manages dataset loading and preparation
- ModelModule: Handles model initialization and configuration
- OptimizerModule: Manages optimization strategy

See ARCHITECTURE.md for more details on the architectural improvements.
"""
import os
import argparse
import numpy as np
import logging
import torch
import gc
import wandb
import pretty_errors
from configs.cmnext_init_cfg import _C as config, update_config
from common.losses import TruForLoss

# Import custom modules
from trainer import Trainer
from data.data_module import DataModule
from models.model_module import ModelModule
from common.optimizer_module import OptimizerModule

# Configure garbage collection
gc.collect()

import warnings
warnings.filterwarnings("ignore", message="libpng warning: iCCP: known incorrect sRGB profile")
warnings.filterwarnings("ignore", message="libpng warning: iCCP: profile 'ICC profile': 'bTRC': ICC profile tag start not a multiple of 4")
warnings.filterwarnings("ignore", message="Corrupt EXIF data", module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")
pretty_errors.configure(
    separator_character='*',
    filename_display=pretty_errors.FILENAME_EXTENDED,
    line_number_first=True,
    display_link=True,
    lines_before=5,
    lines_after=2,
    line_color=pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
    code_color='  ' + pretty_errors.default_config.line_color,
    truncate_code=True,
    display_locals=True
)

def main():
    """
    Main training function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Training script for CMNeXt with confidence')
    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
    parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')
    parser.add_argument('-train_bayar', '--train_bayar', action='store_true', help='finetune bayar conv')
    parser.add_argument('-exp', '--exp', type=str, default=None, help='Yaml experiment file')
    parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--ckpt', type=str, default='', help='Resume from checkpoint path')
    args = parser.parse_args()

    # Update configuration from experiment file
    global config
    config = update_config(config, args.exp)

    # Set up logging
    loglvl = getattr(logging, args.log.upper())
    logging.basicConfig(level=loglvl, format='%(message)s')

    # Set up device
    gpu = args.gpu
    device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
    np.set_printoptions(formatter={'float': '{: 7.3f}'.format})
    print(f"Device: {device}")
    torch.set_flush_denormal(True)

    # Configure CUDA settings
    if device != 'cpu':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED

    # Initialize wandb
    wandb.init(
        project="mmfusion",  # replace it with your project name
        name=config.MODEL.NAME,    # use model name as run name
        config={
            "learning_rate": config.LEARNING_RATE,
            "architecture": "CMNeXtWithConf",
            "backbone": config.MODEL.BACKBONE,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "image_size": config.DATASET.IMG_SIZE,
            "modalities": config.MODEL.MODALS,
        }
    )

    # Initialize model module
    model_module = ModelModule(config, device, train_bayar=args.train_bayar)
    model, modal_extractor = model_module.setup()

    # Initialize data module
    data_module = DataModule(config)
    train_loader, val_loader, class_weights = data_module.setup()

    # Initialize criterion
    criterion = TruForLoss(weights=class_weights.to(device), ignore_index=-1)

    # Initialize optimizer module
    optimizer_module = OptimizerModule(config)
    optimizer, lr_scheduler, scaler = optimizer_module.setup(model, modal_extractor)

    # Update optimizer module with actual iterations per epoch
    iters_per_epoch = len(train_loader)
    optimizer_module.update_iters_per_epoch(iters_per_epoch)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        modal_extractor=modal_extractor,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        device=device,
        train_bayar=args.train_bayar
    )

    # Load checkpoint if provided
    start_epoch = model_module.load_checkpoint(args.ckpt)

    # Training loop
    min_loss = float('inf')
    for epoch in range(start_epoch, config.EPOCHS):
        # Shuffle dataset for balanced sampling
        data_module.shuffle_train_dataset()

        # Training phase
        train_loss = trainer.train_epoch(epoch, train_loader)

        # Validation phase
        val_loss, f1_best, f1_fixed = trainer.validate_epoch(epoch, val_loader)

        # Save best model
        if val_loss < min_loss:
            min_loss = val_loss
            trainer.save_checkpoint(epoch, val_loss, f1_best, f1_fixed, is_best=True)

    # Save final model
    trainer.save_final_model(config.EPOCHS - 1)

    # Close wandb
    wandb.finish()

if __name__ == '__main__':
    main()
