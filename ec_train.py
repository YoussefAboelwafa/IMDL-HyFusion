"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
Enhanced for better architecture and training pipeline
"""
import os
import argparse
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import gc
import wandb
import pretty_errors
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.cmnext_init_cfg import _C as config, update_config
from common.losses import TruForLoss
from common.utils import AverageMeter
from common.lr_schedule import WarmUpPolyLR
from common.split_params import group_weight
from common.metrics import computeLocalizationMetrics
from data.datasets import MixDataset
from models.modal_extract import ModalitiesExtractor
from models.cmnext_conf import CMNeXtWithConf

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


def edge_loss(pred, target, weights=None):
    """Binary cross entropy loss for edge detection with Sobel operator"""
    target = target.float()
    
    # Handle different tensor dimensions
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    elif pred.dim() == 4:
        # If multi-channel, take the first channel or mean
        pred = pred.mean(dim=1)
    
    # Ensure target and pred have the same shape
    if target.shape != pred.shape:
        target = F.interpolate(target.unsqueeze(1), size=pred.shape[-2:], mode='nearest').squeeze(1)
    
    # Apply Sobel edge detection to target if it's not already an edge map
    if target.max() > 0.5:  # If target appears to be a mask rather than edges
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        
        target_unsqueezed = target.unsqueeze(1)
        edge_x = F.conv2d(target_unsqueezed, sobel_x, padding=1)
        edge_y = F.conv2d(target_unsqueezed, sobel_y, padding=1)
        target = torch.sqrt(edge_x**2 + edge_y**2).squeeze(1)
        target = (target > 0.1).float()  # Threshold to create binary edge map
    
    if weights is not None:
        loss = F.binary_cross_entropy_with_logits(pred, target, weight=weights, reduction='mean')
    else:
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
    return loss


def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    loglvl = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=loglvl, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)


def setup_device(gpu_id):
    """Setup device and CUDA configurations"""
    device = f'cuda:{gpu_id}' if gpu_id >= 0 and torch.cuda.is_available() else 'cpu'
    
    if device != 'cpu':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED
        torch.cuda.empty_cache()
    
    torch.set_flush_denormal(True)
    return device


def setup_wandb(config):
    """Initialize Weights & Biases logging"""
    try:
        wandb.login(relogin=True, key="950a4876c7dde9edca91c06ccc130f66964130f9") 
        wandb.init(
            project="mmfusion",
            name=config.MODEL.NAME,
            config={
                "learning_rate": config.LEARNING_RATE,
                "architecture": "CMNeXtWithConf", 
                "backbone": config.MODEL.BACKBONE,
                "epochs": config.EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "image_size": config.DATASET.IMG_SIZE,
                "modalities": config.MODEL.MODALS,
            },
            mode="online"
        )
    except Exception as e:
        logging.warning(f"Failed to initialize wandb: {e}. Continuing without wandb logging.")


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
    parser.add_argument('--model_pretrained', type=str, default='', help='Path to pretrained model weights')
    args = parser.parse_args()

    # Update configuration from experiment file
    global config
    config = update_config(config, args.exp)

    # Set up logging
    logger = setup_logging(args.log)
    
    # Set up device
    device = setup_device(args.gpu)
    logger.info(f"Device: {device}")
    
    # Initialize wandb
    setup_wandb(config)
    
    # Initialize models
    try:
        modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], config.MODEL.NP_WEIGHTS)
        if 'bayar' in config.MODEL.MODALS:
            pretrained_path = 'pretrained/pretrained/modal_extractor/bayar_mhsa.pth'
            if os.path.exists(pretrained_path):
                modal_extractor.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')), strict=False)
                logger.info(f"Loaded pretrained modal extractor from {pretrained_path}")
            else:
                logger.warning(f"Pretrained modal extractor not found at {pretrained_path}")
            
            if not args.train_bayar:
                modal_extractor.bayar.eval()
                for param in modal_extractor.bayar.parameters():
                    param.requires_grad = False
                logger.info("Bayar layer set to eval mode and frozen")

        model = CMNeXtWithConf(config.MODEL)
        modal_extractor.to(device)
        model = model.to(device)
        logger.info("Models initialized and moved to device successfully")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

    # Initialize datasets
    train = MixDataset(config.DATASET.TRAIN,
                    config.DATASET.IMG_SIZE,
                    train=True,
                    class_weight=config.DATASET.CLASS_WEIGHTS)

    val = MixDataset(config.DATASET.VAL,
                    config.DATASET.IMG_SIZE,
                    train=False)

    logger.info(train.get_info())
    def worker_init_fn(worker_id):
        # Set different seed for each worker
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Initialize data loaders
    train_loader = DataLoader(
        train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(val,
                            batch_size=1,
                            shuffle=False,
                            num_workers=config.WORKERS,
                            pin_memory=True)

    # Initialize loss function
    criterion = TruForLoss(weights=train.class_weights.to(device), ignore_index=-1)

    # Create output directories
    os.makedirs('./ckpt/{}'.format(config.MODEL.NAME), exist_ok=True)
    logdir = './{}/{}'.format(config.LOG_DIR, config.MODEL.NAME)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter('./{}/{}'.format(config.LOG_DIR, config.MODEL.NAME))

    # Setup optimizer parameters
    params = []
    cmnext_params = []
    modal_extract_params = []
    cmnext_params = group_weight(cmnext_params, model, torch.nn.BatchNorm2d, config.LEARNING_RATE)
    modal_extract_params = group_weight(modal_extract_params, modal_extractor, torch.nn.BatchNorm2d, config.LEARNING_RATE)

    params.append(dict(params=cmnext_params[0]['params'] + modal_extract_params[0]['params'], lr=config.LEARNING_RATE))
    params.append(dict(params=cmnext_params[1]['params'] + modal_extract_params[1]['params'], weight_decay=.0,
                    lr=config.LEARNING_RATE))

    optimizer = torch.optim.SGD(params,
                                lr=config.LEARNING_RATE,
                                momentum=config.SGD_MOMENTUM,
                                weight_decay=config.WD
                                )

    # Setup learning rate scheduler
    iters_per_epoch = len(train_loader)
    max_iters = config.EPOCHS * iters_per_epoch

    lr_schedule = WarmUpPolyLR(optimizer,
                            start_lr=config.LEARNING_RATE,
                            lr_power=config.POLY_POWER,
                            total_iters=max_iters,
                            warmup_steps=iters_per_epoch * config.WARMUP_EPOCHS)

    scaler = torch.cuda.amp.GradScaler()

    # Cleanup
    del params
    del cmnext_params
    del modal_extract_params
    gc.collect()
    torch.cuda.empty_cache()

    # Load checkpoint if provided
    start_epoch = 0

    if args.model_pretrained and os.path.exists(args.model_pretrained):
        model.load_training_checkpoint(args.model_pretrained,modal_extractor=modal_extractor, optimizer=optimizer, scaler=scaler,lr_schedule=lr_schedule, map_location=device)

    if args.ckpt and os.path.exists(args.ckpt):
        logger.info(f'Loading checkpoint from {args.ckpt}')
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        modal_extractor.load_state_dict(ckpt['extractor_state_dict']) 
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        logger.info(f'Resumed from epoch {start_epoch}')
    def train_epoch(epoch, model, modal_extractor, train_loader, criterion, optimizer, scaler, writer, device, config):
        """Run one training epoch"""
        torch.cuda.empty_cache()

        model.set_train()
        if args.train_bayar:
            modal_extractor.set_train()

        avg_loss = AverageMeter()
        edge_loss_avg = AverageMeter()
        iters_per_epoch = len(train_loader)

        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{config.EPOCHS}', unit='steps')
        optimizer.zero_grad(set_to_none=True)

        for step, (images, name, masks, _) in enumerate(pbar):
            # try:
                images = images.to(device, non_blocking=True)
                masks = masks.squeeze(1).to(device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    modals = modal_extractor(images)
                    images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    inp = [images_norm] + modals

                    pred, edge, _ = model(inp)

                    # Create edge ground truth using Sobel operator
                    edge_gt = torch.zeros_like(edge)
                    if masks.max() > 0:  # Only compute edges if there are positive pixels
                        edge_gt[masks == 1] = 1

                    edge_loss_val = edge_loss(edge, edge_gt)
                    
                    # Main segmentation loss
                    seg_loss = criterion(pred, masks) / config.ACCUMULATE_ITERS
                    
                    # Combined loss
                    total_loss = seg_loss + edge_loss_val * getattr(config, 'EDGE_LOSS_WEIGHT', 0.1)
                    loss = total_loss

                scaler.scale(loss).backward()
                
                if config.WD > 0:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data.add_(param.data, alpha=config.WD)
                            
                if ((step + 1) % config.ACCUMULATE_ITERS == 0) or (step + 1 == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    if step % 100 == 0:
                        torch.cuda.empty_cache()

                avg_loss.update(loss.detach().item())
                edge_loss_avg.update(edge_loss_val.detach().item())

                curr_iters = epoch * iters_per_epoch + step
                lr_schedule.step(cur_iter=curr_iters)
                
                try:
                    wandb.log({
                            "train/step_loss": loss.detach().item(),
                            "train/edge_loss": edge_loss_val.detach().item(),
                            "train/learning_rate": optimizer.param_groups[0]['lr']
                        }, step=curr_iters)
                except:
                    pass  # Continue without wandb if it fails
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], curr_iters)

                if step == 0:
                    maps = torch.nn.functional.softmax(pred, dim=1)[:, 1, :, :]
                    writer.add_images('Images-Masks-Preds',
                                    torch.cat((
                                        images,
                                        torch.tile(masks.unsqueeze(1), (1, 3, 1, 1)),
                                        torch.tile(maps.unsqueeze(1), (1, 3, 1, 1))), -2),
                                    epoch)

                pbar.set_postfix({"last_loss": loss.detach().item(), "epoch_loss": avg_loss.average()})
                
                # Clear intermediate tensors
                del pred, edge, loss
                torch.cuda.empty_cache()
                gc.collect()
                

        writer.add_scalar('Training Loss', avg_loss.average(), epoch)
        try:
            wandb.log({
                "train/epoch_loss": avg_loss.average(),
                "train/epoch_edge_loss": edge_loss_avg.average(),
                "epoch": epoch
            })
        except:
            pass  # Continue without wandb if it fails

        return avg_loss.average()

    def validate_epoch(epoch, model, modal_extractor, val_loader, criterion, writer, device, config):
        """Run one validation epoch"""
        model.set_val()
        modal_extractor.set_val()

        val_loss_avg = AverageMeter()
        f1 = []
        f1th = []

        pbar = tqdm(val_loader, desc=f'Validating Epoch {epoch + 1}/{config.EPOCHS}', unit='steps')

        for step, (images, _, masks, lab) in enumerate(pbar):
            with torch.no_grad():
                images = images.to(device, non_blocking=True)
                masks = masks.squeeze(1).to(device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    if any(torch.isnan(t).any() for t in images):
                        logging.error("Images contain NaN values!")
                        raise ValueError("Images contain NaN values")
                    modals = modal_extractor(images)
                    images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    inp = [images_norm] + modals

                    pred, edge, _ = model(inp)
                    val_loss = criterion(pred, masks)
                    
                    # Optional: also compute edge loss for validation tracking
                    if hasattr(config, 'TRACK_EDGE_VAL') and config.TRACK_EDGE_VAL:
                        edge_gt = torch.zeros_like(edge)
                        if masks.max() > 0:
                            edge_gt[masks == 1] = 1
                        edge_val_loss = edge_loss(edge, edge_gt)
                        val_loss += edge_val_loss * getattr(config, 'EDGE_LOSS_WEIGHT', 0.1)

                val_loss_avg.update(val_loss.detach().item())

                gt = masks.squeeze().cpu().numpy()
                map = torch.nn.functional.softmax(pred, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
                F1_best, F1_th = computeLocalizationMetrics(map, gt)
                f1.append(F1_best)
                f1th.append(F1_th)
                
                # Clear memory
                del images, masks, modals, images_norm, pred, edge
                gc.collect()
                if step % 50 == 0:
                    torch.cuda.empty_cache()

        writer.add_scalar('Val Loss', val_loss_avg.average(), epoch)
        writer.add_scalar('Val F1 best', np.nanmean(f1), epoch)
        writer.add_scalar('Val F1 fixed', np.nanmean(f1th), epoch)
        metrics = {
            "val/loss": val_loss_avg.average(),
            "val/f1_best": np.nanmean(f1),
            "val/f1_fixed": np.nanmean(f1th),
            "epoch": epoch
        }
        try:
            wandb.log(metrics)
        except:
            pass  # Continue without wandb if it fails
        return val_loss_avg.average(), np.nanmean(f1), np.nanmean(f1th)

    # Training loop
    min_loss = float('inf')
    best_f1 = 0.0
    
    logger.info(f"Starting training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, config.EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{config.EPOCHS}")
        
        # Shuffle dataset for balanced sampling
        train.shuffle()

        # Training phase
        train_loss = train_epoch(epoch, model, modal_extractor, train_loader, criterion, 
                                optimizer, scaler, writer, device, config)
        
        # Validation phase
        val_loss, f1_best, f1_fixed = validate_epoch(epoch, model, modal_extractor, val_loader,
                                                    criterion, writer, device, config)

        logger.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"F1 Best: {f1_best:.4f}, F1 Fixed: {f1_fixed:.4f}")

        # Save best model based on validation loss
        if val_loss < min_loss:
            min_loss = val_loss
            best_f1 = f1_best
            result = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_f1_best': f1_best,
                'val_f1_fixed': f1_fixed,
                'state_dict': model.state_dict(),
                'extractor_state_dict': modal_extractor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'lr_schedule': lr_schedule.__dict__,
                'config': config
            }
            save_path = f'./ckpt/{config.MODEL.NAME}/best_val_loss.pth'
            torch.save(result, save_path)
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")

            # Log best model to wandb
            try:
                wandb.log({
                    "best_val_loss": val_loss,
                    "best_f1_best": f1_best,
                    "best_f1_fixed": f1_fixed,
                    "best_model_epoch": epoch
                })
                wandb.save(save_path)
            except:
                pass  # Continue without wandb if it fails
            
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_f1_best': f1_best,
                'val_f1_fixed': f1_fixed,
                'state_dict': model.state_dict(),
                'extractor_state_dict': modal_extractor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'lr_schedule': lr_schedule.__dict__,
                'config': config
            }
            checkpoint_path = f'./ckpt/{config.MODEL.NAME}/checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch + 1}")
            
        writer.flush()

    # Save final model
    final_result = {
        'epoch': config.EPOCHS - 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_f1_best': f1_best,
        'val_f1_fixed': f1_fixed,
        'state_dict': model.state_dict(),
        'extractor_state_dict': modal_extractor.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'config': config
    }
    final_path = f'./ckpt/{config.MODEL.NAME}/final.pth'
    torch.save(final_result, final_path)
    logger.info(f"Training completed. Final model saved to {final_path}")
    logger.info(f"Best validation loss: {min_loss:.4f}, Best F1: {best_f1:.4f}")
    
    try:
        wandb.finish()
    except:
        pass  # Continue without wandb if it fails


if __name__ == "__main__":
    main()