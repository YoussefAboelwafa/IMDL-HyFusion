"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
Enhanced for better architecture
"""
import os
import torch
import numpy as np
import logging
import gc
import cv2
import os.path as osp
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import wandb

from common.utils import AverageMeter
from common.metrics import computeLocalizationMetrics


class Trainer:
    """
    Trainer class for handling the training and validation process
    """
    def __init__(self, 
                 model, 
                 modal_extractor, 
                 criterion, 
                 optimizer, 
                 scaler, 
                 config, 
                 device, 
                 train_bayar=False):
        """
        Initialize the trainer

        Args:
            model: The main model
            modal_extractor: The modalities extractor
            criterion: Loss function
            optimizer: Optimizer
            scaler: Gradient scaler for mixed precision
            config: Configuration object
            device: Device to use (cuda/cpu)
            train_bayar: Whether to train the bayar filter
        """
        self.model = model
        self.modal_extractor = modal_extractor
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.device = device
        self.train_bayar = train_bayar

        # Create directories for checkpoints and logs
        os.makedirs(f'./ckpt/{config.MODEL.NAME}', exist_ok=True)
        self.logdir = f'./{config.LOG_DIR}/{config.MODEL.NAME}'
        os.makedirs(self.logdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)

        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )

        self.min_loss = float('inf')
        self.save_freq = 100  # Save frequency for visualizations

    def train_epoch(self, epoch, train_loader):
        """
        Run one training epoch

        Args:
            epoch: Current epoch number
            train_loader: DataLoader for training data

        Returns:
            Average loss for the epoch
        """
        # Clear cache at start of epoch
        torch.cuda.empty_cache()

        self.model.set_train()
        if self.train_bayar:
            self.modal_extractor.set_train()

        avg_loss = AverageMeter()
        edge_loss_avg = AverageMeter()
        iters_per_epoch = len(train_loader)

        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{self.config.EPOCHS}', unit='steps')
        self.optimizer.zero_grad(set_to_none=True)

        for step, (images, name, masks, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.squeeze(1).to(self.device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                modals = self.modal_extractor(images)
                images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                inp = [images_norm] + modals

                pred, edge, sem_map = self.model(inp)

                edge_gt = torch.zeros_like(edge)
                edge_gt[masks == 1] = 1

                edge_loss_val = self._edge_loss(edge, edge_gt)
                loss = self.criterion(pred, masks) / self.config.ACCUMULATE_ITERS
                loss += edge_loss_val * self.config.EDGE_LOSS_WEIGHT

                if (step + 1) % self.save_freq == 0:
                    self._save_visualizations(step, name, images, masks, pred, edge, sem_map)

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Apply weight decay if configured
            if self.config.WD > 0:
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.add_(param.data, alpha=self.config.WD)

            # Update weights with gradient accumulation
            if ((step + 1) % self.config.ACCUMULATE_ITERS == 0) or (step + 1 == len(train_loader)):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # Clear cache less frequently (every 500 steps instead of 100)
                # This reduces overhead from memory management operations
                if step % 500 == 0:
                    torch.cuda.empty_cache()

            avg_loss.update(loss.detach().item())
            edge_loss_avg.update(edge_loss_val.detach().item())

            curr_iters = epoch * iters_per_epoch + step

            # Update learning rate
            self.scheduler.step(loss.detach().item())

            # Log metrics (but less frequently to reduce overhead)
            if step % 10 == 0:  # Log every 10 steps instead of every step
                wandb.log({
                    "train/step_loss": loss.detach().item(),
                    "train/edge_loss": edge_loss_val.detach().item(),
                    "train/learning_rate": self.optimizer.param_groups[0]['lr']
                }, step=curr_iters)
                self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], curr_iters)

            if step == 0:
                self._log_images(epoch, images, masks, pred)

            pbar.set_postfix({"last_loss": loss.detach().item(), "epoch_loss": avg_loss.average()})

            # Clear intermediate tensors but don't call gc.collect() on every step
            # This reduces overhead from frequent garbage collection
            del pred, edge, sem_map, loss

            # Only perform memory cleanup every 50 steps
            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        # Log epoch metrics
        self.writer.add_scalar('Training Loss', avg_loss.average(), epoch)
        wandb.log({
            "train/epoch_loss": avg_loss.average(),
            "train/epoch_edge_loss": edge_loss_avg.average(),
            "epoch": epoch
        })

        return avg_loss.average()

    def validate_epoch(self, epoch, val_loader):
        """
        Run one validation epoch

        Args:
            epoch: Current epoch number
            val_loader: DataLoader for validation data

        Returns:
            Tuple of (validation loss, F1 best, F1 fixed)
        """
        self.model.set_val()
        self.modal_extractor.set_val()

        val_loss_avg = AverageMeter()
        f1 = []
        f1th = []

        pbar = tqdm(val_loader, desc=f'Validating Epoch {epoch + 1}/{self.config.EPOCHS}', unit='steps')

        for step, (images, _, masks, lab) in enumerate(pbar):
            with torch.no_grad():
                images = images.to(self.device, non_blocking=True)
                masks = masks.squeeze(1).to(self.device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    modals = self.modal_extractor(images)
                    images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    inp = [images_norm] + modals

                    pred, edge, sem_map = self.model(inp)
                    val_loss = self.criterion(pred, masks)

                val_loss_avg.update(val_loss.detach().item())

                gt = masks.squeeze().cpu().numpy()
                map_pred = torch.nn.functional.softmax(pred, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
                F1_best, F1_th = computeLocalizationMetrics(map_pred, gt)
                f1.append(F1_best)
                f1th.append(F1_th)

                # Clean up memory
                del images, masks, modals, images_norm, pred, edge, sem_map, val_loss
                gc.collect()
                if step % 50 == 0:
                    torch.cuda.empty_cache()

        # Log validation metrics
        self.writer.add_scalar('Val Loss', val_loss_avg.average(), epoch)
        self.writer.add_scalar('Val F1 best', np.nanmean(f1), epoch)
        self.writer.add_scalar('Val F1 fixed', np.nanmean(f1th), epoch)

        metrics = {
            "val/loss": val_loss_avg.average(),
            "val/f1_best": np.nanmean(f1),
            "val/f1_fixed": np.nanmean(f1th),
            "epoch": epoch
        }
        wandb.log(metrics)

        return val_loss_avg.average(), np.nanmean(f1), np.nanmean(f1th)

    def save_checkpoint(self, epoch, val_loss, f1_best, f1_fixed, is_best=False):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            f1_best: Best F1 score
            f1_fixed: Fixed threshold F1 score
            is_best: Whether this is the best model so far
        """
        result = {
            'epoch': epoch,
            'val_loss': val_loss,
            'val_f1_best': f1_best,
            'val_f1_fixed': f1_fixed,
            'state_dict': self.model.state_dict(),
            'extractor_state_dict': self.modal_extractor.state_dict()
        }

        if is_best:
            save_path = f'./ckpt/{self.config.MODEL.NAME}/best_val_loss.pth'
            torch.save(result, save_path)

            # Log best model to wandb
            wandb.log({
                "best_val_loss": val_loss,
                "best_f1_best": f1_best,
                "best_f1_fixed": f1_fixed,
                "best_model_epoch": epoch
            })
            wandb.save(save_path)
        else:
            # Save regular checkpoint
            torch.save(result, f'./ckpt/{self.config.MODEL.NAME}/checkpoint_epoch_{epoch}.pth')

    def save_final_model(self, epoch):
        """
        Save the final model

        Args:
            epoch: Final epoch number
        """
        result = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'extractor_state_dict': self.modal_extractor.state_dict()
        }
        torch.save(result, f'./ckpt/{self.config.MODEL.NAME}/final.pth')

    def _edge_loss(self, pred, target, weights=None):
        """
        Binary cross entropy loss for edge detection

        Args:
            pred: Predicted edge map
            target: Target edge map
            weights: Optional weights for loss calculation

        Returns:
            Edge loss value
        """
        if weights is None:
            weights = torch.ones_like(target)

        weights = weights.float()
        loss = F.binary_cross_entropy_with_logits(pred, target, weight=weights, reduction='mean')
        return loss

    def _save_visualizations(self, step, name, images, masks, pred, edge, sem_map):
        """
        Save visualization maps

        Args:
            step: Current step
            name: Image names
            images: Input images
            masks: Ground truth masks
            pred: Model predictions
            edge: Edge predictions
            sem_map: Semantic maps
        """
        maps_dir = osp.join('./outputs', self.config.MODEL.NAME, f'step_{step}')
        os.makedirs(maps_dir, exist_ok=True)

        # Convert predictions to numpy and process for visualization
        pred_softmax = torch.nn.functional.softmax(pred, dim=1)
        pred_prob = pred_softmax[:, 1, :, :].cpu().detach().numpy()  # Get probability for class 1
        edge_prob = torch.sigmoid(edge).cpu().detach().numpy()
        sem_map_np = sem_map.cpu().detach().numpy()

        # Save prediction maps for each image in batch
        for idx, img_name in enumerate(name):
            # Save probability map
            print(f'Saving maps for {img_name}')
            prob_map = (pred_prob[idx] * 255).astype(np.uint8)
            cv2.imwrite(osp.join(maps_dir, f'{img_name}_prob.png'), prob_map)

            # Save binary map with threshold 0.5
            binary_map = (pred_prob[idx] > 0.5).astype(np.uint8) * 255
            cv2.imwrite(osp.join(maps_dir, f'{img_name}_binary.png'), binary_map)

            # Save edge map
            edge_map_vis = (edge_prob[idx] * 255).astype(np.uint8)
            cv2.imwrite(osp.join(maps_dir, f'{img_name}_edge.png'), edge_map_vis)

            # Save ground truth
            gt = (masks[idx].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(osp.join(maps_dir, f'{img_name}_gt.png'), gt)

            # Save semantic map
            sem_map_vis = (sem_map_np[idx] * 255).astype(np.uint8)
            cv2.imwrite(osp.join(maps_dir, f'{img_name}_sem.png'), sem_map_vis)

    def _log_images(self, epoch, images, masks, pred):
        """
        Log images to tensorboard

        Args:
            epoch: Current epoch
            images: Input images
            masks: Ground truth masks
            pred: Model predictions
        """
        maps = torch.nn.functional.softmax(pred, dim=1)[:, 1, :, :]
        self.writer.add_images('Images-Masks-Preds',
                            torch.cat((
                                images,
                                torch.tile(masks.unsqueeze(1), (1, 3, 1, 1)),
                                torch.tile(maps.unsqueeze(1), (1, 3, 1, 1))), -2),
                            epoch)
