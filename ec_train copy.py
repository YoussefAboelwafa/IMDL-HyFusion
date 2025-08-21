"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from common.utils import AverageMeter
from common.losses import TruForLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import logging
import torch
import torchvision.transforms.functional as TF

from data.datasets import MixDataset
from common.metrics import computeLocalizationMetrics
# from common.losses import edge_loss
from models.cmnext_conf import CMNeXtWithConf
from common.split_params import group_weight
from common.lr_schedule import WarmUpPolyLR
from models.modal_extract import ModalitiesExtractor
from configs.cmnext_init_cfg import _C as config, update_config
import pretty_errors
import cv2
import os.path as osp
import gc
import wandb
gc.collect()

def edge_loss(pred, target, weights=None):
    """
    Binary cross entropy loss for edge detection
    """
    if weights is None:
        weights = torch.ones_like(target)
    
    # Apply class weights
    from torch.nn import functional as F
    weights = weights.float()
    loss = F.binary_cross_entropy_with_logits(pred, target, weight=weights, reduction='mean')
    return loss

import warnings
warnings.filterwarnings("ignore", message="libpng warning: iCCP: known incorrect sRGB profile")
warnings.filterwarnings("ignore", message="libpng warning: iCCP: profile 'ICC profile': 'bTRC': ICC profile tag start not a multiple of 4")
warnings.filterwarnings("ignore", message="Corrupt EXIF data", module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")
#ignore all warnings
warnings.filterwarnings("ignore")
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
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
    parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')
    parser.add_argument('-train_bayar', '--train_bayar', action='store_true', help='finetune bayar conv')
    parser.add_argument('-exp', '--exp', type=str, default=None, help='Yaml experiment file')
    parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--ckpt', type=str, default='', help='Resume from checkpoint path')
    args = parser.parse_args()

    print(torch.cuda.is_available())
    config = update_config(config, args.exp)
    torch.cuda.empty_cache()
    gpu = args.gpu
    loglvl = getattr(logging, args.log.upper())
    logging.basicConfig(level=loglvl, format='%(message)s')

    device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
    np.set_printoptions(formatter={'float': '{: 7.3f}'.format})
    print(f"Device: {device}")
    torch.set_flush_denormal(True)
    if device != 'cpu':
        # cudnn setting
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED


    modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], config.MODEL.NP_WEIGHTS)
    if 'bayar' in config.MODEL.MODALS:
        modal_extractor.load_state_dict(torch.load('pretrained/modal_extractor/bayar_mhsa.pth',map_location=torch.device('cpu')), strict=False)
        if not args.train_bayar:
            modal_extractor.bayar.eval()
            for param in modal_extractor.bayar.parameters():
                param.requires_grad = False

    model = CMNeXtWithConf(config.MODEL)
    wandb.init(
        project="mmfusion",  # replace with your project name
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
    modal_extractor.to(device)
    model = model.to(device)

    train = MixDataset(config.DATASET.TRAIN,
                    config.DATASET.IMG_SIZE,
                    train=True,
                    class_weight=config.DATASET.CLASS_WEIGHTS)

    val = MixDataset(config.DATASET.VAL,
                    config.DATASET.IMG_SIZE,
                    train=False)

    logging.info(train.get_info())
    def worker_init_fn(worker_id):
        # Set different seed for each worker
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Modify DataLoader initialization
    train_loader = DataLoader(
        train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True  # Prevent irregular batch sizes
    )
    val_loader = DataLoader(val,
                            batch_size=1,
                            shuffle=False,
                            num_workers=config.WORKERS,
                            pin_memory=True,
                            persistent_workers=True,
                            drop_last=True
                            )

    criterion = TruForLoss(weights=train.class_weights.to(device), ignore_index=-1)

    os.makedirs('./ckpt/{}'.format(config.MODEL.NAME), exist_ok=True)
    logdir = './{}/{}'.format(config.LOG_DIR, config.MODEL.NAME)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter('./{}/{}'.format(config.LOG_DIR, config.MODEL.NAME))

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

    iters_per_epoch = len(train_loader)
    iters = 0
    max_iters = config.EPOCHS * iters_per_epoch
    min_loss = 100

    lr_schedule = WarmUpPolyLR(optimizer,
                            start_lr=config.LEARNING_RATE,
                            lr_power=config.POLY_POWER,
                            total_iters=max_iters,
                            warmup_steps=iters_per_epoch * config.WARMUP_EPOCHS)

    scaler = torch.cuda.amp.GradScaler()

    del params
    del cmnext_params
    del modal_extract_params
    gc.collect()
    torch.cuda.empty_cache()
    SAVE_FREQ = 100

    if args.ckpt and os.path.exists(args.ckpt):
        logging.info(f'Loading checkpoint from {args.ckpt}')
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        modal_extractor.load_state_dict(ckpt['extractor_state_dict']) 
        start_epoch = ckpt['epoch'] + 1
    else:
        start_epoch = 0
    def train_epoch(epoch, model, modal_extractor, train_loader, criterion, optimizer, scaler, writer, device, config):
        """Run one training epoch"""
        # Clear cache at start of epoch
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
            # check for NaN in images and masks
            images = images.to(device, non_blocking=True)
            masks = masks.squeeze(1).to(device, non_blocking=True)
            # del name  # Free memory if not needed
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                modals = modal_extractor(images)
                # check modals for NaN values
                images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                inp = [images_norm] + modals
                pred, edge, sem_map = model(inp)
                
                edge_gt = torch.zeros_like(edge)
                edge_gt[masks == 1] = 1
                
                edge_loss_val = edge_loss(edge, edge_gt)
                loss = criterion(pred, masks) / config.ACCUMULATE_ITERS
                loss += edge_loss_val * config.EDGE_LOSS_WEIGHT

                if (step + 1) % SAVE_FREQ == 0:
                    maps_dir = osp.join('./outputs', config.MODEL.NAME, f'step_{step}')
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

                
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5
            )
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            # Add proper weight decay
            scaler.scale(loss).backward()
            if config.WD > 0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.add_(param.data, alpha=config.WD)
            if ((step + 1) % config.ACCUMULATE_ITERS == 0) or (step + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Clear cache periodically
                if step % 100 == 0:
                    torch.cuda.empty_cache()
            
            

            avg_loss.update(loss.detach().item())
            edge_loss_avg.update(edge_loss_val.detach().item())

            curr_iters = epoch * iters_per_epoch + step
            # lr_schedule.step(cur_iter=curr_iters)
            # scheduler.step(loss.detach().item())
            wandb.log({
                    "train/step_loss": loss.detach().item(),
                    "train/edge_loss": edge_loss_val.detach().item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                }, step=curr_iters)
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
            del pred
            del edge
            del sem_map
            del loss
            torch.cuda.empty_cache()
            gc.collect()
        writer.add_scalar('Training Loss', avg_loss.average(), epoch)
        wandb.log({
            "train/epoch_loss": avg_loss.average(),
            "train/epoch_edge_loss": edge_loss_avg.average(),
            "epoch": epoch
        })

        return avg_loss.average()

    def validate_epoch(epoch, model, modal_extractor, val_loader, criterion, writer, device, config):
        """Run one validation epoch"""
        model.set_val()
        modal_extractor.set_val()
        
        val_loss_avg = AverageMeter()
        # edge_val_loss_avg = AverageMeter()
        f1 = []
        f1th = []
        
        pbar = tqdm(val_loader, desc=f'Validating Epoch {epoch + 1}/{config.EPOCHS}', unit='steps')
        
        for step, (images, _, masks, lab) in enumerate(pbar):
            # if any(torch.isnan(t).any() for t in images) or any(torch.isnan(t).any() for t in masks):
            #     logging.error("Modalities contain NaN values!")
            #     raise ValueError("Modalities contain NaN values")
            with torch.no_grad():
                images = images.to(device, non_blocking=True)
                masks = masks.squeeze(1).to(device, non_blocking=True)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    if any(torch.isnan(t).any() for t in images):
                        logging.error("Images contain NaN values!")
                        raise ValueError("Images contain NaN values")
                    modals = modal_extractor(images)
                    # check modals for NaN values
                    # if any(torch.isnan(t).any() for t in modals):
                    #     logging.error("Modalities contain NaN values!")
                    #     raise ValueError("Modalities contain NaN values")
                    images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    inp = [images_norm] + modals
                    # print(inp[0].shape)
                    pred, edge, sem_map = model(inp)
                    
                    # edge_gt = torch.zeros_like(edge)
                    # edge_gt[masks == 1] = 1
                    # edge_loss_val = edge_loss(edge, edge_gt)
                    val_loss = criterion(pred, masks)
                    # print(f"val loss: {val_loss.item()} ")
                val_loss_avg.update(val_loss.detach().item())
                # print("val_loss_avg: ", val_loss_avg.average())
                # edge_val_loss_avg.update(edge_loss_val.detach().item())

                gt = masks.squeeze().cpu().numpy()
                map = torch.nn.functional.softmax(pred, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
                F1_best, F1_th = computeLocalizationMetrics(map, gt)
                f1.append(F1_best)
                f1th.append(F1_th)
                del images
                del masks
                del modals
                del images_norm
                del pred
                del edge
                del sem_map
                # del val_loss
                gc.collect()
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
        wandb.log(metrics)
        return val_loss_avg.average(), np.nanmean(f1), np.nanmean(f1th)

    # Replace the training loop with:
    min_loss = 100
    for epoch in range(start_epoch, config.EPOCHS):
        train.shuffle()  # for balanced sampling


        # # Validation phase
        # val_loss, f1_best, f1_fixed = validate_epoch(epoch, model, modal_extractor, val_loader,
        #                                             criterion, writer, device, config)
        # Training phase
        train_loss = train_epoch(epoch, model, modal_extractor, train_loader, criterion, 
                                optimizer, scaler, writer, device, config)

        # print(val_loss, f1_best, f1_fixed)
        torch.cuda.empty_cache()
        gc.collect()
        # Validation phase
        val_loss, f1_best, f1_fixed = validate_epoch(epoch, model, modal_extractor, val_loader,
                                                    criterion, writer, device, config)
        
        
        # Save best model
        if val_loss < min_loss:
            min_loss = val_loss
            result = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_f1_best': f1_best,
                'val_f1_fixed': f1_fixed,
                'state_dict': model.state_dict(),
                'extractor_state_dict': modal_extractor.state_dict()
            }
            save_path = f'./ckpt/{config.MODEL.NAME}/best_val_loss.pth'
            torch.save(result, save_path)
            # print(ty)
            # Log best model to wandb
            wandb.log({
                "best_val_loss": val_loss,
                "best_f1_best": f1_best,
                "best_f1_fixed": f1_fixed,
                "best_model_epoch": epoch
            })
            wandb.save(save_path)

        writer.flush()

    # Save final model
    result = {
        'epoch': config.EPOCHS - 1,
        # 'val_loss': val_loss,
        # 'val_f1_best': f1_best,
        # 'val_f1_fixed': f1_fixed,
        'state_dict': model.state_dict(),
        'extractor_state_dict': modal_extractor.state_dict()
    }
    torch.save(result, f'./ckpt/{config.MODEL.NAME}/final.pth')
