"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from models.base import BaseModel
from models.heads import SegFormerHead
import logging
from models.modules.esb import ESB
from models.modules.segmentation import get_semantic_map
from models.modules.dual_atten import DAHead
# 1. Implement gradient checkpointing
from torch.utils.checkpoint import checkpoint



class ResidualFusion(nn.Module):
    """A proper module for residual fusion to replace lambda functions"""
    def __init__(self, fusion_module, channels):
        super(ResidualFusion, self).__init__()
        self.fusion_module = fusion_module
        self.channels = channels
    
    def forward(self, x):
        return self.fusion_module(x) + x[:, :self.channels, :, :]


class CMNeXtWithConf(BaseModel):
    def __init__(self, cfg=None) -> None:
        backbone = cfg.BACKBONE
        num_classes = cfg.NUM_CLASSES
        modals = cfg.MODALS
        logging.info(f'Training phase: {cfg.TRAIN_PHASE}')
        logging.info(f'Loading Model: {cfg.NAME}, backbone: {cfg.BACKBONE}')
        super().__init__(backbone, num_classes, modals)

        # Get backbone channels
        channels = self.backbone.channels
        hidden_dim = 256 if 'B0' in backbone or 'B1' in backbone else 512

        # Initialize heads
        self.decode_head = SegFormerHead(channels, hidden_dim, num_classes)
        self.conf_head = SegFormerHead(channels, hidden_dim, 1)

        # Dual attention head
        self.da_head = DAHead(in_channels=channels, nclass = num_classes)

        # Feature Pyramid Network (FPN) style enhancement
        self.fpn_convs = nn.ModuleList()
        self.fpn_laterals = nn.ModuleList()

        # Create lateral connections and output convs for FPN
        for i, c in enumerate(channels):
            # Lateral connections (reduce channel dimensions)
            self.fpn_laterals.append(nn.Conv2d(c, hidden_dim, kernel_size=1))

            # Output convolutions (3x3 conv to smooth features)
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(hidden_dim, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            ))

        # Edge detection branch
        self.edge_branch = ESB(2, sobel=True)
        self.edge_channel_reduce = nn.Conv2d(2048, 1, kernel_size=1)

        # Enhanced feature fusion with residual connections
        self.adjust_layers = nn.ModuleList()
        for c in [64, 128, 320, 512]:
            # Create a more complex fusion module with residual connection
            fusion_module = nn.Sequential(
                # First branch: direct 1x1 conv (main path)
                nn.Conv2d(c + 1, c, kernel_size=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),

                # Second branch: deeper processing
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
            )

            # Residual connection wrapper
            
            residual_fusion = ResidualFusion(fusion_module, c)

            self.adjust_layers.append(residual_fusion)

        # Enhanced detection head with more layers and wider dimensions
        if cfg.DETECTION == 'confpool':
            self.detection = nn.Sequential(
                nn.Linear(8, 256),  # Wider first layer
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),  # Less aggressive dropout
                nn.BatchNorm1d(256),
                nn.Linear(256, 256),  # Additional layer
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.BatchNorm1d(256),
                nn.Linear(256, 128),  # Additional layer
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.BatchNorm1d(128),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        self.train_phase = cfg.TRAIN_PHASE
        assert self.train_phase in ['localization', 'detection']

        # Initialize weights and load pretrained
        self.apply(self._init_weights)
        self.init_pretrained(cfg.PRETRAINED, backbone)

        # Freeze parameters for detection phase
        if self.train_phase == 'detection':
            self._freeze_localization_params()

    def _freeze_localization_params(self):
        """Freeze backbone and localization head parameters"""
        for module in [self.backbone, self.decode_head]:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def set_train(self):
        if self.train_phase == 'localization':
            self.backbone.train()
            self.decode_head.train()
            self.edge_branch.train()
            self.da_head.train()
        elif self.train_phase == 'detection':
            self.conf_head.train()
            self.detection.train()
            self.backbone.eval()
            self.decode_head.train()
        else:
            raise ValueError(f'Train phase {self.train_phase} not recognized!')

    def set_val(self):
        if self.train_phase == 'localization':
            self.backbone.eval()
            self.decode_head.eval()
        elif self.train_phase == 'detection':
            self.conf_head.eval()
            self.detection.eval()
        else:
            raise ValueError(f'Train phase {self.train_phase} not recognized!')

    def forward(self, x: list, masks: list = None):
        # Memory optimization: Use gradient checkpointing during training, disable during inference
        use_checkpointing = self.training and hasattr(self, 'use_gradient_checkpointing') and self.use_gradient_checkpointing
        
        # get edge map with memory optimization
        edges, edge_map = self.edge_branch(x[0])  # edge_map: [1, 1, 2048, 32, 32]
        edge_map = edge_map.squeeze(2)  # Remove extra dimension to get [B, C, H, W]
        # Reduce edge map channels to 1
        edge_map = self.edge_channel_reduce(edge_map)  # Now shape: [B, 1, H, W]

        # Use backbone with optional gradient checkpointing
        if use_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            if masks is not None:
                y = checkpoint(create_custom_forward(self.backbone), x, masks)
            else:
                y = checkpoint(create_custom_forward(self.backbone), x)
        else:
            if masks is not None:
                y = self.backbone(x, masks)
            else:
                y = self.backbone(x)

        # Memory-efficient feature processing
        enhanced_features = []
        
        # Process features one at a time to reduce peak memory usage
        for idx, feat in enumerate(y):
            # Resize edge map on-demand to save memory
            edge_map_resized = F.interpolate(
                edge_map, 
                size=feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )

            # Concatenate along channel dimension
            enhanced_feat = torch.cat([feat, edge_map_resized], dim=1)

            # Apply enhanced fusion module with residual connection
            enhanced_feat = self.adjust_layers[idx](enhanced_feat)

            # Apply ReLU after residual connection (use in-place operation for memory efficiency)
            enhanced_feat = F.relu(enhanced_feat, inplace=True)

            enhanced_features.append(enhanced_feat)
            
            # Clean up intermediate tensors
            del edge_map_resized, feat
            if idx < len(y) - 1:  # Don't delete on last iteration as we still need it
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Clean up backbone features to free memory
        del y
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Apply dual attention head for enhanced feature extraction
        enhanced_features = self.da_head(enhanced_features)

        # Memory-efficient FPN processing
        fpn_features = []

        # Process laterals one at a time to reduce memory
        laterals = []
        for i, feat in enumerate(enhanced_features):
            lateral = self.fpn_laterals[i](feat)
            laterals.append(lateral)

        # Clean up enhanced features after lateral processing
        del enhanced_features
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Top-down pathway with memory optimization
        prev_features = laterals[-1]
        fpn_out = self.fpn_convs[-1](prev_features)
        fpn_features.append(fpn_out)

        # Process from high to low resolution with immediate cleanup
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample higher level features
            upsample = F.interpolate(
                prev_features, 
                size=laterals[i].shape[2:],
                mode='bilinear', 
                align_corners=False
            )

            # Add lateral connection (skip connection)
            prev_features = laterals[i] + upsample
            
            # Clean up upsample tensor immediately
            del upsample

            # Apply 3x3 conv to smooth features
            fpn_out = self.fpn_convs[i](prev_features)
            fpn_features.insert(0, fpn_out)
            
            # Clean up lateral tensor after use
            del laterals[i]

        # Clean up remaining tensors
        del laterals
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # Pass through decode head with FPN enhanced features
        out = self.decode_head(fpn_features)
        out = F.interpolate(out, size=x[0].shape[2:], mode='bilinear', align_corners=False)

        if self.train_phase == 'detection':
            # Use FPN-enhanced features for confidence prediction as well
            conf = self.conf_head(fpn_features)
            conf = F.interpolate(conf, size=x[0].shape[2:], mode='bilinear', align_corners=False)

            # Clean up fpn_features after use
            del fpn_features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            from .layer_utils import weighted_statistics_pooling
            f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
            f2 = weighted_statistics_pooling(out[:, 1:2, :, :] - out[:, 0:1, :, :], F.logsigmoid(conf)).view(
                out.shape[0], -1)

            # Pass through enhanced detection head
            det = self.detection(torch.cat((f1, f2), -1))
            
            # Clean up intermediate tensors
            del f1, f2
            
            return out, conf, det
        else:
            # Clean up fpn_features after use in localization phase
            del fpn_features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return out, edges, None


    def init_pretrained(self, pretrained: str = None, backbone: str = None) -> None:
        if pretrained:
            logging.info('Loading pretrained module: {}'.format(pretrained))
            if self.backbone.num_modals > 0:
                load_dualpath_model(self.backbone, pretrained, backbone)
            else:
                try:
                    # Try to load as a full model checkpoint first
                    result = self.load_checkpoint(pretrained, strict=False)
                    logging.info(f"Loaded pretrained model: {len(result['loaded_keys'])} parameters loaded, "
                               f"{len(result['missing_keys'])} missing, {len(result['unexpected_keys'])} unexpected")
                except Exception as e:
                    # Fallback to loading only backbone
                    logging.warning(f"Failed to load as full model checkpoint: {e}. Trying backbone only...")
                    checkpoint = torch.load(pretrained, map_location='cpu')
                    if 'state_dict' in checkpoint.keys():
                        checkpoint = checkpoint['state_dict']
                    if 'model' in checkpoint.keys():
                        checkpoint = checkpoint['model']
                    msg = self.backbone.load_state_dict(checkpoint, strict=False)
                    print(msg)

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False, map_location='cpu'):
        """
        Load checkpoint with all available components and initialize weights for remaining components.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            strict (bool): Whether to strictly enforce that the keys in checkpoint match the keys in model
            map_location: Device to map the checkpoint to
            
        Returns:
            dict: Loading information including missing and unexpected keys
        """
        logging.info(f'Loading checkpoint from: {checkpoint_path}')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Get current model state dict
        model_state_dict = self.state_dict()
        
        # Track loading statistics
        loaded_keys = []
        missing_keys = []
        unexpected_keys = []
        size_mismatched_keys = []
        
        # Create a new state dict with only compatible keys
        compatible_state_dict = {}
        
        for key, param in state_dict.items():
            if key in model_state_dict:
                if param.shape == model_state_dict[key].shape:
                    compatible_state_dict[key] = param
                    loaded_keys.append(key)
                else:
                    size_mismatched_keys.append(f"{key}: checkpoint {param.shape} vs model {model_state_dict[key].shape}")
                    logging.warning(f"Size mismatch for {key}: checkpoint {param.shape} vs model {model_state_dict[key].shape}")
            else:
                unexpected_keys.append(key)
        
        # Find missing keys
        for key in model_state_dict.keys():
            if key not in compatible_state_dict:
                missing_keys.append(key)
        
        # Load compatible weights
        loading_result = self.load_state_dict(compatible_state_dict, strict=False)
        
        # Initialize weights for missing components
        if missing_keys:
            logging.info(f"Initializing weights for {len(missing_keys)} missing components...")
            
            # Create a temporary model to get properly initialized weights
            for key in missing_keys:
                module_names = key.split('.')
                module = self
                
                # Navigate to the parent module
                for name in module_names[:-1]:
                    if hasattr(module, name):
                        module = getattr(module, name)
                    else:
                        break
                else:
                    # Get the parameter name
                    param_name = module_names[-1]
                    if hasattr(module, param_name):
                        param = getattr(module, param_name)
                        if isinstance(param, nn.Parameter):
                            # Initialize the parameter based on its type
                            if len(param.shape) >= 2:  # Weight matrix
                                if 'weight' in param_name:
                                    if len(param.shape) == 2:  # Linear layer
                                        nn.init.trunc_normal_(param, std=0.02)
                                    else:  # Conv layer
                                        nn.init.kaiming_normal_(param)
                                elif 'bias' in param_name:
                                    nn.init.zeros_(param)
                            else:  # Bias or 1D parameter
                                if 'weight' in param_name:
                                    nn.init.ones_(param)
                                else:
                                    nn.init.zeros_(param)
                        
                            logging.debug(f"Initialized {key} with shape {param.shape}")
        
        # Log loading summary
        logging.info(f"Checkpoint loading summary:")
        logging.info(f"  - Loaded: {len(loaded_keys)} parameters")
        logging.info(f"  - Missing: {len(missing_keys)} parameters")
        logging.info(f"  - Unexpected: {len(unexpected_keys)} parameters")
        logging.info(f"  - Size mismatched: {len(size_mismatched_keys)} parameters")
        
        if missing_keys and len(missing_keys) <= 20:  # Show details if not too many
            logging.info(f"Missing keys: {missing_keys}")
        elif missing_keys:
            logging.info(f"Missing keys (showing first 20): {missing_keys[:20]}")
            
        if unexpected_keys and len(unexpected_keys) <= 20:
            logging.info(f"Unexpected keys: {unexpected_keys}")
        elif unexpected_keys:
            logging.info(f"Unexpected keys (showing first 20): {unexpected_keys[:20]}")
            
        if size_mismatched_keys:
            logging.warning(f"Size mismatched keys: {size_mismatched_keys}")
        
        # Return comprehensive loading information
        return {
            'loaded_keys': loaded_keys,
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'size_mismatched_keys': size_mismatched_keys,
            'loading_result': loading_result,
            'checkpoint_info': {k: v for k, v in checkpoint.items() if k != 'state_dict' and k != 'model'}
        }

    def load_training_checkpoint(self, checkpoint_path: str, modal_extractor=None, optimizer=None, scaler=None, lr_schedule=None, map_location='cpu'):
        """
        Load a training checkpoint with model, modal_extractor, optimizer, and other training components.
        
        Args:
            checkpoint_path (str): Path to the training checkpoint file
            modal_extractor: Modal extractor model to load state into
            optimizer: Optimizer to load state into
            scaler: GradScaler to load state into  
            lr_schedule: Learning rate scheduler to load state into
            map_location: Device to map the checkpoint to
            
        Returns:
            dict: Checkpoint information including epoch, losses, and loading results
        """
        logging.info(f'Loading training checkpoint from: {checkpoint_path}')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        results = {}
        
        # Load model state dict
        if 'state_dict' in checkpoint:
            model_result = self.load_checkpoint_state_dict(checkpoint['state_dict'])
            results['model_loading'] = model_result
            logging.info("Model state dict loaded successfully")
        else:
            logging.warning("No 'state_dict' found in checkpoint")
        
        # Load modal extractor state dict
        if modal_extractor is not None and 'extractor_state_dict' in checkpoint:
            try:
                extractor_result = modal_extractor.load_state_dict(checkpoint['extractor_state_dict'], strict=False)
                results['extractor_loading'] = extractor_result
                logging.info("Modal extractor state dict loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load modal extractor state dict: {e}")
                results['extractor_loading'] = {'error': str(e)}
        elif modal_extractor is not None:
            logging.warning("Modal extractor provided but no 'extractor_state_dict' found in checkpoint")
        
        # Load optimizer state dict
        if optimizer is not None and 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                results['optimizer_loaded'] = True
                logging.info("Optimizer state dict loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load optimizer state dict: {e}")
                results['optimizer_loaded'] = False
                results['optimizer_error'] = str(e)
        elif optimizer is not None:
            logging.warning("Optimizer provided but no 'optimizer' found in checkpoint")
        
        # Load scaler state dict
        if scaler is not None and 'scaler' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler'])
                results['scaler_loaded'] = True
                logging.info("Scaler state dict loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load scaler state dict: {e}")
                results['scaler_loaded'] = False
                results['scaler_error'] = str(e)
        elif scaler is not None:
            logging.warning("Scaler provided but no 'scaler' found in checkpoint")
        
        # Load learning rate schedule state
        if lr_schedule is not None and 'lr_schedule' in checkpoint:
            try:
                # Update lr_schedule attributes from checkpoint
                lr_schedule.__dict__.update(checkpoint['lr_schedule'])
                results['lr_schedule_loaded'] = True
                logging.info("Learning rate schedule state loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load learning rate schedule state: {e}")
                results['lr_schedule_loaded'] = False
                results['lr_schedule_error'] = str(e)
        elif lr_schedule is not None:
            logging.warning("Learning rate schedule provided but no 'lr_schedule' found in checkpoint")
        
        # Extract training metadata
        training_info = {}
        for key in ['epoch', 'train_loss', 'val_loss', 'val_f1_best', 'val_f1_fixed']:
            if key in checkpoint:
                training_info[key] = checkpoint[key]
        
        results['training_info'] = training_info
        results['config'] = checkpoint.get('config', None)
        
        # Log training info
        if training_info:
            logging.info(f"Training checkpoint info: {training_info}")
        
        return results

    def load_checkpoint_state_dict(self, state_dict, strict=False):
        """
        Load only the model state dict with intelligent handling of missing/extra keys.
        
        Args:
            state_dict: State dictionary to load
            strict: Whether to strictly match keys
            
        Returns:
            dict: Loading result information
        """
        # Get current model state dict
        model_state_dict = self.state_dict()
        
        # Track loading statistics
        loaded_keys = []
        missing_keys = []
        unexpected_keys = []
        size_mismatched_keys = []
        
        # Create a new state dict with only compatible keys
        compatible_state_dict = {}
        
        for key, param in state_dict.items():
            if key in model_state_dict:
                if param.shape == model_state_dict[key].shape:
                    compatible_state_dict[key] = param
                    loaded_keys.append(key)
                else:
                    size_mismatched_keys.append(f"{key}: checkpoint {param.shape} vs model {model_state_dict[key].shape}")
            else:
                unexpected_keys.append(key)
        
        # Find missing keys
        for key in model_state_dict.keys():
            if key not in compatible_state_dict:
                missing_keys.append(key)
        
        # Load compatible weights
        loading_result = self.load_state_dict(compatible_state_dict, strict=False)
        
        # Initialize missing components if any
        if missing_keys:
            self._initialize_missing_components(missing_keys)
        
        return {
            'loaded_keys': loaded_keys,
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'size_mismatched_keys': size_mismatched_keys,
            'loading_result': loading_result
        }

    def _initialize_missing_components(self, missing_keys):
        """Initialize weights for missing components"""
        logging.info(f"Initializing weights for {len(missing_keys)} missing components...")
        
        # Group missing keys by module
        modules_to_init = set()
        for key in missing_keys:
            module_path = '.'.join(key.split('.')[:-1])
            if module_path:
                modules_to_init.add(module_path)
        
        # Initialize each module
        for module_path in modules_to_init:
            module = self
            try:
                for name in module_path.split('.'):
                    module = getattr(module, name)
                
                # Apply weight initialization to the module
                module.apply(self._initialize_module_weights)
                logging.debug(f"Initialized module: {module_path}")
                
            except AttributeError:
                logging.warning(f"Could not find module: {module_path}")

    def _initialize_module_weights(self, module: nn.Module):
        """
        Initialize weights for a specific module using the same strategy as _init_weights
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            if hasattr(module, 'eps'):
                module.eps = 0.001
            if hasattr(module, 'momentum'):
                module.momentum = 0.1

    def enable_gradient_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing for memory efficiency during training"""
        self.use_gradient_checkpointing = enable
        logging.info(f"Gradient checkpointing {'enabled' if enable else 'disabled'}")

    def set_inference_mode(self, enable=True):
        """
        Set inference mode for maximum memory efficiency.
        This will disable gradient computation and optimize for inference.
        """
        if enable:
            self.eval()
            torch.set_grad_enabled(False)
            logging.info("Inference mode enabled - gradients disabled for memory efficiency")
        else:
            torch.set_grad_enabled(True)
            logging.info("Inference mode disabled - gradients enabled")

    @torch.no_grad()
    def forward_inference(self, x: list, masks: list = None, tile_size=None, overlap=0.1):
        """
        Memory-efficient inference with optional tiling for very large images.
        
        Args:
            x: Input images list
            masks: Optional masks
            tile_size: If provided, split input into tiles of this size for processing
            overlap: Overlap ratio between tiles (0.0 to 1.0)
        
        Returns:
            Model outputs with reduced memory footprint
        """
        original_training = self.training
        self.eval()
        
        try:
            if tile_size is not None:
                return self._forward_with_tiling(x, masks, tile_size, overlap)
            else:
                return self._forward_with_memory_optimization(x, masks)
        finally:
            self.train(original_training)

    def _forward_with_memory_optimization(self, x: list, masks: list = None):
        """Forward pass with aggressive memory optimization"""
        # Process with torch.no_grad for inference
        with torch.no_grad():
            # Get edge map with immediate cleanup
            edges, edge_map = self.edge_branch(x[0])
            edge_map = edge_map.squeeze(2)
            edge_map = self.edge_channel_reduce(edge_map)
            
            # Clear edge computation intermediates
            del edges  # Don't need edges output for inference
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Backbone forward
            if masks is not None:
                y = self.backbone(x, masks)
            else:
                y = self.backbone(x)

            # Process features with minimal memory footprint
            enhanced_features = []
            
            for idx, feat in enumerate(y):
                # Process one feature at a time
                edge_resized = F.interpolate(
                    edge_map, size=feat.shape[2:], mode='bilinear', align_corners=False
                )
                enhanced_feat = torch.cat([feat, edge_resized], dim=1)
                enhanced_feat = self.adjust_layers[idx](enhanced_feat)
                enhanced_feat = F.relu(enhanced_feat, inplace=True)
                enhanced_features.append(enhanced_feat)
                
                # Immediate cleanup
                del edge_resized, feat
                
            del y, edge_map  # Clean up backbone outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # DA head processing
            enhanced_features = self.da_head(enhanced_features)

            # FPN with memory optimization
            fpn_features = self._process_fpn_memory_efficient(enhanced_features)
            del enhanced_features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Final prediction
            out = self.decode_head(fpn_features)
            out = F.interpolate(out, size=x[0].shape[2:], mode='bilinear', align_corners=False)

            if self.train_phase == 'detection':
                conf = self.conf_head(fpn_features)
                conf = F.interpolate(conf, size=x[0].shape[2:], mode='bilinear', align_corners=False)
                del fpn_features
                
                from .layer_utils import weighted_statistics_pooling
                f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
                f2 = weighted_statistics_pooling(out[:, 1:2, :, :] - out[:, 0:1, :, :], F.logsigmoid(conf)).view(
                    out.shape[0], -1)
                det = self.detection(torch.cat((f1, f2), -1))
                del f1, f2
                
                return out, conf, det
            else:
                del fpn_features
                return out, None, None

    def _process_fpn_memory_efficient(self, enhanced_features):
        """Process FPN with minimal memory usage"""
        fpn_features = []
        laterals = []
        
        # Process laterals one by one
        for i, feat in enumerate(enhanced_features):
            lateral = self.fpn_laterals[i](feat)
            laterals.append(lateral)

        # Top-down processing with immediate cleanup
        prev_features = laterals[-1]
        fpn_out = self.fpn_convs[-1](prev_features)
        fpn_features.append(fpn_out)

        for i in range(len(laterals) - 2, -1, -1):
            upsample = F.interpolate(
                prev_features, size=laterals[i].shape[2:], mode='bilinear', align_corners=False
            )
            prev_features = laterals[i] + upsample
            del upsample, laterals[i]  # Immediate cleanup
            
            fpn_out = self.fpn_convs[i](prev_features)
            fpn_features.insert(0, fpn_out)

        del laterals
        return fpn_features

    def _forward_with_tiling(self, x: list, masks: list = None, tile_size=512, overlap=0.1):
        """
        Process very large images by splitting into tiles.
        Useful for high-resolution inference when memory is limited.
        """
        # This is a placeholder for tiled inference implementation
        # For now, fall back to regular forward
        logging.warning("Tiled inference not yet implemented, using regular forward")
        return self._forward_with_memory_optimization(x, masks)

    def get_memory_usage(self):
        """Get current GPU memory usage if available"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
        else:
            return "CUDA not available"

    def optimize_for_inference(self):
        """Apply various optimizations for inference"""
        self.eval()
        
        # Fuse batch norm layers if possible
        try:
            torch.jit.optimize_for_inference(self)
            logging.info("JIT optimization applied")
        except:
            logging.warning("JIT optimization failed, continuing without it")
        
        # Set to inference mode
        self.set_inference_mode(True)
        
        logging.info("Model optimized for inference")

def load_dualpath_model(model, model_file, backbone):
    extra_pretrained = model_file if 'MHSA' in backbone else None
    if isinstance(extra_pretrained, str):
        raw_state_dict_ext = torch.load(extra_pretrained, map_location=torch.device('cpu'))
        if 'state_dict' in raw_state_dict_ext.keys():
            raw_state_dict_ext = raw_state_dict_ext['state_dict']
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v

    if isinstance(extra_pretrained, str):
        for k, v in raw_state_dict_ext.items():
            if k.find('patch_embed1.proj') >= 0:
                state_dict[k.replace('patch_embed1.proj', 'extra_downsample_layers.0.proj.module')] = v
            if k.find('patch_embed2.proj') >= 0:
                state_dict[k.replace('patch_embed2.proj', 'extra_downsample_layers.1.proj.module')] = v
            if k.find('patch_embed3.proj') >= 0:
                state_dict[k.replace('patch_embed3.proj', 'extra_downsample_layers.2.proj.module')] = v
            if k.find('patch_embed4.proj') >= 0:
                state_dict[k.replace('patch_embed4.proj', 'extra_downsample_layers.3.proj.module')] = v

            if k.find('patch_embed1.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed1.norm', 'extra_downsample_layers.0.norm.ln_{}'.format(i))] = v
            if k.find('patch_embed2.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed2.norm', 'extra_downsample_layers.1.norm.ln_{}'.format(i))] = v
            if k.find('patch_embed3.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed3.norm', 'extra_downsample_layers.2.norm.ln_{}'.format(i))] = v
            if k.find('patch_embed4.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed4.norm', 'extra_downsample_layers.3.norm.ln_{}'.format(i))] = v
            elif k.find('block') >= 0:
                state_dict[k.replace('block', 'extra_block')] = v
            elif k.find('norm') >= 0:
                state_dict[k.replace('norm', 'extra_norm')] = v

    msg = model.load_state_dict(state_dict, strict=False)
    del state_dict


if __name__ == '__main__':
    from configs.cmnext_init_cfg import _C as cfg
    logging.basicConfig(level=getattr(logging, 'INFO'))

    model = CMNeXtWithConf(cfg.MODEL)
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024) * 2,
         torch.ones(1, 3, 1024, 1024) * 3]
    y = model(x)
    print(y.shape)
