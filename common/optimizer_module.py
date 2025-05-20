"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
Enhanced for better architecture
"""
import torch
import gc
from common.split_params import group_weight
from common.lr_schedule import WarmUpPolyLR


class OptimizerModule:
    """
    Optimizer module for handling optimization strategy
    """
    def __init__(self, config):
        """
        Initialize the optimizer module

        Args:
            config: Configuration object containing optimizer parameters
        """
        self.config = config
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None

    def setup(self, model, modal_extractor):
        """
        Set up the optimizer and learning rate scheduler

        Args:
            model: The main model
            modal_extractor: The modalities extractor

        Returns:
            Tuple of (optimizer, lr_scheduler, scaler)
        """
        # Group parameters for optimization
        params = []
        cmnext_params = []
        modal_extract_params = []

        # Group parameters by layer type
        cmnext_params = group_weight(cmnext_params, model, torch.nn.BatchNorm2d, self.config.LEARNING_RATE)
        modal_extract_params = group_weight(modal_extract_params, modal_extractor, torch.nn.BatchNorm2d, self.config.LEARNING_RATE)

        # Combine parameters
        params.append(dict(
            params=cmnext_params[0]['params'] + modal_extract_params[0]['params'], 
            lr=self.config.LEARNING_RATE
        ))
        params.append(dict(
            params=cmnext_params[1]['params'] + modal_extract_params[1]['params'], 
            weight_decay=.0,
            lr=self.config.LEARNING_RATE
        ))

        # Initialize optimizer based on configuration
        optimizer_type = getattr(self.config, 'OPTIMIZER', 'sgd').lower()

        if optimizer_type == 'adam':
            # Adam optimizer with better convergence properties
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.config.LEARNING_RATE,
                betas=(0.9, 0.999),  # Default Adam betas
                eps=1e-8,  # Default epsilon for numerical stability
                weight_decay=self.config.WD,
                amsgrad=False  # AMSGrad variant not used by default
            )
        elif optimizer_type == 'adamw':
            # AdamW optimizer with improved weight decay handling
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.config.LEARNING_RATE,
                betas=(0.9, 0.999),  # Default AdamW betas
                eps=1e-8,  # Default epsilon for numerical stability
                weight_decay=self.config.WD,
                amsgrad=False  # AMSGrad variant not used by default
            )
        else:  # Default to SGD
            # SGD optimizer with momentum and weight decay
            self.optimizer = torch.optim.SGD(
                params,
                lr=self.config.LEARNING_RATE,
                momentum=self.config.SGD_MOMENTUM,
                weight_decay=self.config.WD,
                nesterov=True  # Enable Nesterov momentum for better convergence
            )

        # Calculate iterations for learning rate scheduler
        iters_per_epoch = self._estimate_iters_per_epoch()
        max_iters = self.config.EPOCHS * iters_per_epoch

        # Initialize learning rate scheduler
        self.lr_scheduler = WarmUpPolyLR(
            self.optimizer,
            start_lr=self.config.LEARNING_RATE,
            lr_power=self.config.POLY_POWER,
            total_iters=max_iters,
            warmup_steps=iters_per_epoch * self.config.WARMUP_EPOCHS
        )

        # Initialize gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

        # Clean up memory
        del params
        del cmnext_params
        del modal_extract_params
        gc.collect()
        torch.cuda.empty_cache()

        return self.optimizer, self.lr_scheduler, self.scaler

    def _estimate_iters_per_epoch(self):
        """
        Estimate the number of iterations per epoch based on dataset size and batch size

        Returns:
            Estimated number of iterations per epoch
        """
        # This is a rough estimate; in practice, we would calculate this from the actual dataset size
        # For now, we'll use a placeholder value that will be updated when the actual data loader is created
        return 1000  # Placeholder value

    def update_iters_per_epoch(self, iters_per_epoch):
        """
        Update the learning rate scheduler with the actual number of iterations per epoch

        Args:
            iters_per_epoch: Actual number of iterations per epoch
        """
        if self.lr_scheduler is not None:
            max_iters = self.config.EPOCHS * iters_per_epoch
            self.lr_scheduler.total_iters = max_iters
            self.lr_scheduler.warmup_steps = iters_per_epoch * self.config.WARMUP_EPOCHS
