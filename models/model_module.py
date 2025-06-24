"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
Enhanced for better architecture
"""
import torch
import logging
from models.cmnext_conf import CMNeXtWithConf
from models.modal_extract import ModalitiesExtractor


class ModelModule:
    """
    Model module for handling model initialization and configuration
    """
    def __init__(self, config, device, train_bayar=False):
        """
        Initialize the model module
        
        Args:
            config: Configuration object containing model parameters
            device: Device to use (cuda/cpu)
            train_bayar: Whether to train the bayar filter
        """
        self.config = config
        self.device = device
        self.train_bayar = train_bayar
        self.model = None
        self.modal_extractor = None
        
    def setup(self):
        """
        Set up the models
        
        Returns:
            Tuple of (model, modal_extractor)
        """
        # Initialize modalities extractor
        self.modal_extractor = ModalitiesExtractor(
            self.config.MODEL.MODALS[1:], 
            self.config.MODEL.NP_WEIGHTS
        )
        
        # Load pretrained weights for bayar if available
        if 'bayar' in self.config.MODEL.MODALS:
            self.modal_extractor.load_state_dict(
                torch.load('pretrained/modal_extractor/bayar_mhsa.pth', map_location=torch.device('cpu')), 
                strict=False
            )
            if not self.train_bayar:
                self.modal_extractor.bayar.eval()
                for param in self.modal_extractor.bayar.parameters():
                    param.requires_grad = False
        
        # Initialize main model
        self.model = CMNeXtWithConf(self.config.MODEL)
        
        # Move models to device
        self.modal_extractor.to(self.device)
        self.model.to(self.device)
        
        return self.model, self.modal_extractor
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Starting epoch number
        """
        if not checkpoint_path:
            return 0
            
        logging.info(f'Loading checkpoint from {checkpoint_path}')
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        if self.model is not None:
            self.model.load_state_dict(ckpt['state_dict'])
            
        if self.modal_extractor is not None:
            self.modal_extractor.load_state_dict(ckpt['extractor_state_dict'])
            
        return ckpt.get('epoch', -1) + 1