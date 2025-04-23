"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.base import BaseModel
from models.heads import SegFormerHead
import logging
from models.modules.esb import ESB
from models.modules.segmentation import get_semantic_map
from models.modules.dual_atten import DAHead
# 1. Implement gradient checkpointing
from torch.utils.checkpoint import checkpoint

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
        
        # Edge detection branch
        self.edge_branch = ESB(2, sobel=True)
        self.edge_channel_reduce = nn.Conv2d(2048, 1, kernel_size=1)
        
        # Channel adjustment layers with BatchNorm and ReLU
        self.adjust_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c + 2, c, kernel_size=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            ) for c in [64, 128, 320, 512]
        ])
        
        # Detection head
        if cfg.DETECTION == 'confpool':
            self.detection = nn.Sequential(
                nn.Linear(8, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
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
      # get semantic map and convert to float
      sem_map = get_semantic_map(image=x[0])  # [1, 1, 512, 512]
      sem_map = sem_map.unsqueeze(1).float()  # Convert to float and ensure [B, C, H, W] format
          
      # get edge map
      edges, edge_map = self.edge_branch(x[0])  # edge_map: [1, 1, 2048, 32, 32]
      edge_map = edge_map.squeeze(2)  # Remove extra dimension to get [B, C, H, W]
      # Reduce edge map channels to 1
      edge_map = self.edge_channel_reduce(edge_map)  # Now shape: [B, 1, H, W]

      if masks is not None:
          y = self.backbone(x, masks)
      else:
          y = self.backbone(x)  # List of 4 tensors with different scales
      
      # Resize semantic map and edge map to match each feature map scale
      enhanced_features = []
      for idx, feat in enumerate(y):
          # Get current feature map size
          curr_size = feat.shape[2:]  # This will be a tuple (H, W)
          
          # Resize semantic map to current scale
          sem_map_resized = F.interpolate(sem_map, 
                                        size=curr_size,  # Pass tuple of (H, W)
                                        mode='bilinear',
                                        align_corners=False)
          
          # Resize edge map to current scale
          edge_map_resized = F.interpolate(edge_map,
                                        size=curr_size,  # Pass tuple of (H, W)
                                        mode='bilinear',
                                        align_corners=False)
          
          # Concatenate along channel dimension
          enhanced_feat = torch.cat([feat, sem_map_resized, edge_map_resized], dim=1)
          
          # Apply 1x1 conv to match original channel dimensions
          if idx == 0:
              enhanced_feat = self.adjust_layers[0](enhanced_feat)  # Output channels: 64
          elif idx == 1:
              enhanced_feat = self.adjust_layers[1](enhanced_feat)  # Output channels: 128
          elif idx == 2:
              enhanced_feat = self.adjust_layers[2](enhanced_feat)  # Output channels: 320
          else:
              enhanced_feat = self.adjust_layers[3](enhanced_feat)  # Output channels: 512
              
          enhanced_features.append(enhanced_feat)

      # Pass through decode head
    #   enhanced_features = self.da_head(enhanced_features)
      out = self.decode_head(enhanced_features)
      out = F.interpolate(out, size=x[0].shape[2:], mode='bilinear', align_corners=False)

      if self.train_phase == 'detection':
          conf = self.conf_head(enhanced_features)
          conf = F.interpolate(conf, size=x[0].shape[2:], mode='bilinear', align_corners=False)
          from .layer_utils import weighted_statistics_pooling
          f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
          f2 = weighted_statistics_pooling(out[:, 1:2, :, :] - out[:, 0:1, :, :], F.logsigmoid(conf)).view(
              out.shape[0], -1)
          det = self.detection(torch.cat((f1, f2), -1))
          return out, conf, det

      return out, edges, sem_map


    def init_pretrained(self, pretrained: str = None, backbone: str = None) -> None:
        if pretrained:
            logging.info('Loading pretrained module: {}'.format(pretrained))
            if self.backbone.num_modals > 0:
                load_dualpath_model(self.backbone, pretrained, backbone)
            else:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                if 'model' in checkpoint.keys():
                    checkpoint = checkpoint['model']
                msg = self.backbone.load_state_dict(checkpoint, strict=False)
                print(msg)


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
