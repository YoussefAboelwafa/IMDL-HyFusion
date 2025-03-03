# import argparse
# import numpy as np
# from torch.utils.data import DataLoader
# import torch
# import torchvision.transforms.functional as TF
# import logging
# import matplotlib.pyplot as plt
# from data.datasets import ManipulationDataset
# from models.cmnext_conf import CMNeXtWithConf
# from models.modal_extract import ModalitiesExtractor
# from configs.cmnext_init_cfg import _C as config, update_config
# from PIL import Image

import torch

print(torch.__version__) # pytorch版本
print(torch.version.cuda) # cuda版本
print(torch.cuda.is_available()) # 查看cuda是否可用

# parser = argparse.ArgumentParser(description='Infer')
# parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
# parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')
# parser.add_argument('-exp', '--exp', type=str, default='experiments/ec_example_phase2.yaml', help='Yaml experiment file')
# parser.add_argument('-ckpt', '--ckpt', type=str, default='ckpt/early_fusion_detection.pth', help='Checkpoint')
# parser.add_argument('-path', '--path', type=str, default='example.png', help='Image path')
# parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

# args = parser.parse_args()

# config = update_config(config, args.exp)

# loglvl = getattr(logging, args.log.upper())
# logging.basicConfig(level=loglvl)

# gpu = args.gpu

# device ='cpu'
# np.set_printoptions(formatter={'float': '{: 7.3f}'.format})
# print(f"Device: {device}")
# if device != 'cpu':
#     # cudnn setting
#     import torch.backends.cudnn as cudnn

#     cudnn.benchmark = False
#     cudnn.deterministic = True
#     cudnn.enabled = config.CUDNN.ENABLED


# modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], config.MODEL.NP_WEIGHTS)

# model = CMNeXtWithConf(config.MODEL)
# ckpt = torch.load(args.ckpt,map_location=torch.device('cpu'))

# model.load_state_dict(ckpt['state_dict'])
# modal_extractor.load_state_dict(ckpt['extractor_state_dict'])

# modal_extractor.to(device)
# model = model.to(device)
# modal_extractor.eval()
# model.eval()


# # get the image to infer
# image_path = "/mnt/c/Users/a.samir/Desktop/WS/IMDL-HyFusion/download.jpg"
# target = "mask.png"
# image = Image.open(image_path).convert('RGB')
# inp = TF.to_tensor(image).unsqueeze(0).to(device)

# with torch.no_grad():
#     images = inp.to(device, non_blocking=True)
#     # masks = masks.squeeze(1).to(device, non_blocking=True)

#     modals = modal_extractor(inp)

#     inp_norm = TF.normalize(inp, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     inp = [inp_norm] + modals

#     anomaly, confidence, detection = model(inp)

#     # gt = masks.squeeze().cpu().numpy()
#     map = torch.nn.functional.softmax(anomaly, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
#     det = detection.item()
#     print(f"Detection: {det}")
#     # print(f"Anomaly map: {map}")
#     # print(f"Anomaly map shape: {map.shape}")
#     plt.imsave(target, map, cmap='RdBu_r', vmin=0, vmax=1)
#     plt.imsave("conf.png", confidence.squeeze().cpu().numpy(), cmap='RdBu_r', vmin=0, vmax=1)