# import argparse
# import numpy as np
# import torch
# import torchvision.transforms.functional as TF
# import logging
# import matplotlib.pyplot as plt
# from models.cmnext_conf_mm import CMNeXtWithConf
# from models.modal_extract import ModalitiesExtractor
# from configs.cmnext_init_cfg import _C as config, update_config
# import cv2
# from albumentations.pytorch import ToTensorV2
# import albumentations as A
# from pathlib import Path
# from tqdm import tqdm

# def preprocess_image(image_path, device):
#     """
#     Preprocess the input image for inference.
#     Args:
#         image_path (str): Path to the input image.
#         device (str): Device to which the tensor should be moved.
#     Returns:
#         torch.Tensor: Preprocessed image tensor.
#     """
#     # --- Single image inference ---
#     img_data = cv2.imread(image_path)
#     if img_data is None:
#         raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")
#     image = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
#     h, w, c = image.shape

#     # resize to 512x512
#     transform = A.Compose([ 
#         A.Resize(height=512, width=512, p=1.0),
#     ], additional_targets={'image': 'image'})
#     image = transform(image=image)['image']

#     # Use the same transforms as in ManipulationDataset (no augmentation, just ToTensorV2)
#     image_tensor = ToTensorV2()(image=image)['image']
#     image_tensor = image_tensor / 256.0
#     image_tensor = image_tensor.unsqueeze(0).to(device)
#     return image_tensor, image_path

# def load_images(images_dir, device):
#     """
#     Load and preprocess all images in the specified directory.
#     Args:
#         images_dir (str): Directory containing images.
#         device (str): Device to which the tensors should be moved.
#     Returns:
#         list: List of preprocessed image tensors.
#     """
#     image_tensors = []
#     # all images in the directory with all extensions
#     for img_path in sorted(Path(images_dir).glob("*")):
#         image_tensor, image_path = preprocess_image(str(img_path), device)
#         image_tensors.append((image_tensor, image_path))
#     return image_tensors


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Infer')
#     parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
#     parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')
#     parser.add_argument('-exp', '--exp', type=str, default='experiments/ec_example_phase2.yaml', help='Yaml experiment file')
#     parser.add_argument('-ckpt', '--ckpt', type=str, default='ckpt/early_fusion_localization.pth', help='Checkpoint')
#     parser.add_argument('-path', '--path', type=str, default='images', help='Images path')
#     parser.add_argument('-out', '--out', type=str, default='output', help='Output directory for masks')
#     parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

#     args = parser.parse_args()

#     config = update_config(config, args.exp)

#     loglvl = getattr(logging, args.log.upper())
#     logging.basicConfig(level=loglvl)

#     gpu = args.gpu
#     device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
#     print(f"Using device: {device}")
#     print(f"Using path: {args.path}")
#     np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

#     if device != 'cpu':
#         import torch.backends.cudnn as cudnn
#         cudnn.benchmark = False
#         cudnn.deterministic = True
#         cudnn.enabled = config.CUDNN.ENABLED

#     modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], config.MODEL.NP_WEIGHTS)
#     model = CMNeXtWithConf(config.MODEL)
#     print(args.ckpt)
#     ckpt = torch.load(args.ckpt, weights_only=False, map_location=torch.device('cpu'))
#     model.load_state_dict(ckpt['state_dict'])
#     modal_extractor.load_state_dict(ckpt['extractor_state_dict'])
#     # model.load_training_checkpoint(args.ckpt, modal_extractor=modal_extractor, map_location=device)
#     modal_extractor.to(device)
#     model = model.to(device)
#     modal_extractor.eval()
#     model.eval()

#     # Load images from the specified directory
#     image_tensors = load_images(args.path, device)

#     pbar = tqdm(image_tensors, desc="Processing images")

#     for image_tensor, img_path in pbar:
#         with torch.no_grad():
#             modals = modal_extractor(image_tensor)
#         images_norm = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         inp = [images_norm] + modals
#         anomaly, confidence, detection = model(inp)
#         map = torch.nn.functional.softmax(anomaly, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
#         # Binarize the anomaly map with a threshold (e.g., 0.5)
#         threshold = 0.8
#         binary_map = (map > threshold).astype(np.uint8)
#         # Ensure output directory exists
#         output_dir = Path(args.out)
#         output_dir.mkdir(exist_ok=True)
#         print("Output directory:", output_dir)
#         img_path = Path(img_path).name
#         # Save the anomaly map and binary map
#         target = output_dir / (img_path.split(".")[-2] + "_mask.png")
#         target_bin = output_dir / (img_path.split(".")[-2] + "_mask_bin.png")
#         plt.imsave(target, map,cmap="viridis", vmin=0, vmax=1)
#         plt.imsave(target_bin, binary_map, cmap='gray', vmin=0, vmax=1)
#         print(f"Saved mask to {target}")
#         print(f"Saved binary mask to {target_bin}")
#         print(f"Detection score: {detection.item()}")

import argparse
import numpy as np
import torch
import torchvision.transforms.functional as TF
import logging
import matplotlib.pyplot as plt
from models.cmnext_conf_mm import CMNeXtWithConf
from models.modal_extract import ModalitiesExtractor
from configs.cmnext_init_cfg import _C as config, update_config
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
from pathlib import Path
from tqdm import tqdm

def preprocess_image(image_path, device):
    img_data = cv2.imread(image_path)
    if img_data is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    transform = A.Compose([A.Resize(height=512, width=512)])
    image = transform(image=image)['image']
    image_tensor = ToTensorV2()(image=image)['image'] / 256.0
    return image_tensor.unsqueeze(0).to(device)

def load_image_paths(images_dir):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    return sorted([p for p in Path(images_dir).glob("*") if p.is_file() and p.suffix.lower() in exts])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer')
    parser.add_argument('-gpu', '--gpu', type=int, default=0)
    parser.add_argument('-log', '--log', type=str, default='INFO')
    parser.add_argument('-exp', '--exp', type=str, default='experiments/ec_example_phase2.yaml')
    parser.add_argument('-ckpt', '--ckpt', type=str, default='ckpt/early_fusion_localization.pth')
    parser.add_argument('-path', '--path', type=str, default='images')
    parser.add_argument('-out', '--out', type=str, default='output')
    parser.add_argument('-bs', '--batch_size', type=int, default=3)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = update_config(config, args.exp)
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'

    if device != 'cpu':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = False
        cudnn.deterministic = True
        cudnn.enabled = config.CUDNN.ENABLED

    modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], config.MODEL.NP_WEIGHTS)
    model = CMNeXtWithConf(config.MODEL)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    modal_extractor.load_state_dict(ckpt['extractor_state_dict'])
    model.to(device).eval()
    modal_extractor.to(device).eval()

    image_paths = load_image_paths(args.path)
    output_dir = Path(args.out)
    output_dir.mkdir(exist_ok=True)

    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + args.batch_size]
        batch_tensors = [preprocess_image(str(p), device) for p in batch_paths]
        image_batch = torch.cat(batch_tensors, dim=0)

        with torch.no_grad():
            modals = modal_extractor(image_batch)
            images_norm = TF.normalize(image_batch, mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
            anomaly, confidence, detection = model([images_norm] + modals)
            maps = torch.nn.functional.softmax(anomaly, dim=1)[:, 1, :, :].detach().cpu().numpy()

        for j, path in enumerate(batch_paths):
            map = maps[j]
            binary_map = (map > 0.5).astype(np.uint8)
            name = Path(path).stem
            original_img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

            plt.imsave(output_dir / f"{name}_original.png", original_img)
            plt.imsave(output_dir / f"{name}_mask.png", map, cmap="viridis", vmin=0, vmax=1)
            plt.imsave(output_dir / f"{name}_mask_bin.png", binary_map, cmap="gray", vmin=0, vmax=1)

            print(f"Saved: {name}_original.png, {name}_mask.png, {name}_mask_bin.png")
            print(f"Detection score: {detection[j].item()}")
