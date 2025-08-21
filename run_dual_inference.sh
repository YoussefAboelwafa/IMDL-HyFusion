#!/bin/bash

# Example script for running dual model inference
# Update paths according to your setup

# Configuration
GPU=0
INPUT_DIR="/scratch/dr/a.samir/IMDL/IMDL-HyFusion/data/Columbia/4cam_auth"
OUTPUT_DIR_SEG="inference_results_seg_columbia"
OUTPUT_DIR_MM="inference_results_mm_columbia"

# Model configurations and checkpoints
EXP_MM="/scratch/dr/a.samir/IMDL/IMDL-HyFusion/experiments/ec_example_phase2.yaml"
EXP_SEG="/scratch/dr/a.samir/IMDL/IMDL-HyFusion/experiments/ec_example_phase2.yaml" 
CKPT_MM="/scratch/dr/a.samir/IMDL/IMDL-HyFusion/ckpt/early_fusion_localization.pth"
CKPT_SEG="/home/a.samir/IMDL-HyFusion/ckpt/Segmentation-FPN-lr0.00001-secondary/best_val_loss_epoch14.pth"

# Optional: ground truth file for metrics computation
GT_FILE="path/to/ground_truth.txt"

echo "Running dual model inference..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR_SEG"
echo "Output directory: $OUTPUT_DIR_MM"

# Basic inference (no metrics)
python lol.py \
    -gpu $GPU \
    -exp $EXP_MM \
    -ckpt $CKPT_MM \
    -path $INPUT_DIR \
    -out $OUTPUT_DIR_MM \

python lol2.py \
   -gpu $GPU \
   -exp $EXP_SEG \
   -ckpt $CKPT_SEG \
   -path $INPUT_DIR \
   -out $OUTPUT_DIR_SEG

# Uncomment below for inference with metrics computation
# python inference_dual_models.py \
#     -gpu $GPU \
#     -exp_mm $EXP_MM \
#     -exp_seg $EXP_SEG \
#     -ckpt_mm $CKPT_MM \
#     -ckpt_seg $CKPT_SEG \
#     -input_dir $INPUT_DIR \
#     -output_dir $OUTPUT_DIR \
#     -gt_file $GT_FILE \
#     --save_maps \
#     --compute_metrics

echo "Inference completed. Check results in: $OUTPUT_DIR"
