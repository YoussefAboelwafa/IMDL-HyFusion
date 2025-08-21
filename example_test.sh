source data.sh
exp="./experiments/ec_example.yaml"
ckpt="/home/a.samir/IMDL-HyFusion/ckpt/Segmentation-FPN-lr0.00001/best_val_loss.pth"
res="./results/Segmentation-FPN-lr0.00001"
mkdir -p $res
$pint test_localization.py --exp $exp --ckpt $ckpt --manip $columbia_manip > $res/Columbia.txt
# clear
$pint test_localization.py --exp $exp --ckpt $ckpt --manip $cover_manip > $res/COVER.txt
# # clear
# # $pint test_localization.py --exp $exp --ckpt $ckpt --manip $dso1_manip > $res/DSO-1.txt
# # clear
$pint test_localization.py --exp $exp --ckpt $ckpt --manip $cocoglide_manip > $res/CocoGlide.txt
# # clear
$pint test_localization.py --exp $exp --ckpt $ckpt --manip $casiav1_manip > $res/Casiav1.txt