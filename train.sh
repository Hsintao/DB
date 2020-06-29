CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --num_gpus 2 --resume pretrained/totaltext_resnet50

#CUDA_VISIBLE_DEVICES=4 python  train.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --num_gpus 1

CUDA_VISIBLE_DEVICES=4 python  train.py experiments/seg_detector/1.yaml --num_gpus 1 --resume pretrained/totaltext_resnet50
CUDA_VISIBLE_DEVICES=4 python eval.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --polygon --box_thresh 0.6 \
--resume  workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model_lr1e-3_epoch200/model_epoch_