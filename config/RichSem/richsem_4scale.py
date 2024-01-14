_base_ = ['baseline_4scale.py']

epochs = 24
lr_drop = 20

use_language=True
use_visual_distill=True
distill_type='clip_logits'
clip_distill_objective='gt'
distill_loss_coef=0.5
use_imagenet=True
imagenet_use_mosaic=True
clip_model='RN50'
use_dynamic_distill_weight=False
resnet_pretrain_path=''
