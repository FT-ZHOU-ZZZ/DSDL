:<<BLOCK
lr: learning rate
lrp: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is lr * lrp
batch-size: number of images per batch
image-size: size of the image
epochs: number of training epochs
evaluate: evaluate model on validation set
resume: path to checkpoint
BLOCK

CUDA_VISIBLE_DEVICES=0 python3 demo_coco.py data/coco --image-size 448 --batch-size 32 -e --resume checkpoint/coco/coco_checkpoint.pth.tar --lambd 0.1 --beta 0.000001