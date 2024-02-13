#!/bin/bash

# parameters for cifar10 resnet8
# model=resnet8
# dataset=cifar10
# epochs=100
# batch_size=512
# lr=0.05
# momentum=0.9
# weight_decay=0.256
# save_dir=$model-$dataset

#parameters for cifar10 vgg16
# model=vgg16
# dataset=cifar10
# epochs=300
# batch_size=128
# lr=0.05
# momentum=0.9
# weight_decay=5e-4
# save_dir=$model-$dataset

#parameters for mnist resnet8
model=resnet8
dataset=cifar10
epochs=300
batch_size=512
lr=0.05
momentum=0.9
weight_decay=0.256
save_dir=$model-$dataset

echo "python benchmark.py  --arch=$model  --dataset=$dataset --epochs=$epochs --batch_size=$batch_size --lr=$lr --momentum=$momentum --weight_decay=$weight_decay --save_dir=save_$save_dir"
python benchmark.py  --arch=$model  --dataset=$dataset --epochs=$epochs --batch_size=$batch_size --lr=$lr --momentum=$momentum --weight_decay=$weight_decay --save_dir=save_$save_dir