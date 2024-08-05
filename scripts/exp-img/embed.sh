cd $1


# Optional pre-trained encoders are:
# "rn34-hf", "rn50-hf", "rn34-ms", "rn50-ms", "mobilenetv3-hf", "clip-vit-l14"
enc="clip-vit-l14"


# Optional downstream datasets are:
# "cifar10", "svhn", "stl10", "food101"
ds="stl10"


python exp_img.py \
    --encoder ${enc} \
    --dataset ${ds} \
    --exp-type embed \
    --embed-bs 2000 \
    --save-dir ./results/exp-img/embed/${ds}/${enc} \
    --save-name embed
