cd $1


# Optional pre-trained encoders are:
# "rn34-hf", "rn50-hf", "rn34-ms", "rn50-ms", "mobilenetv3-hf", "clip-vit-l14"
enc="rn34-hf"


# when you switch dataset, you should also set the classification classes number.
# Optional datasets and corresponding classes number are:
# ("cifar10", "10"), ("svhn", "10"), ("stl10", "10"), ("food101", "101")
ds="cifar10"; cls="10"


python exp_img.py \
    --encoder ${enc} \
    --dataset ${ds} \
    --cls-num ${cls} \
    --exp-type train \
    --exp-config ./configs/exp-img/train.yaml \
    --trainset-path ./results/exp-img/embed/${ds}/${enc}/embed_trainset.npz \
    --testset-path ./results/exp-img/embed/${ds}/${enc}/embed_testset.npz \
    --save-dir ./results/exp-img/train/${ds}/${enc} \
    --save-name train
