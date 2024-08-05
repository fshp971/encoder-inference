cd $1


# Optional downstream pre-trained encoders are:
# "rn34-hf", "rn50-hf", "rn34-ms", "rn50-ms", "mobilenetv3-hf", "clip-vit-l14"
enc="rn34-hf"


# Optional downstream datasets are:
# "cifar10", "svhn", "stl10", "food101"
ds="cifar10"


# If you plan to use "JPEG-defense", add the following argument:
#   --jpeg-def


# If you plan to use "resizing-mitigation" against "JPEG-defense",
# add the following argument:
#   --resize-atk 128


python exp_img.py \
    --encoder ${enc} \
    --exp-type eval \
    --classifier-path ./results/exp-img/train/${ds}/${enc}/train_classifier.pt \
    --exp-config ./configs/exp-img/eval.yaml \
    --save-dir ./results/exp-img/eval \
    --save-name eval
