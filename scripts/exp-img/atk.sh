cd $1


# Optional pre-trained encoders are:
# "rn34-hf", "rn50-hf", "rn34-ms", "rn50-ms", "mobilenetv3-hf", "clip-vit-l14"
enc="rn34-hf"


for i in {1..10}
do

python exp_img.py \
    --encoder ${enc} \
    --exp-type atk \
    --exp-config ./configs/exp-img/atk.yaml \
    --atk-target-img ./configs/obj-imgs/$i.bmp \
    --save-dir ./results/exp-img/atk/${enc}/$i \
    --save-name atk

done
