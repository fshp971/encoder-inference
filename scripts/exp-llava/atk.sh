cd $1


# Optional pre-trained encoders are:
# "clip-vit-b16", "clip-vit-b32", "clip-vit-l14", "clip-vit-l14-336",
# "openclip-vit-b32", "openclip-vit-l14", "openclip-vit-h14"
enc="clip-vit-l14-336"


for i in {1..10}
do

python exp_llava.py \
    --encoder ${enc} \
    --exp-type atk \
    --exp-config ./configs/exp-llava/atk.yaml \
    --atk-target-img ./configs/obj-imgs/$i.bmp \
    --save-dir ./results/exp-llava/atk/${enc}/$i \
    --save-name atk

done
