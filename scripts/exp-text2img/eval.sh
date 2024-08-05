cd $1


# Optional downstream text2img services are:
# "sd-v1-2", "sd-v1-4", "sd-v2-1-base", "sd-v2-1"
downstream="sd-v1-2"


python exp_text2img.py \
    --text2img-model ${downstream} \
    --exp-type eval \
    --exp-config ./configs/exp-text2img/eval-b16.yaml \
    --save-dir ./results/exp-text2img/eval/${downstream} \
    --save-name eval-b16
