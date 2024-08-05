cd $1


python exp_llava.py \
    --exp-type furatk \
    --exp-config ./configs/exp-llava/furatk.yaml \
    --save-dir ./results/exp-llava/furatk \
    --atk-target-img ./configs/llava-imgs/tar-example-1.png \
    --further-atk-img-num 16 \
    --further-atk-img ./results/exp-llava/furatk/gen-adv-1 \
    --save-name furatk
