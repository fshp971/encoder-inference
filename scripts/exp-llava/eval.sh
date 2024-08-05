cd $1


python exp_llava.py \
    --exp-type eval \
    --exp-config ./configs/exp-llava/eval-l14-336.yaml \
    --save-dir ./results/exp-llava/eval \
    --save-name eval
