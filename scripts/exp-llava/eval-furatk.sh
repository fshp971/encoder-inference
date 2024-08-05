cd $1


python exp_llava.py \
    --exp-type eval-furatk \
    --save-dir ./results/exp-llava/eval-furatk \
    --eval-imgs ./results/exp-llava/furatk/gen-adv-1-fin.npy \
    --save-name eval-furatk

    # example target-image in the paper:
    #   --eval-imgs ./configs/llava-imgs/tar-example-1.png \

    # example adv-image in the paper:
    #   --eval-imgs ./configs/llava-imgs/adv-example-1.png \
