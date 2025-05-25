cd $1

# Below three lines set the targeted downstream service (i.e., rn50-hf + cifar10)
# When you switch dataset, you should also set the classification classes number.
# Optional datasets and corresponding classes number are:
# ("cifar10", "10"), ("svhn", "10"), ("stl10", "10"), ("food101", "101")
ds="cifar10"; cls="10"
enc="rn50-hf"

# Optional steal-type are:
# "logit-label", "hard-label"
steal_type="logit-label"

python exp_img_steal.py \
    --encoder ${enc} \
    --dataset ${ds} \
    --cls-num ${cls} \
    --surrogate-ds-size 50000 \
    --steal-type logit-label \
    --exp-config ./configs/exp-img/furatk/scratch.yaml \
    --steal-npz-data-path ./results/imagenet-val.npz \
    --trainset-path ./results/exp-img/embed/${ds}/${enc}/embed_trainset.npz \
    --testset-path ./results/exp-img/embed/${ds}/${enc}/embed_testset.npz \
    --classifier-path ./results/exp-img/train/${ds}/${enc}/train_classifier.pt \
    --save-dir ./results/exp-img/steal/scratch \
    --save-name steal
