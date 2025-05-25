cd $1

# Below three lines set the targeted downstream service (i.e., rn50-hf + cifar10)
# When you switch dataset, you should also set the classification classes number.
# Optional datasets and corresponding classes number are:
# ("cifar10", "10"), ("svhn", "10"), ("stl10", "10"), ("food101", "101")
ds="cifar10"; cls="10"
enc="rn50-hf"

# Below line set the correct PEI-revealed encoder
inf_enc="rn50-hf"

# Optional steal-type are:
# "logit-label", "hard-label"
steal_type="logit-label"

python exp_img_steal.py \
    --encoder ${enc} \
    --dataset ${ds} \
    --cls-num ${cls} \
    --pei-assist \
    --surrogate-ds-size 50000 \
    --steal-type ${steal_type} \
    --exp-config ./configs/exp-img/furatk/downstream.yaml \
    --inferred-encoder ${inf_enc} \
    --steal-npz-data-path ./results/imagenet-val.npz \
    --trainset-path ./results/exp-img/embed/${ds}/${enc}/embed_trainset.npz \
    --testset-path ./results/exp-img/embed/${ds}/${enc}/embed_testset.npz \
    --inferred-trainset-path ./results/exp-img/embed/${ds}/${inf_enc}/embed_trainset.npz \
    --inferred-testset-path ./results/exp-img/embed/${ds}/${inf_enc}/embed_testset.npz \
    --classifier-path ./results/exp-img/train/${ds}/${enc}/train_classifier.pt \
    --save-dir ./results/exp-img/steal/correct \
    --save-name steal
