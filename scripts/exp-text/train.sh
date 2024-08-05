cd $1


# Optional pre-trained encoders are:
# "bertbase", "bertlarge", "t5small", "t5base", "robertabase", "clip-vit-l14"
enc="bertbase"


# when you switch dataset, you should also set the classification classes number.
# Optional datasets and corresponding classes number are:
# ("sst5", "5"), ("trec", "6"), ("yelp", "5"), ("agnews", "4")
ds="sst5"; cls="5"


python exp_text.py \
    --encoder ${enc} \
    --dataset ${ds} \
    --cls-num ${cls} \
    --exp-type train \
    --exp-config ./configs/exp-text/train.yaml \
    --trainset-path ./results/exp-text/embed/${ds}/${enc}/embed_trainset.npz \
    --testset-path ./results/exp-text/embed/${ds}/${enc}/embed_testset.npz \
    --save-dir ./results/exp-text/train/${ds}/${enc} \
    --save-name train
