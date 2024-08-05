cd $1


# Optional pre-trained encoders are:
# "bertbase", "bertlarge", "t5small", "t5base", "robertabase", "clip-vit-l14"
enc="bertbase"


# Optional downstream datasets are:
# "sst5", "trec", "yelp", "agnews"
ds="sst5"


python exp_text.py \
    --encoder ${enc} \
    --dataset ${ds} \
    --exp-type embed \
    --embed-bs 2000 \
    --save-dir ./results/exp-text/embed/${ds}/${enc} \
    --save-name embed
