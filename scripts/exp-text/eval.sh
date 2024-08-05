cd $1


# Optional pre-trained encoders are:
# "bertbase", "bertlarge", "t5small", "t5base", "robertabase", "clip-vit-l14"
enc="bertbase"


# Optional downstream datasets are:
# "sst5", "trec", "yelp", "agnews"
ds="sst5"


# If you plan to use "Flipping-defense", add the folloinwg argument:
#   --def-flip-rate 0.2


python exp_text.py \
    --encoder ${enc} \
    --exp-type eval \
    --classifier-path ./results/exp-text/train/${ds}/${enc}/train_classifier.pt \
    --exp-config ./configs/exp-text/eval.yaml \
    --save-dir ./results/exp-text/eval \
    --save-name eval
