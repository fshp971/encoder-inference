cd $1


# All the 20 PEI objective texts.
sentences=(
    'black bridge'
    'dark battery'
    'orange cherry'
    'yellow rabbit'
    'white network'
    'hot fish'
    'dirty softdrink'
    'green album'
    'beautiful oven'
    'purple dragon'
    'red sunflower'
    'dirty battery'
    'black mobilephone'
    'heavy carcamera'
    'green aircraft'
    'muddy album'
    'purple network'
    'hot house'
    'yellow bridge'
    'ice opera'
)


# Optional pre-trained encoders are:
# "clip-vit-b16", "clip-vit-b32", "clip-vit-l14", "openclip-vit-b32", "openclip-vit-l14", "openclip-vit-h14"
enc="clip-vit-l14"


for i in {1..20}
do

sen=${sentences[$i-1]}
# echo $sen

python exp_text2img.py \
    --encoder ${enc} \
    --exp-type atk \
    --exp-config ./configs/exp-text2img/atk.yaml \
    --atk-target-text "${sen}" \
    --save-dir ./results/exp-text2img/atk/${enc}/$i \
    --save-name atk

done
