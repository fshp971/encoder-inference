# PEI Attack vs Text-to-image Services

Example scripts and configurations for text-to-image experiments are collected in [scripts/exp-text2img](../scripts/exp-text2img) and [configs/exp-text2img](../configs/exp-text2img) respectively.

Here we present a brief tutorial on how to run experiments with these scripts and configurations (Please check **comments in these scripts** to see how to set encoder, downstream task, etc.).

**Step 0:** Enter the folder:

```
cd ./scripts/exp-text2img
```

**Step 1:** Synthesize PEI attack texts for a candidate encoder (e.g., `clip-vit-l14`) with the configuration [configs/exp-text2img/atk.yaml](../configs/exp-text2img/atk.yaml) (PEI objective texts are presented in the script):

```
bash atk.sh ../../
```

The used PEI objective texts are generated via [scripts/gen_obj_text.py](../scripts/gen_obj_text.py).

**Step 2:** Calculate the PEI score of a candidate encoder (e.g., `clip-vit-b16` with the configuration [configs/exp-text2img/eval-b16.yaml](../configs/exp-text2img/eval-b16.yaml)) on a targeted downstream text-to-image service (e.g., `sd-v1-2`):

```
bash eval.sh ../../
```

Obtained PEI scores can then be used to calculate **PEI z-scores** to conduct the PEI attack (see the paper for details).
