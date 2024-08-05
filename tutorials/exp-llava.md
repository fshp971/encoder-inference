# PEI-assisted Adversarial Attack vs LLaVA

Example scripts and configurations for LLaVA ([llava-1.5-13b](https://huggingface.co/llava-hf/llava-1.5-13b-hf)) experiments are collected in [scripts/exp-llava](../scripts/exp-llava) and [configs/exp-llava](../configs/exp-llava) respectively.

Here we present a brief tutorial on how to run experiments with these scripts and configurations (Please check **comments in these scripts** to see how to set encoder, downstream task, etc.).

**Step 0:** Enter the folder:

```
cd ./scripts/exp-llava
```

**Step 1:** Synthesize PEI attack images for a candidate encoder (e.g., `clip-vit-l14-336`) with the configuration [configs/exp-llava/atk.yaml](../configs/exp-llava/atk.yaml) and objective images in [scripts/obj-img](../scripts/obj-img):

```
bash atk.sh ../../
```

**Step 2:** Calculate the PEI score of a candidate encoder (e.g., `clip-vit-l14-336` with the configuration [configs/exp-llava/eval-l14-336.yaml](../configs/exp-llava/eval-l14-336.yaml)) on the targeted LLaVA service:

```
bash eval.sh ../../
```

Obtained PEI scores can then be used to calculate **PEI z-scores** to conduct the PEI attack (see the paper for details).

**Step 3**: Once the image encoder used by LLaVA (in our case, it is `clip-vit-l14-336` that used by `llava-1.5-13b`) is revealed by the PEI attack, since the used `clip-vit-l14-336` is an open-source model, we can therefore generate PEI-assisted adversarial examples in a white-box manner (where targeted harmful images are in [./configs/llava-imgs](../configs/llava-imgs)):

```
bash furatk.sh ../../
```

**PS:** Sample PEI-assisted adversarial examples can also be found in [./configs/llava-imgs](../configs/llava-imgs).

**Step 4:** Evaluate generated PEI-assisted adversarial examples on the targeted LLaVA service:

```
bash eval-furatk.sh ../../
