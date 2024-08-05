# PEI Attack vs Image Classification Services

Example scripts and configurations for image classification experiments are collected in [scripts/exp-img](../scripts/exp-img) and [configs/exp-img](../configs/exp-img) respectively.

Here we present a brief tutorial on how to run experiments with these scripts and configurations (Please check **comments in these scripts** to see how to set encoder, downstream task, etc.).

**Step 0:** Enter the folder:

```
cd ./scripts/exp-img
```

**Step 1:** Build the embedding dataset, e.g., `rn34-hf + cifar10`:

```
bash embed.sh ../../
```

**Step 2:** Train the downstream classifier on the built embedding dataset e.g., `rn34-hf + cifar10`, with the configuration [configs/exp-img/train.yaml](../configs/exp-img/train.yaml):

```
bash train.sh ../../
```

**Step 3:** Synthesize PEI attack images for a candidate encoder (e.g., `rn34-hf`) with the configuration [configs/exp-img/atk.yaml](../configs/exp-img/atk.yaml) and objective images in [scripts/obj-img](../scripts/obj-img):

```
bash atk.sh ../../
```

**Step 4:** Calculate the PEI score of a candidate encoder (e.g., `rn50-hf`) on a targeted downstream service (e.g., `rn34-hf + cifar10`) with the configuration [configs/exp-img/eval.yaml](../configs/exp-img/eval.yaml):

```
bash eval.sh ../../
```

Obtained PEI scores can then be used to calculate **PEI z-scores** to conduct the PEI attack (see the paper for details).
