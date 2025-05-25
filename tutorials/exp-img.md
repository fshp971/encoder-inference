# Experiments on Image Classification Services

Example scripts and configurations for image classification experiments are collected in [scripts/exp-img](../scripts/exp-img) and [configs/exp-img](../configs/exp-img) respectively.

Here we present a brief tutorial on how to run experiments with these scripts and configurations (Please check **comments in these scripts** to see how to set encoder, downstream task, etc.).

**Step 0:** Enter the folder:

```bash
cd ./scripts/exp-img
```

## PEI Attack vs. Vision Encoder in Image Classification Services

**Step 1:** Build the embedding dataset, e.g., `rn34-hf + cifar10`:

```bash
bash embed.sh ../../
```

**Step 2:** Train the downstream classifier on the built embedding dataset e.g., `rn34-hf + cifar10`, with the configuration [configs/exp-img/train.yaml](../configs/exp-img/train.yaml):

```bash
bash train.sh ../../
```

**Step 3:** Synthesize PEI attack images for a candidate encoder (e.g., `rn34-hf`) with the configuration [configs/exp-img/atk.yaml](../configs/exp-img/atk.yaml) and objective images in [scripts/obj-img](../scripts/obj-img):

```bash
bash atk.sh ../../
```

**Step 4:** Calculate the PEI score of a candidate encoder (e.g., `rn50-hf`) on a targeted downstream service (e.g., `rn34-hf + cifar10`) with the configuration [configs/exp-img/eval.yaml](../configs/exp-img/eval.yaml):

```bash
bash eval.sh ../../
```

Obtained PEI scores can then be used to calculate **PEI z-scores** to conduct the PEI attack (see the paper for details).

## PEI-assisted Model Stealing vs. Image Classification Services

**Step 5:** Rebuild the ImageNet validation set into a single `.npz` file (to improve I/O performance):

```bash
python furatk/build_imagenet_val.py --data-folder-path {PATH_TO_IMAGENET_VAL_SET} --save-dir ../../results
```

where `{PATH_TO_IMAGENET_VAL_SET}` is the path to the dictionary of the extracted ImageNet validation set. The rebuilded ImageNet validation set will be stored in `../../results/imagenet-val.npz`.

**Step 6:** Perform model stealing with "correct"/"wrong"/"scratch" surrogate model (when the targeted service is `rn50-hf + cifar10` and will return `logit-label` for queries):

```bash
# model stealing with "correct" surrogate model
bash furatk/correct-steal.sh ../../

# model stealing with "wrong" surrogate model
bash furatk/wrong-steal.sh ../../

# model stealing with "scratch" surrogate model
bash furatk/scratch-steal.sh ../../
