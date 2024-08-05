# PEI Attack vs Text Classification Services

Example scripts and configurations for text classification experiments are collected in [scripts/exp-text](../scripts/exp-text) and [configs/exp-text](../configs/exp-text) respectively.

Here we present a brief tutorial on how to run experiments with these scripts and configurations (Please check **comments in these scripts** to see how to set encoder, downstream task, etc.).

**Step 0:** Enter the folder:

```
cd ./scripts/exp-text
```

**Step 1:** Build the embedding dataset, e.g., `bertbase + sst5`:

```
bash embed.sh ../../
```

**Step 2:** Train the downstream classifier on the built embedding dataset e.g., `bertbase + sst5`, with the configuration [configs/exp-text/train.yaml](../configs/exp-text/train.yaml):

```
bash train.sh ../../
```

**Step 3:** Synthesize PEI attack texts for a candidate encoder (e.g., `bertbase`) with the configuration [configs/exp-text/atk.yaml](../configs/exp-text/atk.yaml) (PEI objective texts are presented in the script):

```
bash atk.sh ../../
```

The used PEI objective texts are generated via [scripts/gen_obj_text.py](../scripts/gen_obj_text.py).

**Step 4:** Calculate the PEI score of a candidate encoder (e.g., `bertlarge`) on a targeted downstream service (e.g., `bertbase + sst5`) with the configuration [configs/exp-text/eval.yaml](../configs/exp-text/eval.yaml):

```
bash eval.sh ../../
```

Obtained PEI scores can then be used to calculate **PEI z-scores** to conduct the PEI attack (see the paper for details).
