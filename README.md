# EDGE

Official pytorch implementation for ["Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling"](https://arxiv.org/pdf/2305.04111.pdf). Our code is devloped based on https://github.com/ehoogeboom/multinomial_diffusion. 

We use the evaluation modules provided by https://github.com/uoguelph-mlrg/GGM-metrics and https://github.com/hheidrich/CELL.

## Environment requirement 
```
dgl
prettytable
scikit-learn
tensorboard
tensorflow
tensorflow-gan
torch
torch-geometric
tqdm
wandb
```
and dependencies from  https://github.com/uoguelph-mlrg/GGM-metrics, https://github.com/hheidrich/CELL and https://github.com/ehoogeboom/multinomial_diffusion.

## Training your degree sequence model
See node.ipynb, once you train the model, it's saved to the "./graphs" directory.

## Training script
🌟**IMPORTANT note on running EDGE for your own datasets**: Do not use large diffusion steps for small graphs with less than 100 nodes, for those small graph datasets, please try #diffusion steps={8,16,32,64}

### 1. Training template for generic graph datasets
By default, we use an empirical degree sampler, which randomly takes a degree sequence from the training data as $d^0$ to perform degree guidance. You can replace the keyword `empirical` with `neural` in the option `--empty_graph_sampler` if you have trained your neural degree sampler.
```
#!/bin/bash

python train.py \
        --epochs 50000 \
        --num_generation 64 \
        --diffusion_dim 64 \
        --diffusion_steps 128 \
        --device cuda:1 \
        --dataset Ego \
        --batch_size 8 \
        --clip_value 1 \
        --lr 1e-4 \
        --optimizer adam \
        --final_prob_edge 1 0 \
        --sample_time_method importance \
        --check_every 500 \
        --eval_every 500 \
        --noise_schedule linear \
        --dp_rate 0.1 \
        --loss_type vb_ce_xt_prescribred_st \
        --arch TGNN_degree_guided \
        --parametrization xt_prescribed_st \
        --empty_graph_sampler empirical \     
        --degree \
        --num_heads 8 8 8 8 1 
```

### 2. training template for large network datasets
```
#!/bin/bash

python train.py \
        --epochs 50000 \
        --num_generation 64 \
        --num_iter 256 \
        --diffusion_dim 64 \
        --diffusion_steps 512 \
        --device cuda:0 \
        --dataset polblogs \
        --batch_size 4 \
        --clip_value 1 \
        --lr 1e-4 \
        --optimizer adam \
        --final_prob_edge 1 0 \
        --sample_time_method importance \
        --check_every 50 \
        --eval_every 50 \
        --noise_schedule linear \
        --dp_rate 0.1 \
        --loss_type vb_ce_xt_prescribred_st \
        --arch TGNN_degree_guided \
        --parametrization xt_prescribed_st \
        --degree \
        --num_heads 8 8 8 8 1 
```
Evaluation is done every `eval_every` epochs. You can also re-evaluate a specific checkpoint using the script below. 

## Evaluation script
```
python evaluate.py \
        --run_name 2023-05-29_18-29-35 \
        --dataset polblogs \
        --num_samples 8 \
        --checkpoints 5500
```

## Results
Training results can be found in wandb/{dataset_name}/multinomial_diffusion/multistep/{run_name}

## Work in progress
We are still working on integrating the following two features into our code:

 * [ ]  Even faster sampling by incrementally modifying graph.
 
 * [ ]  Attributed graph generation.
