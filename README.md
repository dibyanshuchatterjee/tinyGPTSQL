# tinyGPTSQL
This repository showcases the pre-training phase of a GPT like model that is trained on SQL data. The model was trained on a M1 processor and on mps GPU. The pretraining process uses pytorch's mac version to utilize the mps GPU.

# Overview
This project implements a SQL Writer model based on the GPT (Generative Pretrained Transformer) architecture. The model is trained to generate SQL queries given a context. The implementation closely follows the official GPT paper.

# Model Architecture
The model consists of a transformer architecture with self-attention mechanism. It has the following components:

## -- GPT Model: The main model composed of a series of transformer blocks. It takes input indices, token embeddings, and position embeddings to generate SQL queries.

## -- MultiHeadAttention: Implements the self-attention mechanism with multiple attention heads in parallel.

## -- FeedFoward: A simple linear layer followed by a non-linearity.

## -- Block: Represents a transformer block, combining multi-head self-attention and feedforward layers.

# Training
The model is trained on SQL data using independent sequences processed in parallel. The training process includes estimating loss, preventing gradient accumulation, and updating model parameters using AdamW optimizer.

# Hyperparameters
## num_batch: Number of independent sequences processed in parallel.
## context_size: Maximum context length.
## max_iters: Maximum number of training iterations.
## eval_interval: Interval for evaluating and printing training and validation losses.
## earning_rate: Learning rate for the optimizer.
## eval_iters: Number of iterations for evaluation during loss estimation.
## n_embd: Dimensionality of the embeddings.
## n_head: Number of attention heads in the self-attention mechanism.
## n_layer: Number of transformer blocks.
## dropout: Dropout probability.

# Dependencies
Pytorch

# Usage

-- Run own_test.py to generate sql text from the prompt SELECT
-- The model can be trained using LLM-Pretraining.py file

