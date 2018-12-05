#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:29:49 2018

@author: gaoyi
"""

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 50
image_size = 28

# params for source dataset
src_dataset = "MNIST"
src_encoder_restore = '/home/gaoyi/domain_adapation/ADDA/checkpoints/ADDA-source-encoder-final.pt'
src_classifier_restore = '/home/gaoyi/domain_adapation/ADDA/checkpoints/ADDA-source-classifier-final.pt'
src_model_trained = True

# params for target dataset
tgt_dataset = 'mnist_m'
tgt_dataset_root = '/home/gaoyi/domain_adapation/DANN/data/mnist_m'
tgt_encoder_restore = '/home/gaoyi/domain_adapation/ADDA/checkpoints/ADDA-target-encoder-final.pt'
tgt_model_trained = True

# params for training network
num_gpu = 1
num_epochs_pre = 30
num_epochs = 30
log_step_pre = 20
log_step = 30
eval_step_pre = 5 
save_step_pre = 5
save_step = 5
manual_seed = None
image_size = 28

d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = '/home/gaoyi/domain_adapation/ADDA/checkpoints/ADDA-critic-final.pt'

# params for optimizing models
c_learning_rate = 1e-4
d_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9

model_root = '/home/gaoyi/domain_adapation/ADDA/checkpoints'

