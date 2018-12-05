#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:14:19 2018

@author: gaoyi
"""

import params
from utils import get_data_loader, init_model, init_random_seed
from models.discriminator import Discriminator
from models.lenet import LeNetClassifier,LeNetEncoder2, LeNetEncoder
from core import pretrain , adapt , test
if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)
    
    #set loader
    src_data_loader = get_data_loader(params.src_dataset)
    src_data_loader_eval = get_data_loader(params.src_dataset,train=False)
    tgt_data_loader = get_data_loader(params.tgt_dataset)
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)
    
    src_encoder = init_model(net=LeNetEncoder(),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=LeNetClassifier(),
                                restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=LeNetEncoder2(),
                             restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)
    # SRC domain
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)
    

    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        
        src_encoder, src_classifier = pretrain.train_src(
            src_encoder, src_classifier, src_data_loader)
        
    print("=== Evaluating classifier for source domain ===")
    pretrain.eval_src(src_encoder, src_classifier, src_data_loader_eval)
    
    # TGT domain    
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)
    
    
    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = adapt.train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    test.eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    test.eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)

    

