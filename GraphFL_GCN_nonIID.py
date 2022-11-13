#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from gnn_ml_update_nonIID import LocalUpdate_GCN, test_inference_GCN

from gnn_ml_models import GCN
from utils_gnn import *




if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'


    if args.dataset == 'cora' or args.dataset == 'citeseer':
        # load dataset and user groups
        adj, features, labels, idx_train, idx_qry, idx_test, user_groups, user_qry_groups = \
                load_citation(args.dataset, args.normalization, args.train_size, args.num_users)

    else:
        if not args.is_split:
            ### droopit
            adj, features, labels = \
                get_dataset(args.dataset, data_path='./data/'+args.dataset+'.npz', standardize=True, train_examples_per_class=None, val_examples_per_class=None)
            #print(labels.shape)     
            
        else:
            sub_adj, sub_fea, adj, features, labels, idxs_st_ed = \
                get_sub_dataset(args.dataset, data_path='./data/'+args.dataset+'.npz', standardize=True, num_subgraph=args.num_users, ratio_overlap=args.ratio_overlap)
           
        idx_train, idx_qry, idx_test, user_groups, user_qry_groups = \
                            get_train_val_test_split(random_state= np.random.RandomState(args.seed), labels=labels,
                                 num_users=args.num_users, train_examples_per_class=None, val_examples_per_class=None,
                                 test_examples_per_class=None,
                                 train_size=args.train_size, val_size=300, test_size=None)
        labels = torch.LongTensor(labels)  


    print('#train:', len(idx_train), '#test:', len(idx_test))

    if args.dataset == 'cora':
        nclass = 7

    if args.dataset == 'citeseer':
        nclass = 6

    if args.dataset == 'cora_full':
        nclass = 70

    if args.dataset == 'ms_academic_cs':
        nclass = 15



    # BUILD MODEL
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)

    global_model = GCN(nfeat=features.size(1), nhid=args.hidden, nclass=nclass, dropout=args.dropout)
    
    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    #print(global_model)

    # copy weights
    global_weights = global_model.state_dict()


    # Training
    test_accuracy = []
    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #print(idxs_users)

        local_model = LocalUpdate_GCN(args=args, features=features, adj=adj, labels=labels,
                                  user_groups=user_groups, user_qry_groups=user_qry_groups, 
                                  idxs_users=idxs_users, logger=logger)
        
        ## Meta-train: Learn a shared model based on support/query set
        shared_model, _ = local_model.forward(model=copy.deepcopy(global_model))
        #print('train loss:', loss)
        
        ## Local client finetunes the shared model using its support set 
        for idx in idxs_users:
            
            w, loss = local_model.finetuning(model=copy.deepcopy(shared_model), idxs=user_groups[idx])
        
            local_weights.append(copy.deepcopy(w))
            #local_losses.append(copy.deepcopy(loss))
            local_losses.append((loss))
            #print('local weight:', local_weights)
        
        # Global average: update global weights
        global_weights = average_weights(local_weights)
        #print('global weights:', global_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        #train_loss.append(loss_avg)
        # print('train loss:', loss_avg)

        # Test inference after completion of training
        test_acc, test_loss = test_inference_GCN(args, global_model, features, adj, labels, idx_test)

        print(f' \n Results after {args.epochs} global rounds of training:')
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

        test_accuracy.append(test_acc)

    print('max acc:', max(test_accuracy))


