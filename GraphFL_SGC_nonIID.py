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

from options_sgc import args_parser
from gnn_ml_update_nonIID import LocalUpdate_SGC, test_inference_SGC

from gnn_ml_models import SGC
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

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    if args.dataset == 'cora' or args.dataset == 'citeseer':
        # load dataset and user groups
        adj, features, labels, idx_train, idx_qry, idx_test, user_groups, user_qry_groups = \
                load_citation(args.dataset, args.normalization, args.per_class, args.num_users)

    else:
        ### droopit
        adj, features, labels = \
            get_dataset(args.dataset, data_path='./data/'+args.dataset+'.npz', standardize=True, train_examples_per_class=None, val_examples_per_class=None)
        #print(labels.shape)
        idx_train, idx_val, idx_test, user_groups, user_qry_groups = \
                        get_train_val_test_split(random_state= np.random.RandomState(args.seed), labels=labels,
                             num_users=args.num_users, train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=args.train_size, val_size=300, test_size=None)
        labels = torch.LongTensor(labels)       
        #print(idx_train)

    # print('fea:', features)
    # print('adj:', adj)

    print('#train:', len(idx_train), '#test:', len(idx_test))

    if args.dataset == 'cora':
        nclass = 7

    if args.dataset == 'citeseer':
        nclass = 6

    if args.dataset == 'cora_full':
        nclass = 70

    if args.dataset == 'ms_academic_cs':
        nclass = 15


    features = sgc_precompute(features, adj, args.degree)
    global_model = SGC(nfeat=features.size(1), nclass=nclass)

    #print('size:', features.size())

    # BUILD MODEL
    features = features.to(device)
    labels = labels.to(device)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()
    #print(global_weights)


    # Training
    train_loss, test_accuracy = [], []
    print_every = 2
    
    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #print(idxs_users)

        local_model = LocalUpdate_SGC(args=args, features=features, labels=labels,
                                  user_groups=user_groups, user_qry_groups=user_qry_groups, 
                                  idxs_users=idxs_users, logger=logger)
        
        shared_model, _ = local_model.forward(model=copy.deepcopy(global_model))
        #print('train loss:', loss)
        
        for idx in idxs_users:
            
            w, loss = local_model.finetuning(model=copy.deepcopy(shared_model), idxs=user_groups[idx])
        
            local_weights.append(copy.deepcopy(w))
            #local_losses.append(copy.deepcopy(loss))
            local_losses.append((loss))
            #print('local weight:', local_weights)
        
        #   print('local loss:', local_losses)


        # update global weights
        global_weights = average_weights(local_weights)
        #print('global weights:', global_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        #train_loss.append(loss_avg)
        # print('train loss:', loss_avg)


        # # Calculate avg training accuracy over all users at every epoch
        # list_acc, list_loss = [], []

        # global_model.eval()
        # for c in range(args.num_users):
        #     local_model = LocalUpdate_GCN(args=args, features=features, adj=adj, labels=labels,
        #                               idxs=user_groups[c], idxs_qry=user_qry_groups[idx], logger=logger)
        #     acc, loss = local_model.inference(model=global_model)
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        #     # print('list acc:',list_acc)
        #     # print('list loss:', list_loss)

        # train_accuracy.append(sum(list_acc)/len(list_acc))
        # print('train acc:', train_accuracy)

        # # print global training loss after every 'i' rounds
        # if (epoch+1) % print_every == 0:
        #     print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        #     print(f'Training Loss : {np.mean(np.array((train_loss)))}')
        #     print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))


        # Test inference after completion of training
        test_acc, test_loss = test_inference_SGC(args, global_model, features, labels, idx_test)

        #print(f' \n Results after {args.epochs} global rounds of training:')
        #print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

        test_accuracy.append(test_acc)

    print('max acc:', max(test_accuracy))
