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

from options_ood import args_parser
from gnn_ml_update_ood import LocalUpdate_GCN
from gnn_ml_models import GCN
from utils_gnn import *


from sklearn import preprocessing


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

        if not args.is_split:
            ### droopit
            adj, features, labels = \
                get_dataset(args.dataset, data_path='./data/'+args.dataset+'.npz', standardize=True, train_examples_per_class=None, val_examples_per_class=None)
            #print(labels.shape)     
        else:
            sub_adj, subgraph_fea, features, labels = \
                get_sub_dataset(args.dataset, data_path='./data/'+args.dataset+'.npz', standardize=True, num_subgraph=args.num_users, ratio_overlap=args.ratio_overlap)
           
        idx_train, idx_qry, idx_test, user_groups, user_qry_groups = \
                            get_train_val_test_split(random_state= np.random.RandomState(args.seed), labels=labels,
                                 num_users=args.num_users, train_examples_per_class=None, val_examples_per_class=None,
                                 test_examples_per_class=None,
                                 train_size=args.train_size, val_size=300, test_size=None)
    
    label_encoder = preprocessing.LabelEncoder() 

    x_spt_test, y_spt_test, x_qry_test, y_qry_test = [], [], [], []

    task_num = int(args.frac * args.num_users)
    print('task_num', task_num)

    for task in range(task_num):
        #x_spt_one, y_spt_one, x_qry_one, y_qry_one = read_data('traintest/'+str(args.dataset)+'_test_task_'+str(task+1)+'_shot_1_way_'+str(args.n_way) + '_query_test_' + str(args.k_qry_test))  
        x_spt_one, y_spt_one, x_qry_one, y_qry_one = read_data('traintest/'+str(args.dataset)+'_test_task_'+str(task+1)+'_shot_'+str(args.k_spt)+'_way_'+str(args.n_way) + '_query_test_' + str(args.k_qry_test))  
        x_spt_test.append(x_spt_one)
        y_spt_test.append(torch.tensor(label_encoder.fit_transform(y_spt_one)))
        x_qry_test.append(x_qry_one)
        y_qry_test.append(torch.tensor(label_encoder.fit_transform(y_qry_one)))


    if args.dataset == 'cora':
        nclass = 7

    if args.dataset == 'citeseer':
        nclass = 6

    if args.dataset == 'cora_full':
        nclass = 70

    if args.dataset == 'ms_academic_cs':
        nclass = 15

    # BUILD MODEL
    global_model = GCN(nfeat=features.size(1), nhid=args.hidden, nclass=nclass, dropout=args.dropout)

    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)

    # Set the model to train and send it to device.
    global_model.to(device)
    
    global_model.train()
    print(global_model)


    # copy weights
    global_weights = global_model.state_dict()
    #print(global_weights)

    # Training
    max_test_accuracy = []
    
    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()

        m = max(int(task_num), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(idxs_users)

        x_spt_train, y_spt_train, x_qry_train, y_qry_train = [], [], [], []
        
        for idx in idxs_users:
            x_spt_one, y_spt_one, x_qry_one, y_qry_one = read_data('traintest/'+str(args.dataset)+'_train_task_'+str(idx+1)+'_shot_'+str(args.k_spt)+'_way_'+str(args.n_way) + '_query_train_' + str(args.k_qry_train) )  
            x_spt_train.append(x_spt_one)
            y_spt_train.append(torch.tensor(label_encoder.fit_transform(y_spt_one)))
            x_qry_train.append(x_qry_one)
            y_qry_train.append(torch.tensor(label_encoder.fit_transform(y_qry_one)))


        local_model = LocalUpdate_GCN(args=args, features=features, adj=adj, labels=labels)
        # local_model = LocalUpdate_GCN(args=args, features=features, adj=adj, labels=labels,
        #                          idxs_users=idxs_users, x_spt_train=x_spt_train, y_spt_train=y_spt_train, 
        #                          x_qry_train=x_qry_train, y_qry_train=y_qry_train)
        
        for task_idx in range(len(idxs_users)):
            w, loss = local_model.forward(copy.deepcopy(global_model), x_spt_train[task_idx], y_spt_train[task_idx], \
                x_qry_train[task_idx], y_qry_train[task_idx])
            local_weights.append(copy.deepcopy(w))
            local_losses.append((loss))
        #print('train loss:', local_losses)
        
        
        # update global weights
        global_weights = average_weights(local_weights)
        #print('global weights:', global_weights)
        # update global weights
        global_model.load_state_dict(global_weights)

        #loss_avg = sum(local_losses) / len(local_losses)
        #train_loss.append(loss_avg)
        # print('train loss:', loss_avg)


        meta_test_acc = []
        for task_idx in range(task_num):
            test_accs = local_model.finetuning(copy.deepcopy(global_model), \
                x_spt_test[task_idx], y_spt_test[task_idx], \
                x_qry_test[task_idx], y_qry_test[task_idx])
            meta_test_acc.append(test_accs)

        mean_test_acc = np.array(meta_test_acc).mean(axis=0)
        print('mean acc:', mean_test_acc)
        print('max mean:', np.max(mean_test_acc))
        max_test_accuracy.append(np.max(mean_test_acc))

    print('max local epoch:', max_test_accuracy)
    # print('max global round:', np.max(max_test_accuracy))