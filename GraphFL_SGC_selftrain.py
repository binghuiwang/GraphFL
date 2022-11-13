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
from gnn_ml_update_boosttrain import LocalUpdate_SGC, test_inference_SGC

from gnn_ml_models import SGC
from utils_gnn import *


import torch.optim as optim
import torch.nn.functional as F



if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # if args.gpu:
    #     torch.cuda.set_device(args.gpu)
    # device = 'cuda' if args.gpu else 'cpu'

    device = 'cpu'
    print('device:', device)

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

    if args.dataset == 'pubmed':
        nclass = 3

    if args.dataset == 'cora_full':
        nclass = 70

    if args.dataset == 'ms_academic_cs':
        nclass = 15



    features = sgc_precompute(features, adj, args.degree)
    
    features = features.to(device)
    labels = labels.to(device)
    adj = adj.to(device)

    
    ### First run self-train to generate pseudo labels
    global_model = SGC(nfeat=features.size(1), nclass=nclass)
    global_model.to(device)


    optimizer = optim.Adam(global_model.parameters(),
                    lr=0.2, weight_decay=5e-6)

    user_groups_ext,  labels_ext = [[] for _ in range(args.num_users)], [[] for _ in range(args.num_users)]

    if args.boost_train == "selftrain":
        ##### GCN: individual training 
        loss, acc = [], []
        for idx in range(args.num_users):

            for epoch in range(args.local_ep):

                global_model.train()
                global_model.zero_grad()
                #optimizer.zero_grad()
                output = global_model(features[list(user_groups[idx])])
                #print(len(user_groups[idx]))
                loss_train = F.cross_entropy(output, labels[list(user_groups[idx])])
                #acc_train = accuracy(output[list(user_groups[idx])], labels[list(user_groups[idx])])
                loss_train.backward()
                optimizer.step()


            global_model.eval()
            output = global_model(features)
            #output = global_model(features[idx_test])

            user_groups_ext[idx], labels_ext[idx] = selftraining(output.detach().numpy(), nclass, \
                args.pseudo_labels, user_groups[idx], labels.detach().numpy())
            #loss_test = F.cross_entropy(output, labels[idx_test])


    if args.boost_train == "cotrain":

        for idx in range(args.num_users):
            user_groups_ext[idx], labels_ext[idx] = cotraining(adj, args.pseudo_labels, \
                user_groups[idx], labels.detach().numpy())


    if args.boost_train == "lp":
        pass

    labels_ext = torch.LongTensor(labels_ext)
    #print('labels_ext', labels_ext)

    #print('size:', features.size())
    global_model = SGC(nfeat=features.size(1), nclass=nclass)

    # BUILD MODEL
    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()
    #print(global_weights)


    # Training
    test_accuracy = []
    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #print(idxs_users)

        local_model = LocalUpdate_SGC(args=args, features=features, labels=labels_ext,
                                  user_groups=user_groups_ext, user_qry_groups=user_qry_groups, 
                                  idxs_users=idxs_users, logger=logger)
        
        shared_model, _ = local_model.forward(model=copy.deepcopy(global_model))
        #print('train loss:', loss)
        
        for idx in idxs_users:
            
            w, loss = local_model.finetuning(model=copy.deepcopy(shared_model), \
                idxs=user_groups_ext[idx], labels=labels_ext[idx])
        
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


        # Test inference after completion of training
        test_acc, test_loss = test_inference_SGC(args, global_model, features, labels, idx_test)

        #print(f' \n Results after {args.epochs} global rounds of training:')
        #print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

        test_accuracy.append(test_acc)

    print('max acc:', max(test_accuracy))
