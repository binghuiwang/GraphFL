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
from gnn_ml_update_boosttrain import LocalUpdate_GCN, test_inference_GCN

from gnn_ml_models import GCN
from utils_gnn import *

import torch.optim as optim
import torch.nn.functional as F
#from utils_alg import *



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
                load_citation(args.dataset, args.normalization, args.train_size, args.num_users)

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


    # BUILD MODEL
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)


    user_groups_ext,  labels_ext = [[] for _ in range(args.num_users)], [[] for _ in range(args.num_users)]

    if args.boost_train == "selftrain":
        ### First run self-train to generate pseudo labels
        global_model = GCN(nfeat=features.size(1), nhid=args.hidden, nclass=nclass, dropout=args.dropout)
        global_model.to(device)

        optimizer = optim.Adam(global_model.parameters(),
                        lr=0.01, weight_decay=args.weight_decay)

        
        loss, acc = [], []
        for idx in range(args.num_users):

            for epoch in range(args.local_ep):

                global_model.train()
                global_model.zero_grad()
                #optimizer.zero_grad()
                output = global_model(features, adj)
                #print(len(user_groups[idx]))
                loss_train = F.nll_loss(output[list(user_groups[idx])], labels[list(user_groups[idx])])
                #acc_train = accuracy(output[list(user_groups[idx])], labels[list(user_groups[idx])])
                loss_train.backward()
                optimizer.step()

            global_model.eval()
            output = global_model(features, adj)
            #print(output)

            user_groups_ext[idx], labels_ext[idx] = selftraining(output.detach().numpy(), nclass, \
                args.pseudo_labels, user_groups[idx], labels.detach().numpy())


    



