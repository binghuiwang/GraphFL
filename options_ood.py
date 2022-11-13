#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=11,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=3,
                        help="number of users: K")
    parser.add_argument('--per_class', type=int, default=1,
                        help="number of labeled samples per class")
    parser.add_argument('--frac', type=float, default=0.2,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=50,
                        help="the number of local epochs: E")
    parser.add_argument('--local_ep_test', type=int, default=50,
                        help="the number of fine-tuning epochs: E")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='meta learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--train_size', type=int, help='Number of hidden units', default=300)
    parser.add_argument('--is_split', default=True, help="whether spliting the graph")
    parser.add_argument('--ratio_overlap', type=float, default=0.0, help='overlapped ratio between subgraphs')
    

    parser.add_argument('--n_way', type=int, help='n class in each task', default=2)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    parser.add_argument('--k_qry_train', type=int, help='k shot for query set', default=5)
    parser.add_argument('--k_qry_test', type=int, help='k shot for query set', default=15)
    

    
    # model arguments
    parser.add_argument('--model', type=str, default='GCN', help='model name')
    parser.add_argument('--num_classes', type=int, default=7, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=42, help='random seed')


    parser.add_argument('--hidden', type=int, help='Number of hidden units', default=16)
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset to use.')
    parser.add_argument('--normalization', type=str, default='AugNormAdj', help='Normalization method for the adjacency matrix.')
    parser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    return args
