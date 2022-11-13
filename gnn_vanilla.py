import torch
import torch.nn.functional as F
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim

from itertools import combinations
from utils import load_citation, sgc_precompute, set_seed, accuracy
from meta import Meta
from data_generator import *
from sklearn import preprocessing 
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math

from copy import deepcopy


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        #return x
        return F.log_softmax(x, dim=1)


class SGC(nn.Module):
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)


def train_gcn(model, features, train_labels, idx_train, epochs, weight_decay, lr, adj):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], train_labels)
        #loss_train = F.nll_loss(output[idx_train], train_labels[idx_train])
        #loss_train = F.cross_entropy(output[idx_train], train_labels[idx_train])
        loss_train.backward()
        optimizer.step()


        with torch.no_grad():
            model.eval()
            output = model(features, adj)
            #print(output)
            #print(train_labels)
            acc_train = accuracy(output[idx_train], train_labels)

    return model, acc_train
#Train Model


def test_gcn(model, features, test_labels, idx_test, adj):
    model.eval()
    output = model(features, adj)

    return accuracy(output[idx_test], test_labels)
#Test Model



def train_sgc(model, train_features, train_labels, epochs, weight_decay, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()


    with torch.no_grad():
        model.eval()
        output = model(train_features)
        print(output)
        #print(train_labels)
        acc_train = accuracy(output, train_labels)

    #print(model.parameters)

    return model, acc_train


def test_sgc(model, test_features, test_labels):
    model.eval()
    output = model(test_features)
    #print(output)
    return accuracy(output, test_labels)



def main(args):


    adj, features, labels = load_citation(args.dataset, args.normalization)

    if args.model == 'SGC':
        features = sgc_precompute(features, adj, args.degree)


    label_encoder = preprocessing.LabelEncoder() 

    #device = torch.device('cuda')
    device = torch.device('cpu')


    x_spt_idx, y_spt_idx, x_qry_idx, y_qry_idx = [], [], [], []
    
    if args.model == 'SGC':
        model = SGC(nfeat=features.size(1), nclass=args.n_way)

    if args.model == 'GCN':
        model = GCN(nfeat=features.size(1), nhid=args.hidden, nclass=args.n_way, dropout=args.dropout)


    for task in range(args.task_num):
        
        x_spt_one, y_spt_one, x_qry_one, y_qry_one = read_data('traintest/'+str(args.dataset)+'_train_task_'+str(task+1)+'_shot_'+str(args.k_spt)+'_way_'+str(args.n_way) + '_query_train_' + str(args.k_qry_train) )  
        #print(x_spt_one)

        if args.model == 'SGC':
            model, acc_train = train_sgc(model, features[x_spt_one], torch.tensor(label_encoder.fit_transform(y_spt_one)), args.epoch, args.weight_decay, args.lr)
        
        if args.model == 'GCN':
            model, acc_train = train_gcn(model, features, torch.tensor(label_encoder.fit_transform(y_spt_one)), x_spt_one, args.epoch, args.weight_decay, args.lr, adj)
        
        print('acc_train:', acc_train)

        
    for task in range(args.task_num):
        x_spt_one, y_spt_one, x_qry_one, y_qry_one = read_data('traintest/'+str(args.dataset)+'_test_task_'+str(task+1)+'_shot_1_way_'+str(args.n_way) + '_query_test_' + str(args.k_qry_test))  
        
        x_qry_idx += x_qry_one
        y_qry_idx += y_qry_one

    print('#test:', len(y_qry_idx))

    if args.model == 'SGC':
        acc_test = test_sgc(model, features[x_qry_idx], torch.tensor(label_encoder.fit_transform(y_qry_idx)))
    
    if args.model == 'GCN':
        acc_test = test_gcn(model, features, torch.tensor(label_encoder.fit_transform(y_qry_idx)), x_qry_idx, adj)

    print("Train Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_train, acc_test))
    


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--model', type=str, help='model name', default='GCN')
    argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=0.001)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--total_task_num', type=int, help='total tasks', default=50)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry_train', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--k_qry_test', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--hidden', type=int, help='Number of hidden units', default=16)
    argparser.add_argument('--weight_decay', type=float, default=5e-2, help='Weight decay (L2 loss on parameters).')
    argparser.add_argument('--dataset', type=str, default='cora', help='Dataset to use.')
    argparser.add_argument('--normalization', type=str, default='AugNormAdj', help='Normalization method for the adjacency matrix.')
    argparser.add_argument('--seed', type=int, default=42, help='Random seed.')
    argparser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')
    argparser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')


    args = argparser.parse_args()

    main(args)
