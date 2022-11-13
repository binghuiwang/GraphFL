#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils_gnn import accuracy
import torch.nn.functional as F
import numpy as np
import copy
import collections

class LocalUpdate_GCN(nn.Module):
    def __init__(self, args, features, adj, labels, user_groups, user_qry_groups, idxs_users, logger):
        self.args = args
        self.logger = logger

        self.features = features
        self.adj = adj
        self.labels = labels
        self.user_groups = user_groups
        self.user_qry_groups = user_qry_groups
        self.idxs_users = idxs_users

        self.device = 'cuda' if args.gpu else 'cpu'
        


    def forward(self, model):
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.meta_lr,
                                         weight_decay=self.args.weight_decay)


        losses_q = [0 for _ in range(self.args.local_ep + 1)]
        corrects = [0 for _ in range(self.args.local_ep + 1)]


        for idx in self.idxs_users:
    
            outputs = model(self.features, self.adj, vars=None)
            loss = F.nll_loss(outputs[list(self.user_groups[idx])], self.labels[idx][list(self.user_groups[idx])])
            grad = torch.autograd.grad(loss, model.parameters())
            fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, model.parameters())))

            #print('fast_weights:', fast_weights[0])

            with torch.no_grad():
                outputs_q = model(self.features, self.adj, model.parameters())
                loss_q = F.nll_loss(outputs_q[list(self.user_qry_groups[idx])], self.labels[idx][list(self.user_qry_groups[idx])])
                losses_q[0] += loss_q
                correct = accuracy(outputs_q[list(self.user_qry_groups[idx])], self.labels[idx][list(self.user_qry_groups[idx])]).item()
                corrects[0] = corrects[0] + correct

            with torch.no_grad():
                outputs_q = model(self.features, self.adj, fast_weights)
                loss_q = F.nll_loss(outputs_q[list(self.user_qry_groups[idx])], self.labels[idx][list(self.user_qry_groups[idx])])
                losses_q[1] += loss_q
                correct = accuracy(outputs_q[list(self.user_qry_groups[idx])], self.labels[idx][list(self.user_qry_groups[idx])]).item()
                corrects[1] = corrects[1] + correct


            for k in range(1, self.args.local_ep):

                outputs = model(self.features, self.adj, fast_weights)
                loss = F.nll_loss(outputs[list(self.user_groups[idx])], self.labels[idx][list(self.user_groups[idx])])
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, fast_weights)))

                outputs = model(self.features, self.adj, fast_weights)
                loss_q = F.nll_loss(outputs[list(self.user_qry_groups[idx])], self.labels[idx][list(self.user_qry_groups[idx])])
                losses_q[k + 1] += loss_q

                #print('loss:', loss_q)
                with torch.no_grad():
                    correct = accuracy(outputs[list(self.user_qry_groups[idx])], self.labels[idx][list(self.user_qry_groups[idx])]).item()
                    corrects[k + 1] = corrects[k + 1] + correct


        loss_q = losses_q[-1] / len(self.idxs_users)
        optimizer.zero_grad()
        loss_q.backward()
        optimizer.step()

        return model, sum(losses_q) / len(losses_q)



    def finetuning(self, model, idxs, labels):
        

        update_paras = collections.OrderedDict()

        outputs = model(self.features, self.adj, vars=None)
        loss = F.nll_loss(outputs[list(idxs)], labels[list(idxs)])
        grad = torch.autograd.grad(loss, model.parameters())
        fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, model.parameters())))

        for k in range(1, self.args.local_ep_test):
            outputs = model(self.features, self.adj, fast_weights)
            loss = F.nll_loss(outputs[list(idxs)], labels[list(idxs)])
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, fast_weights)))

        #model.load_state_dict(fast_weights)
        update_paras['vars.0'] = torch.tensor(fast_weights[0])
        update_paras['vars.1'] = torch.tensor(fast_weights[1])
        update_paras['vars.2'] = torch.tensor(fast_weights[2])
        update_paras['vars.3'] = torch.tensor(fast_weights[3])

        return update_paras, loss



def test_inference_GCN(args, model, features, adj, labels, idx_test):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    device = 'cuda' if args.gpu else 'cpu'

    output = model(features, adj)
    loss = F.nll_loss(output[idx_test], labels[idx_test])
    acc = accuracy(output[idx_test], labels[idx_test]) / len(idx_test)

    return acc, loss


class LocalUpdate_SGC(nn.Module):

    def __init__(self, args, features, labels, user_groups, user_qry_groups, idxs_users, logger):
        self.args = args
        self.logger = logger

        self.features = features

        self.labels = labels
        self.user_groups = user_groups
        self.user_qry_groups = user_qry_groups
        self.idxs_users = idxs_users

        self.device = 'cuda' if args.gpu else 'cpu'
        


    def forward(self, model):
        
        #model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.meta_lr,
                                         weight_decay=self.args.weight_decay)


        losses_q = [0 for _ in range(self.args.local_ep + 1)]
        corrects = [0 for _ in range(self.args.local_ep + 1)]


        for idx in self.idxs_users:
    
            outputs = model(self.features[list(self.user_groups[idx])], vars=None)
            #loss = F.nll_loss(outputs, self.labels[idx][list(self.user_groups[idx])])
            loss = F.cross_entropy(outputs, self.labels[idx][list(self.user_groups[idx])])
            grad = torch.autograd.grad(loss, model.parameters())
            fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, model.parameters())))

            #print('fast_weights:', fast_weights[0])

            with torch.no_grad():
                outputs_q = model(self.features[list(self.user_qry_groups[idx])], model.parameters())
                #loss_q = F.nll_loss(outputs_q, self.labels[idx][list(self.user_qry_groups[idx])])
                loss_q = F.cross_entropy(outputs_q, self.labels[idx][list(self.user_qry_groups[idx])])
                losses_q[0] += loss_q
                correct = accuracy(outputs_q, self.labels[idx][list(self.user_qry_groups[idx])]).item()
                corrects[0] = corrects[0] + correct

            with torch.no_grad():
                outputs_q = model(self.features[list(self.user_qry_groups[idx])], fast_weights)
                #loss_q = F.nll_loss(outputs_q, self.labels[idx][list(self.user_qry_groups[idx])])
                loss_q = F.cross_entropy(outputs_q, self.labels[idx][list(self.user_qry_groups[idx])])
                losses_q[1] += loss_q
                correct = accuracy(outputs_q, self.labels[idx][list(self.user_qry_groups[idx])]).item()
                corrects[1] = corrects[1] + correct


            for k in range(1, self.args.local_ep):

                outputs = model(self.features[list(self.user_groups[idx])], fast_weights)
                #loss = F.nll_loss(outputs, self.labels[idx][list(self.user_groups[idx])])
                loss = F.cross_entropy(outputs, self.labels[idx][list(self.user_groups[idx])])
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, fast_weights)))

                outputs = model(self.features[list(self.user_qry_groups[idx])], fast_weights)
                #loss_q = F.nll_loss(outputs, self.labels[idx][list(self.user_qry_groups[idx])])
                loss_q = F.cross_entropy(outputs, self.labels[idx][list(self.user_qry_groups[idx])])
                losses_q[k + 1] += loss_q

                #print('loss:', loss_q)
                with torch.no_grad():
                    correct = accuracy(outputs, self.labels[idx][list(self.user_qry_groups[idx])]).item()
                    corrects[k + 1] = corrects[k + 1] + correct


        loss_q = losses_q[-1] / len(self.idxs_users)
        optimizer.zero_grad()
        loss_q.backward()
        optimizer.step()

        return model, sum(losses_q) / len(losses_q)



    def finetuning(self, model, idxs, labels):
        

        update_paras = collections.OrderedDict()

        outputs = model(self.features[list(idxs)], vars=None)
        #loss = F.nll_loss(outputs, self.labels[idx][list(idxs)])
        loss = F.cross_entropy(outputs, labels[list(idxs)])
        grad = torch.autograd.grad(loss, model.parameters())
        fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, model.parameters())))

        for k in range(1, self.args.local_ep_test):
            outputs = model(self.features[list(idxs)], fast_weights)
            #loss = F.nll_loss(outputs, self.labels[idx][list(idxs)])
            loss = F.cross_entropy(outputs, labels[list(idxs)])
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, fast_weights)))

        #model.load_state_dict(fast_weights)
        update_paras['vars.0'] = torch.tensor(fast_weights[0])

        return update_paras, loss



def test_inference_SGC(args, model, features, labels, idx_test):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    device = 'cuda' if args.gpu else 'cpu'

    output = model(features[idx_test])
    #loss = F.nll_loss(output, labels[idx_test])
    loss = F.cross_entropy(output, labels[idx_test])
    acc = accuracy(output, labels[idx_test])  / len(idx_test)

    return acc, loss



