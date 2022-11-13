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
    def __init__(self, args, features, adj, labels):

    # def __init__(self, args, features, adj, labels, idxs_users, x_spt_tr, y_spt_tr, 
    #                               x_qry_tr, y_qry_tr, x_qry_tr, y_qry_tr, logger):
        self.args = args
        #self.logger = logger

        self.features = features
        self.adj = adj
        self.labels = labels
        # self.x_spt_tr = x_spt_tr
        # self.y_spt_tr = y_spt_tr
        # self.x_qry_tr = x_qry_tr
        # self.y_qry_tr = y_qry_tr
        # self.idxs_users = idxs_users

        self.device = 'cuda' if args.gpu else 'cpu'
        


    def forward(self, model, x_spt_tr, y_spt_tr, x_qry_tr, y_qry_tr):
        
        #model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.meta_lr,
                                         weight_decay=self.args.weight_decay)


        losses_q = [0 for _ in range(self.args.local_ep + 1)]
        corrects = [0 for _ in range(self.args.local_ep + 1)]
    
        outputs = model(self.features, self.adj, vars=None)
        loss = F.nll_loss(outputs[x_spt_tr], y_spt_tr)
        grad = torch.autograd.grad(loss, model.parameters())
        fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, model.parameters())))

        #print('fast_weights:', fast_weights[0])

        with torch.no_grad():
            outputs_q = model(self.features, self.adj, model.parameters())
            loss_q = F.nll_loss(outputs_q[x_qry_tr], y_qry_tr)
            losses_q[0] += loss_q
            correct = accuracy(outputs_q[x_qry_tr], y_qry_tr).item()
            corrects[0] = corrects[0] + correct

        with torch.no_grad():
            outputs_q = model(self.features, self.adj, fast_weights)
            loss_q = F.nll_loss(outputs_q[x_qry_tr], y_qry_tr)
            losses_q[1] += loss_q
            correct = accuracy(outputs_q[x_qry_tr], y_qry_tr).item()
            corrects[1] = corrects[1] + correct


        for k in range(1, self.args.local_ep):

            outputs = model(self.features, self.adj, fast_weights)
            loss = F.nll_loss(outputs[x_spt_tr], y_spt_tr)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, fast_weights)))

            outputs = model(self.features, self.adj, fast_weights)
            loss_q = F.nll_loss(outputs[x_qry_tr], y_qry_tr)
            losses_q[k + 1] += loss_q

            #print('loss:', loss_q)
            with torch.no_grad():
                correct = accuracy(outputs[x_qry_tr], y_qry_tr).item()
                corrects[k + 1] = corrects[k + 1] + correct

        loss_q = losses_q[-1]
        optimizer.zero_grad()
        loss_q.backward()
        optimizer.step()

        return model.state_dict(), sum(losses_q) / len(losses_q)


    def finetuning(self, model, x_spt_tr, y_spt_tr, x_qry_tr, y_qry_tr):
        
        corrects = [0 for _ in range(self.args.local_ep_test + 1)]

        outputs = model(self.features, self.adj, vars=None)
        loss = F.nll_loss(outputs[list(x_spt_tr)], y_spt_tr)
        grad = torch.autograd.grad(loss, model.parameters())
        fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, model.parameters())))

        with torch.no_grad():
            outputs_q = model(self.features, self.adj, model.parameters())
            correct = accuracy(outputs_q[x_qry_tr], y_qry_tr).item()
            corrects[0] = corrects[0] + correct

        with torch.no_grad():
            outputs_q = model(self.features, self.adj, fast_weights)
            correct = accuracy(outputs_q[x_qry_tr], y_qry_tr).item()
            corrects[1] = corrects[1] + correct


        for k in range(1, self.args.local_ep_test):
            outputs = model(self.features, self.adj, fast_weights)
            loss = F.nll_loss(outputs[x_spt_tr], y_spt_tr)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, fast_weights)))

            outputs = model(self.features, self.adj, fast_weights)
            with torch.no_grad():
                correct = accuracy(outputs[x_qry_tr], y_qry_tr).item()
                corrects[k + 1] = corrects[k + 1] + correct

        accs = np.array(corrects) / len(y_qry_tr)

        return accs




class LocalUpdate_SGC(nn.Module):

    def __init__(self, args, features, labels):

    # def __init__(self, args, features, adj, labels, idxs_users, x_spt_tr, y_spt_tr, 
    #                               x_qry_tr, y_qry_tr, x_qry_tr, y_qry_tr, logger):
        self.args = args
        #self.logger = logger

        self.features = features
        #self.adj = adj
        self.labels = labels
        # self.x_spt_tr = x_spt_tr
        # self.y_spt_tr = y_spt_tr
        # self.x_qry_tr = x_qry_tr
        # self.y_qry_tr = y_qry_tr
        # self.idxs_users = idxs_users

        self.device = 'cuda' if args.gpu else 'cpu'
        

    def forward(self, model, x_spt_tr, y_spt_tr, x_qry_tr, y_qry_tr):
        
           
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.meta_lr,
                                         weight_decay=self.args.weight_decay)


        losses_q = [0 for _ in range(self.args.local_ep + 1)]
        corrects = [0 for _ in range(self.args.local_ep + 1)]


        outputs = model(self.features[x_spt_tr], vars=None)
        #loss = F.nll_loss(outputs, y_spt_tr)
        loss = F.cross_entropy(outputs, y_spt_tr)
        grad = torch.autograd.grad(loss, model.parameters())
        fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, model.parameters())))

        #print('fast_weights:', fast_weights[0])

        with torch.no_grad():
            outputs_q = model(self.features[x_qry_tr], model.parameters())
            #loss_q = F.nll_loss(outputs_q, y_qry_tr)
            loss_q = F.cross_entropy(outputs_q, y_qry_tr)
            losses_q[0] += loss_q
            correct = accuracy(outputs_q, y_qry_tr).item()
            corrects[0] = corrects[0] + correct

        with torch.no_grad():
            outputs_q = model(self.features[x_qry_tr], fast_weights)
            #loss_q = F.nll_loss(outputs_q, y_qry_tr)
            loss_q = F.cross_entropy(outputs_q, y_qry_tr)
            losses_q[1] += loss_q
            correct = accuracy(outputs_q, y_qry_tr).item()
            corrects[1] = corrects[1] + correct


        for k in range(1, self.args.local_ep):

            outputs = model(self.features[x_spt_tr], fast_weights)
            #loss = F.nll_loss(outputs, y_spt_tr)
            loss = F.cross_entropy(outputs, y_spt_tr)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, fast_weights)))

            outputs = model(self.features[x_qry_tr], fast_weights)
            #loss_q = F.nll_loss(outputs, y_qry_tr)
            loss_q = F.cross_entropy(outputs, y_qry_tr)
            losses_q[k + 1] += loss_q

            #print('loss:', loss_q)
            with torch.no_grad():
                correct = accuracy(outputs, y_qry_tr).item()
                corrects[k + 1] = corrects[k + 1] + correct


        loss_q = losses_q[-1]
        optimizer.zero_grad()
        loss_q.backward()
        optimizer.step()

        return model.state_dict(), sum(losses_q) / len(losses_q)



    def finetuning(self, model, x_spt_tr, y_spt_tr, x_qry_tr, y_qry_tr):
        
        corrects = [0 for _ in range(self.args.local_ep_test + 1)]

        outputs = model(self.features[x_spt_tr], vars=None)
        loss = F.cross_entropy(outputs, y_spt_tr)
        grad = torch.autograd.grad(loss, model.parameters())
        fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, model.parameters())))

        with torch.no_grad():
            outputs_q = model(self.features[x_qry_tr], model.parameters())
            correct = accuracy(outputs_q, y_qry_tr).item()
            corrects[0] = corrects[0] + correct

        with torch.no_grad():
            outputs_q = model(self.features[x_qry_tr], fast_weights)
            correct = accuracy(outputs_q, y_qry_tr).item()
            corrects[1] = corrects[1] + correct


        for k in range(1, self.args.local_ep_test):
            outputs = model(self.features[x_spt_tr], fast_weights)
            loss = F.cross_entropy(outputs, y_spt_tr)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.args.lr * p[0], zip(grad, fast_weights)))

            outputs = model(self.features[x_qry_tr], fast_weights)
            with torch.no_grad():
                correct = accuracy(outputs, y_qry_tr).item()
                corrects[k + 1] = corrects[k + 1] + correct

        accs = np.array(corrects) / len(y_qry_tr)

        return accs


