import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch import optim

from torch.nn import Module
from torch.nn.parameter import Parameter
import math

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # if bias:
        #     self.bias = Parameter(torch.FloatTensor(out_features))
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, adj, weight, bias):

        #support = torch.spmm(input, weight)
        support = torch.mm(input, weight)
        #support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if bias is not None:
        #if self.bias is not None:
            return output + bias
            #return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.vars = nn.ParameterList()
        
        w1 = Parameter(torch.FloatTensor(nfeat, nhid))
        stdv = 1. / math.sqrt(w1.size(1))
        w1.data.uniform_(-stdv, stdv)
        self.vars.append(w1)
        
        b1 = Parameter(torch.FloatTensor(nhid))
        b1.data.uniform_(-stdv, stdv)
        self.vars.append(b1)

        w2 = Parameter(torch.FloatTensor(nhid, nclass))
        stdv = 1. / math.sqrt(w2.size(1))
        w2.data.uniform_(-stdv, stdv)
        self.vars.append(w2)

        b2 = Parameter(torch.FloatTensor(nclass))
        b2.data.uniform_(-stdv, stdv)
        self.vars.append(b2)

    
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        
        print('init:', self.vars[0])


    def forward(self, x, adj, vars=None):

        if vars is None:
            vars = self.vars

        #print('vars0:', vars[0])

        x = F.relu(self.gc1(x, adj, vars[0], vars[1]))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, vars[2], vars[3])
        
        return F.log_softmax(x, dim=1)
        #return x

    def parameters(self):
        return self.vars



class SGC(nn.Module):
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        #self.W = nn.Linear(nfeat, nclass)
        self.vars = nn.ParameterList()
        w = nn.Parameter(torch.ones((nclass,nfeat)))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)


    def forward(self, x, vars=None):

        if vars is None:
            vars = self.vars

        w = vars[0]
        x = F.linear(x, w)

        return x
        #return F.log_softmax(x, dim=1)
        #return self.W(x)


    def parameters(self):
        return self.vars


