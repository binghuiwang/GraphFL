import numpy as np
import scipy.sparse as sp
import torch
import sys
import copy

import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from sklearn.metrics import f1_score
#from sampling import split_iid

from io2 import load_dataset
from preprocess import to_binary_bag_of_words, remove_underrepresented_classes, \
    eliminate_self_loops, binarize_labels
#from util import is_binary_bag_of_words
from scipy.sparse import linalg as slinalg


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def absorption_probability(W, alpha, column=None):
    
    n = W.shape[0]
    print('Calculate absorption probability...')
    W = W.copy().astype(np.float32)
    D = W.sum(1).flat
    L = sp.diags(D, dtype=np.float32) - W
    L += alpha * sp.eye(W.shape[0], dtype=L.dtype)
    L = sp.csc_matrix(L)
    # print(np.linalg.det(L))

    if column is not None:
        A = np.zeros(W.shape)
        # start = time.time()
        A[:, column] = slinalg.spsolve(L, sp.csc_matrix(np.eye(L.shape[0], dtype='float32')[:, column])).toarray()
        # print(time.time()-start)
    else:
        # start = time.time()
        A = slinalg.inv(L).toarray()
        # print(time.time()-start)
    return A


def cotraining(W, t, idx_train, labels):
    
    W = W.to_dense().numpy()
    labels_ext = np.zeros(labels.shape[0], dtype=np.int32)
    labels_ext[list(idx_train)] = labels[list(idx_train)]

    labels_bin = binarize_labels(labels)
    #print(labels.shape)
    train_mask = sample_mask(list(idx_train), labels_bin.shape[0])
    y_train = np.zeros(labels_bin.shape)
    y_train[train_mask, :] = labels_bin[train_mask, :]
    
    A = absorption_probability(W, 1e-4, train_mask)
    idx_train_ext = list(idx_train).copy() #np.where(train_mask)[0]
    
    already_labeled = np.sum(y_train, axis=1)
    if not hasattr(t, '__getitem__'):
        t = [t for _ in range(y_train.shape[1])]
    
    for i in range(y_train.shape[1]):
        y = y_train[:, i:i + 1]
        a = A.dot(y)
        a[already_labeled > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[t]
        index = np.where(a.flat > gate)[0]

        idx_train_ext = np.hstack([idx_train_ext, index])
        labels_ext[index] = i

    print('#train:', (idx_train))
    print('#train ext:', (idx_train_ext))
    print('label:', labels[list(idx_train)])
    print('label ext:', labels_ext[idx_train_ext])

    return idx_train_ext, labels_ext

    
def lp(adj, alpha, y_train, train_mask, y_test):
    
    P = absorption_probability(adj, alpha, column=train_mask)
    P = P[:, train_mask]

    # nearest clssifier
    predicted_labels = np.argmax(P, axis=1)
    # prediction = alpha*P
    
    prediction = np.zeros(P.shape)
    prediction[np.arange(P.shape[0]), predicted_labels] = 1

    y = np.sum(train_mask)
    label_per_sample = np.vstack([np.zeros(y), np.eye(y)])[np.add.accumulate(train_mask) * train_mask]
    sample2label = label_per_sample.T.dot(y_train)
    prediction = prediction.dot(sample2label)

    test_acc = np.sum(prediction * y_test) / np.sum(y_test)
    test_acc_of_class = np.sum(prediction * y_test, axis=0) / np.sum(y_test, axis=0)
    # print(test_acc, test_acc_of_class)
    return test_acc, test_acc_of_class, prediction



def selftraining(prediction, no_class, t, idx_train, labels):

    labels_ext = np.argmax(prediction, axis=1)
    confidence = np.max(prediction, axis=1)
    sorted_index = np.argsort(-confidence)

   
    index = []
    count = [0 for i in range(no_class)]
    for i in sorted_index:
        for j in range(no_class):
            if labels_ext[i] == j and count[j] < t and i not in idx_train:
                index.append(i)
                count[j] += 1

    print('index:', index)
    correct = 0
    for idx in index:
        if labels_ext[idx] == labels[idx]:
            correct += 1
    print('#cor:', correct, '#total:', len(index), 'acc:', correct/len(index))

    labels_ext[list(idx_train)] = labels[list(idx_train)]
    idx_train_ext = list(idx_train) + index
    #print('#train ext:', len(idx_train_ext))

    print('#train:', (idx_train))
    print('label:', labels[idx_train])
    print('#train ext:', (idx_train_ext))
    print('label ext:', labels_ext[idx_train_ext])

    return idx_train_ext, labels_ext



def read_data(file):

    with open(file) as datafile:
        data = datafile.readlines()
        x_spt = [int(item) for item in data[0].split(' ')]
        y_spt = [int(item) for item in data[1].split(' ')]
        x_qry = [int(item) for item in data[2].split(' ')]
        y_qry = [int(item) for item in data[3].split(' ')]
    
    return x_spt, y_spt, x_qry, y_qry



def split(idx_train, num_users):
    """
    Sample I.I.D. client data from train indexes
    :return: dict of image index
    """
    num_items = int(len(idx_train) / num_users)
    dict_users, all_idxs = {}, [i for i in (idx_train)]
    #dict_users, all_idxs = {}, [i for i in range(len(idx_train))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct
    #return correct / len(labels)


def f1(output, labels):
    output = output.max(1)[1]
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, output, average='micro')
    return micro


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="AugNormAdj"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_youtube(dataset_str, is_deepwalk, frac):
    
    # fh = open('data/'+dataset_str+'.edgelist','rb')
    # adj = nx.read_adjlist(fh)
    # adj = nx.adjacency_matrix(adj)
    # #print(adj)

    labelfh = open('data/label.txt')
    labelfile = labelfh.readlines()
    no_labels = len(labelfile)
    print(no_labels)

    # LabelDictList = defaultdict(list)    
    # for line in labelfile:
    #     line_parts = line.split(' ')
    #     LabelDictList[int(line_parts[1])-1].append(int(line_parts[0]))
    # #rint LabelDictList.keys()
    # count = [len(LabelDictList[key]) for key in LabelDictList.keys()]
    # print count
    
    # idxs_train = []
    # for key in LabelDictList.keys():
    # #for idx in range(len(LabelDictList)):
    #     id = random.sample(xrange(len(LabelDictList[key])), 1)[0]
    #     # print id
    #     idxs_train.append(LabelDictList[key][id])
    # print(idxs_train)
    # idxs_val = []
    # for key in LabelDictList.keys():
    # #for idx in range(len(LabelDictList)):
    #     id = random.sample(xrange(len(LabelDictList[key])), 1)[0]
    #     # print id
    #     idxs_val.append(LabelDictList[key][id])
    # print(idxs_val)

    # trainfile = open('data/train_'+str(frac)+'.txt','w')
    # valfile = open('data/val_'+str(frac)+'.txt','w')
    # testfile = open('data/test_'+str(frac)+'.txt','w')
    # for idx in range(no_labels):
    #     #print idx
    #     line_parts = labelfile[idx].split()
    #     if int(line_parts[0]) in idxs_train:
    #         trainfile.write(labelfile[idx])
    #     elif int(line_parts[0]) in idxs_val:
    #         valfile.write(labelfile[idx])
    #     else:
    #         testfile.write(labelfile[idx])
    # trainfile.close()
    # valfile.close()
    # testfile.close()

    no_sample = int(frac * no_labels)
    idxs_sample = random.sample(xrange(no_labels), no_sample)
    trainfile = open('data/train_'+str(frac)+'.txt','w')
    valfile = open('data/val_'+str(frac)+'.txt','w')
    testfile = open('data/test_'+str(frac)+'.txt','w')
    for idx in range(no_labels):
        if idx in idxs_sample[:no_sample/2]:
            trainfile.write(labelfile[idx])
        elif idx in idxs_sample[no_sample/2:]:
            valfile.write(labelfile[idx])
        else:
            testfile.write(labelfile[idx])

    labelfh = open('data/label.txt')
    labelfile = labelfh.readlines()
    labels = np.zeros((1138499,47))

    idx_all = list()
    label_all = list()
    for line in labelfile:
        line_parts = line.split(' ')
        idx_all.append(int(line_parts[0]))
        label_all.append(int(line_parts[1])-1)
    #print(label_train)
    label_all = np.array(label_all).reshape(-1)
    labels[idx_all,:] = np.eye(47)[label_all]

    trainfile = open('data/train_'+str(frac)+'.txt')
    idx_train = list()
    label_train = list()
    for line in trainfile:
        line_parts = line.split(' ')
        idx_train.append(int(line_parts[0]))
     
    valfile = open('data/val_'+str(frac)+'.txt')
    idx_val = list()
    for line in valfile:
        line_parts = line.split(' ')
        idx_val.append(int(line_parts[0]))
    
    testfile = open('data/test_'+str(frac)+'.txt')
    idx_test = list()
    for line in testfile:
        line_parts = line.split(' ')
        idx_test.append(int(line_parts[0]))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    #train_idx = list(np.where(train_mask)[0])
    #print(train_idx)
    print(len(label_all), len(idx_train),len(idx_val),len(idx_test))
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # count = 0
    # if is_deepwalk:
    #     features = []
    #     with open('data/LINE_'+dataset_str+'.embedding') as embedding_file:

    #       lines = embedding_file.readlines()
    #       #for line in lines[:-1]:
    #       for line in lines:
    #         line = line.split(' ')
    #         inner_list = [float(elt.rstrip('\t\r\n')) for elt in line[1:-1]]
    #         #print(inner_list)
    #         if int(line[0]) == 1138499:
    #             if count == 0:
    #                 features.append(inner_list)
    #                 count = 1
    #         if int(line[0]) != 1138499:
    #             features.append(inner_list)
    #       features = np.array(features)
    #       print(features.shape)

    #     #adj = None
    #     return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    # else:
    #     # adj = None
    #     return adj, y_train, y_val, y_test, train_mask, val_mask, test_mask
        

def load_citation(dataset_str="cora", normalization="AugNormAdj", train_size=140, num_users=50, cuda=False):
#def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=False):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    #print(graph)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    #features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0])) # For GCN

    #adj, features = preprocess_citation(adj, features, normalization) # For SGC

    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()


    idx_test = test_idx_range.tolist()
    
    #idx_train = range(len(y))
    #idx_val = range(len(y), len(y)+500)
    #idx_train = range(len(y), len(y)+500)
    #idx_train = range(500)

    idx_train = range(train_size)

    # if per_class != 1:
    #     idx_train = range(len(y))
    #     #idx_train = range(500)
    #     #idx_train = range(offset, offset+int(len(y)))
    #     #idx_val = range(offset+len(y), offset+len(y)+500)
    # else:
    #     label_train = y.argmax(axis=1)
    #     idx_train = []
    #     class_list = [[] for _ in range(y.shape[1])]
    #     for cls in range(y.shape[1]):
    #         class_list[cls] = np.where(label_train == cls)[0]
    #         idx_train.extend(class_list[cls][:per_class*num_users])
    #         #idx_train.extend(class_list[cls][:int(frac*len(class_list[cls]))])
    #         # print idx_train
    #     #idx_val = range(len(y), len(y)+500)

    idx_qry = range(500, len(y)+500)
    
    #print('#train:', len(idx_train), '#val:', len(idx_val), '#test:', len(idx_test))
    print('#train:', len(idx_train), '#qry:', len(idx_qry), '#test:', len(idx_test))

    # print('idx_test:', idx_test) #cora: 1708-2707, citeseer: 2312 - 3326

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_test = idx_test.cuda()
        idx_qry = idx_qry.cuda()

    user_spt_groups = split(idx_train, num_users)
    user_qry_groups = split(idx_qry, num_users)

    return adj, features, labels, idx_train, idx_qry, idx_test, user_spt_groups, user_qry_groups



def sgc_precompute(features, adj, degree):
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features


def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)




def is_binary_bag_of_words(features):
    features_coo = features.tocoo()
    return all(single_entry == 1.0 for _, _, single_entry in zip(features_coo.row, features_coo.col, features_coo.data))


def get_sub_dataset(name, data_path, standardize, num_subgraph=50, ratio_overlap=0.0):
    
    ## num_subgraph * len - (num_subgraph-1) * ratio_overlap * len = graph size
    ### => len = graph size // (num_subgraph - (num_subgraph-1) * ratio_overlap)

    dataset_graph = load_dataset(data_path)

    # some standardization preprocessing
    if standardize:
        dataset_graph = dataset_graph.standardize()
    else:
        dataset_graph = dataset_graph.to_undirected()
        dataset_graph = eliminate_self_loops(dataset_graph)


    adj, node_features, labels = dataset_graph.unpack()
    #print(node_features.shape)
    #print(node_features)

    #labels = binarize_labels(labels)
    labels = torch.LongTensor(labels)
    
    sub_size = int(np.ceil(adj.shape[0] / (num_subgraph - (num_subgraph-1) * ratio_overlap)))
    #sub_size = adj.shape[0] // (num_subgraph - (num_subgraph-1) * ratio_overlap)
    idxs_st_ed = []
    sub_adj = [ [] for _ in range(num_subgraph)]
    for i_sub in range(num_subgraph):
        nb_nd_st = int(np.round(i_sub * (1- ratio_overlap) * sub_size))
        nb_nd_ed = int(np.ceil(i_sub * (1- ratio_overlap) * sub_size) + sub_size)
        if nb_nd_ed > adj.shape[0]-1:
            nb_nd_ed = adj.shape[0]-1
        idxs_st_ed.append([nb_nd_st, nb_nd_ed])
    #print('graph idxs:', idxs_st_ed)


    adj = sp.coo_matrix(adj)
    row_nodes, col_nodes, adj_values = adj.row, adj.col, adj.data
    indices = np.vstack((adj.row, adj.col))
    #print(adj_values.shape)
    print(len(row_nodes), row_nodes)
    print(len(col_nodes), col_nodes)
    
    for idx in range(num_subgraph):
        sub_adj[idx] = sp.coo_matrix((np.zeros(adj_values.shape),indices), shape=adj.shape,
                        dtype=np.float32)

    for idx in range(num_subgraph):
        st_idx, ed_idx = idxs_st_ed[idx][0], idxs_st_ed[idx][1]
        print('start, end:', st_idx, ed_idx)
        idx_rows = [idx for idx, val in enumerate(row_nodes) if val <= ed_idx and val >= st_idx]
        idx_cols = [idx for idx, val in enumerate(col_nodes) if val <= ed_idx and val >= st_idx]
        idx_share = list(set(idx_rows).intersection(set(idx_cols)))
        print(len(idx_rows), len(idx_cols),len(idx_share))
        # print(row_nodes[idx_share])
        # print(col_nodes[idx_share])


        indices_sub = np.vstack((row_nodes[idx_share], col_nodes[idx_share]))
        sub_adj[idx] = sp.coo_matrix((np.ones(len(idx_share)),indices_sub), shape=adj.shape,
                        dtype=np.float32)
        print('#nnz:', sub_adj[idx].count_nonzero())
        sub_adj[idx] = sub_adj[idx] + sub_adj[idx].T.multiply(sub_adj[idx].T > sub_adj[idx]) \
                            - sub_adj[idx].multiply(sub_adj[idx].T > sub_adj[idx])
        sub_adj[idx] = row_normalize(sub_adj[idx] + sp.eye(sub_adj[idx].shape[0]))
        print('#nnz after:', sub_adj[idx].count_nonzero())
        sub_adj[idx] = sparse_mx_to_torch_sparse_tensor(sub_adj[idx])


    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = row_normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # convert to binary bag-of-words feature representation if necessary
    if not is_binary_bag_of_words(node_features):
        node_features = to_binary_bag_of_words(node_features)

    node_features = np.array(node_features.todense())
    print(node_features.shape)
    #print('#nnz fea:', np.count_nonzero(node_features))

    sub_fea = [ [] for _ in range(num_subgraph)]
    for idx in range(num_subgraph):
        st_idx, ed_idx = idxs_st_ed[idx][0], idxs_st_ed[idx][1]
        sub_fea[idx] = np.zeros(node_features.shape)
        sub_fea[idx][st_idx:ed_idx,:] = node_features[st_idx:ed_idx,:]
        #print('#nnz fea_sub:', np.count_nonzero(sub_fea[idx]))
        sub_fea[idx] = torch.FloatTensor(np.array(sub_fea[idx]))
        

    node_features = torch.FloatTensor(node_features)

    return sub_adj, sub_fea, adj, node_features, labels, idxs_st_ed


def get_dataset(name, data_path, standardize, train_examples_per_class=None, val_examples_per_class=None):
    
    dataset_graph = load_dataset(data_path)

    # some standardization preprocessing
    if standardize:
        dataset_graph = dataset_graph.standardize()
    else:
        dataset_graph = dataset_graph.to_undirected()
        dataset_graph = eliminate_self_loops(dataset_graph)

    if train_examples_per_class is not None and val_examples_per_class is not None:
        if name == 'cora_full':
            # cora_full has some classes that have very few instances. We have to remove these in order for
            # split generation not to fail
            dataset_graph = remove_underrepresented_classes(dataset_graph,
                                                            train_examples_per_class, val_examples_per_class)
            dataset_graph = dataset_graph.standardize()
            # To avoid future bugs: the above two lines should be repeated to a fixpoint, otherwise code below might
            # fail. However, for cora_full the fixpoint is reached after one iteration, so leave it like this for now.


    adj, node_features, labels = dataset_graph.unpack()
    # graph_adj, node_features, labels = dataset_graph.unpack()


    # #labels = binarize_labels(labels)
    labels = torch.LongTensor(labels)

    # graph_adj = sp.coo_matrix(graph_adj)
    # values = graph_adj.data
    # indices = np.vstack((graph_adj.row, graph_adj.col))
    
    # adj = sp.coo_matrix((values,indices), shape=graph_adj.shape,
    #                     dtype=np.float32)
    
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = row_normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # adj = torch.sparse_coo_tensor(indices = torch.LongTensor(indices), \
    #     values = torch.FloatTensor(values), size=graph_adj.shape)
   
    # convert to binary bag-of-words feature representation if necessary
    if not is_binary_bag_of_words(node_features):
        node_features = to_binary_bag_of_words(node_features)
    #print(node_features)

    # fea_values = node_features.data
    # fea_indices = np.vstack((node_features.row, node_features.col))
    # node_features = torch.sparse_coo_tensor(indices = torch.LongTensor(fea_indices), \
    #     values = torch.FloatTensor(fea_values), size=node_features.shape)
   
    #node_features = row_normalize(node_features) 
    node_features = torch.FloatTensor(np.array(node_features.todense()))
    #print(node_features)

    #print(node_features)
    #print(len(np.unique(labels)))
    
    # some assertions that need to hold for all datasets
    # adj matrix needs to be symmetric
    #assert (graph_adj != graph_adj.T).nnz == 0
    # features need to be binary bag-of-word vectors
    # assert is_binary_bag_of_words(node_features), f"Non-binary node_features entry!"

    return adj, node_features, labels


def get_train_val_test_split(random_state,
                             labels, num_users,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):

    num_samples, num_classes  = len(labels), len(np.unique(labels))
    print(num_samples, num_classes)
    #num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    user_spt_groups = split(train_indices, num_users)
    user_qry_groups = split(val_indices, num_users)

    print('train indices:', train_indices)

    return train_indices, val_indices, test_indices, user_spt_groups, user_qry_groups


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes  = len(labels), len(np.unique(labels))
    #num_samples, num_classes = labels.shape
    print(num_samples, num_classes)
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])



def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):

    print('\nExperimental configurations:')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Meta-Learning  : {args.meta_lr}')
    print(f'    Weight-decay  : {args.weight_decay}')
    
    print(f'    Global Rounds   : {args.epochs}')
    print(f'    Total users       : {args.num_users}')
    #print(f'    #Per_class  : {args.per_class}')
    print(f'    Fraction of users  : {args.frac}')
    #print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Local Epochs finetune      : {args.local_ep_test}')

    return


