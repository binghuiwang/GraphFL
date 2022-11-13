import random
import argparse
from utils_gnn import *
from itertools import combinations
import numpy as np



def data_generator(labels, dataset, node_num, select_array, task_num, n_way, k_spt, k_qry, flag):

    x_spt = []
    y_spt = []
    x_qry = []
    y_qry = []
    class_idx = [[] for _ in range(n_way)]
    class_spt = [[] for _ in range(n_way)]
    class_qry = [[] for _ in range(n_way)]


    labels_local = labels.clone().detach()
    select_class = random.sample(select_array, n_way)
    print('select_class:', select_class)

    for j in range(node_num):
        for c in range(n_way):
            if labels_local[j] == select_class[c]:
                class_idx[c].append(j)
                #labels_local[j] = c
            

    for c in range(n_way):
        class_spt[c] = random.sample(class_idx[c], k_spt)
        class_qry[c] = [n1 for n1 in class_idx[c] if n1 not in class_spt[c]]


    train_idx = []
    test_idx = []
    for c in range(n_way):
        train_idx += class_spt[c]
        random.shuffle(train_idx)

        test_idx += random.sample(class_qry[c], k_qry)
        random.shuffle(test_idx)

        #print('train_idx:', train_idx)


    x_spt.append(train_idx)
    y_spt.append(labels_local[train_idx])
    x_qry.append(test_idx)
    y_qry.append(labels_local[test_idx]) 


    #print('x_spt:', x_spt)

    # train set
    if flag == 0: 
        
        with open('traintest/'+str(dataset)+'_train_task_'+str(task_num+1)+'_shot_'+str(k_spt)+'_way_'+str(n_way) + '_query_train_' + str(k_qry), "w") as traintestfile:

            traintestfile.write(" ".join(str(item) for item in x_spt[0]))
            traintestfile.write("\n")
            traintestfile.write(" ".join(str(item) for item in y_spt[0].numpy()))
            traintestfile.write("\n")
            traintestfile.write(" ".join(str(item) for item in x_qry[0]))
            traintestfile.write("\n")
            traintestfile.write(" ".join(str(item) for item in y_qry[0].numpy()))
            #traintestfile.write("\n")

    #test set
    if flag == 1:

        with open('traintest/'+str(dataset)+'_test_task_'+str(task_num+1)+'_shot_'+str(k_spt)+'_way_'+str(n_way) + '_query_test_' + str(k_qry), "w") as traintestfile:
        
            traintestfile.write(" ".join(str(item) for item in x_spt[0]))
            traintestfile.write("\n")
            traintestfile.write(" ".join(str(item) for item in y_spt[0].numpy()))
            traintestfile.write("\n")
            traintestfile.write(" ".join(str(item) for item in x_qry[0]))
            traintestfile.write("\n")
            traintestfile.write(" ".join(str(item) for item in y_qry[0].numpy()))



def read_data(file):

    with open(file) as datafile:
        data = datafile.readlines()
        x_spt = [int(item) for item in data[0].split(' ')]
        y_spt = [int(item) for item in data[1].split(' ')]
        x_qry = [int(item) for item in data[2].split(' ')]
        y_qry = [int(item) for item in data[3].split(' ')]
    
    return x_spt, y_spt, x_qry, y_qry


if __name__ == '__main__':


    argparser = argparse.ArgumentParser()

    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=100)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry_train', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--k_qry_test', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--dataset', type=str, default='cora', help='Dataset to use.')
    argparser.add_argument('--normalization', type=str, default='AugNormAdj', help='Normalization method for the adjacency matrix.')
    argparser.add_argument('--seed', type=int, default=42, help='Random seed.')
    
    args = argparser.parse_args()


    if args.dataset == 'citeseer':

        node_num = 3327
        class_label = [0, 1, 2, 3, 4, 5]
        combination = list(combinations(class_label, 2))
        adj, features, labels = load_citation(args.dataset, args.normalization)


    if args.dataset == 'cora':
        node_num = 2708
        class_label = [0, 1, 2, 3, 4, 5, 6]
        combination = list(combinations(class_label, 2))
        adj, features, labels = load_citation(args.dataset, args.normalization)


    if args.dataset == 'ms_academic_cs':
        adj, features, labels = \
            get_dataset(args.dataset, data_path='./data/'+args.dataset+'.npz', standardize=True, train_examples_per_class=None, val_examples_per_class=None)
        node_num = 18333
        class_label = np.unique(labels)
        print('class:', class_label)
        combination = list(combinations(class_label, 5))
        #combination = list(combinations(class_label, args.n_way))

        #print(combination)


    ### train : test = 10, 5
    #### n_way = 2, 3, 5, 
    for i in range(len(combination[:1])):

        test_label = list(combination[i])
        train_label = [n for n in class_label if n not in test_label]
        print('train_label:', train_label)
        print('test_label:', test_label)

        for t in range(args.task_num):
            data_generator(labels, args.dataset, node_num, train_label, t, args.n_way, args.k_spt, args.k_qry_train, 0)
            
            data_generator(labels, args.dataset, node_num, test_label, t, args.n_way, args.k_spt, args.k_qry_test, 1)
         

    x_spt, y_spt, x_qry, y_qry = read_data('traintest/'+str(args.dataset)+'_train_task_'+str(1)+'_shot_'+str(args.k_spt)+'_way_'+str(args.n_way) + '_query_train_' + str(args.k_qry_train))  
    print(x_spt)
    print(y_spt)        
