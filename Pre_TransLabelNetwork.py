# Author:   Jin Xu
# Data:     2020-01-05
# Function: Transform the label network to make it suitable for algorithm training

#--------------------------     import package    --------------------------#

import multiprocessing
import random
import os
import time
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd
from queue import Queue

#--------------------------   global variable  --------------------------#

WORK_PATH = "C:\\Users\\HP\\Desktop\\Exp_TotalEmbeddedCombinationGCN\\data\\" 

#-------------------------- component function --------------------------#

def cos_similarity(vector1, vector2):
    '''
    Calculating cosine similarity in parallel

    Input = (i, j, vector1, vector2):   
        vector1, vector2:   (list/array) two vectors
    Output:
        cos:                (float) cosine similarity 
    '''

    cos = float(np.dot(vector1,vector2) / (np.linalg.norm(vector1)*np.linalg.norm(vector2)))
    
    return cos

def matrix2coo(m):
    """
    Change normal matrix to coo sparse matrix

    Input:
        m:      (np.array 2dim) input normal matrix
    Output:
        coo_m:  (sp.coo_matrix) output coo sparse matrix
    """
    rows, cols, values = [], [], []
    for i in range(0, m.shape[0]):
        for j in range(0, m.shape[1]):
            if m[i,j] != 0:
                rows.append(i)
                cols.append(j)
                values.append(m[i,j])
    coo_m = sp.coo_matrix((values, (rows, cols)), shape = m.shape, dtype = float)

    return coo_m

def random_feature(feature_parameter):
    """
    Generate a random feature

    Input:
        feature_parameter = [number_of_feature, fill_ratio, fill_error, fill_mode] (list)
            number_of_feature:  (int) number of feature that each sample have
            fill_ratio:         (float, 0-1) what percentage of the features in each sample's vector is 1
            fill_error:         (float, 0-1) the error range of fill ratio
            fill_mode:          (string) fill mode is "full1" or "01random"
    Output:
        feature:    (list) a feature vector
    """

    [number_of_feature, fill_ratio, fill_error, fill_mode] = feature_parameter
    feature, fill_label = np.zeros(number_of_feature, int), np.zeros(number_of_feature, int)
    maxfillnumber = int( min((fill_ratio + fill_error), 1) * number_of_feature )
    minfillnumber = int( min((fill_ratio - fill_error), 1) * number_of_feature )
    fillnumber = random.randint(minfillnumber, maxfillnumber)

    for i in range(0, fillnumber):
        fill_local = random.randint(0, number_of_feature - 1)
        while fill_label[fill_local] != 0:
            fill_local = random.randint(0, number_of_feature - 1)
        fill_label[fill_local] = 1

        if fill_mode == 'full1':
            feature[fill_local] = 1
        elif fill_mode == "01random":
            feature[fill_local] = round(random.random(), 10)
        else:
            raise Exception("[Fill Mode Error] Please input 01random or full1 as fill_mode input.")    
    
    return list(feature)


def FillFeatureByGraph(g):
    """
        Random fill feature based on cosine similarity

        Input:
            g: (networkx graph) graph
        Output:
            return matrix of feature
    """

    # Parameter set
    cos_limited = 0.5
    number_of_feature   = max(g.number_of_nodes(), 1000)
    fill_ratio          = 0.3
    fill_error          = 0.05
    fill_mode           = 'full1'
    feature_parameter = [number_of_feature, fill_ratio, fill_error, fill_mode]

    # Initialization
    number_of_node = g.number_of_nodes()
    edge_list = list(g.edges())
    node2feature = {}
    q = Queue(maxsize = 0)
    node_flag = np.zeros(number_of_node, int)
    queue_count, cos_reduce_time, display_interval = 0, 0, 500

    # Build feature by cos
    while (node_flag == 1).all() == False:
        temp = [i for i,x in enumerate(node_flag) if x == 0]
        start_node = random.choice(temp)
        q.put(start_node)

        while q.empty() == False:
            temp_time = time.time()
            queue_count += 1
            if queue_count % display_interval == 0:
                print("    Now queue counting [{:5d}], queue length [{:5d}].".format(queue_count, len(list(q.queue))))
            
            node = q.get()   
            node_flag[node] = 1
            neighbor = list(g.neighbors(node))

            local_cos_limited, local_cos_reduce = cos_limited, 0
            check_flag = False
            while check_flag == False:
                if time.time() - temp_time > 15:
                    local_cos_limited -= 0.01
                    print("    Difficulty in processing node [{:5d}], reducing the cosine similarity threshold to [{:.2f}] in this process.".format(node, local_cos_limited))
                    temp_time = time.time()
                    local_cos_reduce += 1
                node2feature[node] = random_feature(feature_parameter)
                check_flag = True
                for item in neighbor:
                    if item in node2feature:
                        if cos_similarity(node2feature[node], node2feature[item]) < local_cos_limited:
                            check_flag = False
                            break
        
            for item in neighbor:
                if node_flag[item] == 0 and item not in list(q.queue):
                    q.put(item)

            if local_cos_reduce > cos_reduce_time:
                cos_reduce_time = local_cos_reduce
    
    # Check
    check_flag, check_list = True, []
    for edge in edge_list:
        temp = cos_similarity(node2feature[edge[0]], node2feature[edge[1]])
        check_list.append(temp)
        if  temp < cos_limited - 0.01 * cos_reduce_time:
            check_flag = False
    if check_flag == False:
        print(">> Check may find somr wrong! Min cos is [{:.4f}]".format(min(check_list)))
    else:
        print(">> Check finish! Random network meets requirements. Min cos is [{:.4f}]".format(min(check_list)))

    # Change to matrix
    temp = []
    for i in range(0, number_of_node):
        temp.append(node2feature[i])

    return np.array(temp)


#--------------------------   main function    --------------------------#

def changelabelnetwork(path, name, train_num, test_num, fill_feature = "onehot"):
    """
    Change label network to GNN test network

    Input:
        path:           (string) label network data path
        name:           (string) label network name
        train_num:      (int) the number of train sample
        test_num:       (int) the number of test samplefill_feature
        fill_feature：  (string) fill mode, one-hot, all-1 or cos
    Output:
        None
    """
    # Build node index to node type dictionary
    nodeindex2nodetype, nodeid2nodeindex = {}, {}
    label_df = pd.read_csv(path + name + "_Label.csv")
    label_dl = label_df.values.tolist()
    random.shuffle(label_dl)

    node_list, type_list = [], []
    for i in range(0, len(label_dl)):
        node_list.append(label_dl[i][0])
        type_list.append(label_dl[i][1])
    
    index = 0
    for node in node_list:
        nodeid2nodeindex[node] = index
        index = index + 1
    for pair in label_dl:
        nodeindex2nodetype[nodeid2nodeindex[pair[0]]] = pair[1]

    node_num = len(node_list)
    type_num = max(type_list)

    # Build graph data
    print(">> Building graph...")
    graph_df = pd.read_csv(path + name + "_Data.csv")
    graph_dl = graph_df.values.tolist()
    DG = nx.DiGraph() 
    DG.add_nodes_from(list( range(0, node_num) ))
    edgelist = []
    for pair in graph_dl:
        edgelist.append((nodeid2nodeindex[pair[0]], nodeid2nodeindex[pair[1]]))
    DG.add_edges_from(edgelist)  
    g = DG.to_undirected()
    edge_num = g.number_of_edges()

    # Build feature and label matrix
    print(">> Building feature...")
    if fill_feature == "onehot":
        feature = np.identity(node_num)
    elif fill_feature == "all_1":
        feature = np.ones((node_num, node_num), int)
    elif fill_feature == "cos":
        feature = FillFeatureByGraph(g)
    else:
        raise Exception("[ERROR] wrong fill_feature")

    print(">> Building label...")
    label = np.zeros((node_num, type_num), int)
    for i in range(0, node_num):
        label[i][nodeindex2nodetype[i] - 1] = 1

    # Slice x,tx,allx,y,ty,ally
    print(">> Slicing...")
    x = feature[0: train_num]
    allx = feature[0: node_num - test_num]
    tx = feature[node_num - test_num: node_num]
    y = label[0: train_num]
    ally = label[0: node_num - test_num]
    ty = label[node_num - test_num: node_num]

    # Change graph to graph dictionary
    graph_dict = {}
    for nodeindex in range(0, node_num):
        graph_dict[nodeindex] = list(g.neighbors(nodeindex))
    
    print(">> Saving...")
    # Save as file by pickle
    with open(WORK_PATH + "ind.{}.allx".format(network_name), "wb") as f:
        pkl.dump(matrix2coo(allx), f)
    with open(WORK_PATH + "ind.{}.ally".format(network_name), "wb") as f:
        pkl.dump(ally, f)
    with open(WORK_PATH + "ind.{}.tx".format(network_name), "wb") as f:
        pkl.dump(matrix2coo(tx), f)
    with open(WORK_PATH + "ind.{}.ty".format(network_name), "wb") as f:
        pkl.dump(ty, f)
    with open(WORK_PATH + "ind.{}.x".format(network_name), "wb") as f:
        pkl.dump(matrix2coo(x), f)
    with open(WORK_PATH + "ind.{}.y".format(network_name), "wb") as f:
        pkl.dump(y, f)
    with open(WORK_PATH + "ind.{}.graph".format(network_name), "wb") as f:
        pkl.dump(graph_dict, f)
    with open(WORK_PATH + "ind.{}.test.index".format(network_name), "w") as f:
        for i in range(node_num - test_num, node_num):
            f.write(str(i) + "\n")

    # Feedback print
    print(">>> Finish! Change {} Label Network to Training.".format(network_name))
    print("    This network have [{}] nodes and [{}] edges.".format(g.number_of_nodes(), g.number_of_edges()))
    print("    Save in \"" + WORK_PATH)

    return


def buildfeaturenetwork(path, name, train_num, test_num):
    """
    Build feature network to GNN test network

    Input:
        path:           (string) label network data path
        name:           (string) label network name
        train_num:      (int) the number of train sample
        test_num:       (int) the number of test samplefill_feature
        fill_feature：  (string) fill mode, one-hot, all-1 or cos
    Output:
        None
    """
    # Build node index to node type dictionary
    nodeindex2nodetype, nodeid2nodeindex = {}, {}
    label_df = pd.read_csv(path + name + "_Label.csv")
    label_dl = label_df.values.tolist()
    random.shuffle(label_dl)
    node_list, type_list = [], []
    for i in range(0, len(label_dl)):
        node_list.append(label_dl[i][0])
        type_list.append(label_dl[i][1])
    
    index = 0
    for node in node_list:
        nodeid2nodeindex[node] = index
        index = index + 1
    for pair in label_dl:
        nodeindex2nodetype[nodeid2nodeindex[pair[0]]] = pair[1]

    node_num = len(node_list)
    type_num = max(type_list)

    # Build graph data
    print(">> Building graph...")
    graph_df = pd.read_csv(path + name + "_Data.csv")
    graph_dl = graph_df.values.tolist()
    DG = nx.DiGraph() 
    DG.add_nodes_from(list( range(0, node_num) ))
    edgelist = []
    for pair in graph_dl:
        edgelist.append((nodeid2nodeindex[pair[0]], nodeid2nodeindex[pair[1]]))
    DG.add_edges_from(edgelist)  
    g = DG.to_undirected()
    edge_num = g.number_of_edges()

    # Build feature matrix
    print(">> Building feature...")
    feature_df = pd.read_csv(path + name + "_Feature.csv")
    feature_dl = feature_df.values.tolist()
    nodeindex2nodefeature = {}
    for nodeinfo in feature_dl:
        nodeindex2nodefeature[nodeid2nodeindex[nodeinfo[0]]] = nodeinfo[1:]
    feature_num = len(nodeindex2nodefeature[0])
    temp = []
    for node in range(0, node_num):
        temp.append(nodeindex2nodefeature[node])
    feature = np.array(temp)    
    
    # Build label matrix
    print(">> Building label...")
    label = np.zeros((node_num, type_num), int)
    for i in range(0, node_num):
        label[i][nodeindex2nodetype[i] - 1] = 1
    
    # Slice x,tx,allx,y,ty,ally
    print(">> Slicing...")
    x = feature[0: train_num]
    allx = feature[0: node_num - test_num]
    tx = feature[node_num - test_num: node_num]
    y = label[0: train_num]
    ally = label[0: node_num - test_num]
    ty = label[node_num - test_num: node_num]

    # Change graph to graph dictionary
    graph_dict = {}
    for nodeindex in range(0, node_num):
        graph_dict[nodeindex] = list(g.neighbors(nodeindex))
    
    print(">> Saving...")
    # Save as file by pickle
    with open(WORK_PATH + "ind.{}.allx".format(network_name), "wb") as f:
        pkl.dump(matrix2coo(allx), f)
    with open(WORK_PATH + "ind.{}.ally".format(network_name), "wb") as f:
        pkl.dump(ally, f)
    with open(WORK_PATH + "ind.{}.tx".format(network_name), "wb") as f:
        pkl.dump(matrix2coo(tx), f)
    with open(WORK_PATH + "ind.{}.ty".format(network_name), "wb") as f:
        pkl.dump(ty, f)
    with open(WORK_PATH + "ind.{}.x".format(network_name), "wb") as f:
        pkl.dump(matrix2coo(x), f)
    with open(WORK_PATH + "ind.{}.y".format(network_name), "wb") as f:
        pkl.dump(y, f)
    with open(WORK_PATH + "ind.{}.graph".format(network_name), "wb") as f:
        pkl.dump(graph_dict, f)
    with open(WORK_PATH + "ind.{}.test.index".format(network_name), "w") as f:
        for i in range(node_num - test_num, node_num):
            f.write(str(i) + "\n")

    # Feedback print
    print(">>> Finish! Change {} Label Network to Training.".format(network_name))
    print("    This network have [{}] nodes and [{}] edges.".format(g.number_of_nodes(), g.number_of_edges()))
    print("    Save in \"" + WORK_PATH)

    return


##########################################################################

if __name__ == "__main__":
    # path = "E:\\Data\\MC_HeNetwork\\Labeled_Network\\LN_gene\\" # Lab
    
    path = "origin_data\\LN_fb_CMU_Carnegie49\\" # Envy 13
    network_name = "fb_CMU_Carnegie49"

    changelabelnetwork(path, network_name, train_num = 300, test_num = 2000, fill_feature = "onehot")
    '''
    path = "origin_data\\TerrorAttack\\" # Envy 13
    network_name = "TerrorAttack"
    buildfeaturenetwork(path, network_name, train_num = 120, test_num = 500)
    '''
