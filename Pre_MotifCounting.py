# Author:   Jin Xu
# Data:     2020-01-05
# Function: Counting the Network Motif

#--------------------------     import package    --------------------------#

import os
import pickle as pkl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp

#--------------------------   global variable  --------------------------#

PATH = "C:\\Users\\HP\\Desktop\\Exp_TotalEmbeddedCombinationGCN\\data\\"

#-------------------------- component function --------------------------#

def caculate_VFlag(Na, Nb, inse, node_num):
    """
    Caculate VFlag tag vector using to 4-order motif counting.
    
    Input:
        Na:         (list) Node a neighbor 
        Nb:         (list) Node b neighbor
        inse:       (list) The intersection of Na and Nb
        node_num:   (int) Graph node number
    Output:
        VFlag:      (array) VFLAG vector
    """
    
    VFlag = np.zeros(node_num, dtype = int)
    for node in Na:
        VFlag[node] = 1
    for node in Nb:
        VFlag[node] = 2
    for node in inse:
        VFlag[node] = 3

    return VFlag


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


def addNodeMI(nm_dict, node, value):
    """
    Add motif information to node dictionary.
    
    Input:
        nm_dict:    (dictionary) the dictionary of node motif degree
        node:       (int) the id of node
        value:      (int) the change value of node
    Output:
        nm_dict:   (dictionary) changed node motif degree dictionary
    """

    if node not in nm_dict.keys():
        nm_dict[node] = value
    else:
        nm_dict[node] += value
    
    return nm_dict


def addEdgeMI(em_dict, edge, value):
    """
    Add motif information to edge dictionary.
    
    Input:
        em_dict:    (dictionary) the dictionary of edge motif degree
        edge:       (int) the id of edge
        value:      (int) the change value of edge
    Output:
        em_dict:   (dictionary) changed edge motif degree dictionary
    """
    if edge not in em_dict.keys():
        em_dict[edge] = value + 1
    else:
        em_dict[edge] += value
    
    return em_dict


def caculate_MotifAandD(nm_dict, em_dict, node_num, Sparse):
    """
    Change node_motif_dictionary and edge_motif_dictionary to motif_Adjacency matrix and motif_Degree matrix.

    Input:
        nm_dict:    (dictionary) the dictionary of node motif degree
        em_dict:    (dictionary) the dictionary of edge motif degree
        node_num:   (int) the number of the node
        Sparse:     (bool) need sparse matrix or not
    Output:
        motif_A:    (matrix/coo_matrix) motif adjancey matrix
        motif_D:    (matrix/coo_matrix) motif degree matrix
    """
    if Sparse == False:
        motif_D = np.zeros((node_num, node_num), dtype=float)
        for node in nm_dict.keys():
            motif_D[node, node] = nm_dict[node]
        motif_A = np.zeros((node_num, node_num), dtype=float)
        for edge in em_dict.keys():
            motif_A[edge[0], edge[1]] = em_dict[edge]
            motif_A[edge[1], edge[0]] = em_dict[edge]
    else:
        motif_D_rows = np.array(list(nm_dict.keys()))
        motif_D_datas = np.array(list(nm_dict.values()))
        motif_D = sp.coo_matrix((motif_D_datas, (motif_D_rows, motif_D_rows)), shape = (node_num, node_num), dtype = float)
        motif_A_rows, motif_A_cols, motif_A_datas = [], [], []
        edge_list = em_dict.keys()
        for edge in edge_list:
            motif_A_rows.extend([edge[0], edge[1]])
            motif_A_cols.extend([edge[1], edge[0]])
            motif_A_datas.extend([em_dict[edge], em_dict[edge]])
        motif_A = sp.coo_matrix((motif_A_datas, (motif_A_rows, motif_A_cols)), shape = (node_num, node_num), dtype = float)
    
    return motif_A, motif_D

#--------------------------   main function    --------------------------#

def motiffeature(g, Sparse = False):
    """
    Calculate how many different kinds of motifs each node is in

    Input:
        g:              (networkx graph) input graph data
        Sparse:         (bool) Output matrix Sparse or not
    Output:
        motif_feature:  (matrix / coo_matix) motif feature matrix
    """

    # Initialize the motif feature dictionary
    node_num, node_list = g.number_of_nodes(), g.nodes()
    nm_dict = {}
    for node in node_list:
        nm_dict[node] = np.zeros(5, float)
    degree = dict(nx.degree(g))

    for node_a in node_list:
        Na = list(g.neighbors(node_a))
        for node_b in Na:
            if node_b < node_a:
                continue
            Nb = list(g.neighbors(node_b))
            inse = list(set(Na).intersection(set(Nb)))

            # M31 (Three-order triangle motif) Counting
            for node_c in inse:
                nm_dict[node_a][0] += 1/3
                nm_dict[node_b][0] += 1/3
                nm_dict[node_c][0] += 1/3

            # Get VFlag and M32 (Three-order path motif) Counting
            VFlag = caculate_VFlag(Na, Nb, inse, node_num)
            VFlag[node_a] = 0
            VFlag[node_b] = 0
            for i in range(0, len(VFlag)):
                if VFlag[i] == 1 or VFlag[i] == 2:
                    # M32
                    nm_dict[node_a][1] += 1/2
                    nm_dict[node_b][1] += 1/2
                    nm_dict[i][1]      += 1/2

            # M41 (Four-order fully connected motif) & M42 (Four-order stringed ring motif) Counting
            for node_c in inse:
                Nc = list(g.neighbors(node_c))
                for node_d in Nc:
                    if VFlag[node_d] == 3:
                        # M41
                        nm_dict[node_a][2] += 1/12
                        nm_dict[node_b][2] += 1/12
                        nm_dict[node_c][2] += 1/12
                        nm_dict[node_d][2] += 1/12
                    elif VFlag[node_d] == 2 or VFlag[node_d] == 1:
                        # M42
                        nm_dict[node_a][3] += 1/4
                        nm_dict[node_b][3] += 1/4
                        nm_dict[node_c][3] += 1/4
                        nm_dict[node_d][3] += 1/4

            # M43 (Four-order Square Motif) Counting 
            for node_c in Na:
                if VFlag[node_c] != 1 or node_c == node_b:
                    continue
                Nc = list(g.neighbors(node_c))
                for node_d in Nc:
                    if VFlag[node_d] == 2 and node_d != node_a:
                        nm_dict[node_a][4] += 1/4
                        nm_dict[node_b][4] += 1/4
                        nm_dict[node_c][4] += 1/4
                        nm_dict[node_d][4] += 1/4

    # Change dictionary to Matrix
    motif_feature = []
    for node in node_list:
        temp = [degree[node]]
        temp.extend(list(nm_dict[node]))
        motif_feature.append(temp)

    if Sparse == True:
        motif_feature = matrix2coo(np.matrix(motif_feature))
    else:
        motif_feature = np.matrix(motif_feature)

    return motif_feature


def motiffeatureSlice(motif_feature, dataset_str, Sparse = False):
    """
    Slice the motif feature to train, val and test

    Input:
        motif_feature:  (matrix / coo_matrix) motif feature matrix
        dataset_str:    (string) dataset name
        Sparse:         (bool) Matrix Sparse or not
    Output:
        None
    """

    if Sparse == True:
        motif_feature = motif_feature.todense()

    # Input Original data to check size
    with open(PATH + "ind.{}.x".format(dataset_str), 'rb') as f:
        x = pkl.load(f, encoding='latin1')
        x = x.todense()
    with open(PATH + "ind.{}.allx".format(dataset_str), 'rb') as f:
        allx = pkl.load(f, encoding='latin1')
        allx = allx.todense()
    with open(PATH + "ind.{}.tx".format(dataset_str), 'rb') as f:
        tx = pkl.load(f, encoding='latin1')
        tx = tx.todense()
    number_of_x, number_of_allx, number_of_tx = x.shape[0], allx.shape[0], tx.shape[0]
    
    # Get test node index
    index = []
    for line in open(PATH + "ind.{}.test.index".format(dataset_str)):
        index.append(int(line.strip()))
    
    # Slice the motif feature
    motif_train = motif_feature[0: number_of_x]
    motif_all = motif_feature[0: number_of_allx]
    motif_test = []
    for i in index:
        motif_test.append(motif_feature[i].tolist()[0])
    motif_test = np.matrix(motif_test)

    # Save in file
    print(">> Saving...")
    with open(PATH + "ind.{}.motif.allx".format(dataset_str), "wb") as f:
        pkl.dump(matrix2coo(motif_all), f)
    with open(PATH + "ind.{}.motif.tx".format(dataset_str), "wb") as f:
        pkl.dump(matrix2coo(motif_test), f)
    with open(PATH + "ind.{}.motif.x".format(dataset_str), "wb") as f:
        pkl.dump(matrix2coo(motif_train), f)

    return 


def totalMotifCounting(dataset_str, vis = False):
    """
    Calculate the number of five kinds three-order and four-order motifs directly

    Input:
        dataset_str:    (string) the name of dataset
        vis:            (bool) whether need to display the distribution of motifs
    Output:
        None
    """

    with open(PATH + "ind.{}.graph".format(dataset_str), 'rb') as f:
        graph = pkl.load(f, encoding='latin1')
    g = nx.from_dict_of_lists(graph)

    print(">> Motif counting...")
    motif_feature = motiffeature(g, False)

    if vis == True:
        line = []
        node_list = sorted(list(g.nodes()))
        for i in range(0, 6):
            temp = []
            for node in node_list:
                temp.append(motif_feature[node, i])
            line.append(temp)

        plt.figure("degree and motif show")
        colors = ['red', 'darkorange', 'gold', 'darkgreen', 'deepskyblue', 'darkblue', 'purple', 'black', 'pink']
        labels = ['degree', 'M31', 'M32', 'M41', 'M42', 'M43']
        for i in range(0, 6):
            plt.subplot(2, 3, i+1)
            plt.plot(node_list, line[i], color = colors[i], label = labels[i])
        
    print(">> Motif feature slice...")
    motiffeatureSlice(motif_feature, dataset_str)

    return


def MtriangleConting(graph_dict, mia = 1.0, Sparse = False):
    """ 
    Caculate 3-order triangle motif number.
    
    Input:
        graph_dict: (dictionary) the neighbor dictionary of each node in graph
        mia:        (float) motif information retention ratio (hyperparametric)
        Sparse:     (bool) need sparse matrix or not
    Output:
        motif_A:    (matrix/coo_matrix) motif adjancey matrix
        motif_D:    (matrix/coo_matrix) motif degree matrix
    """
    nm_dict, em_dict = {}, {}
    node_list = graph_dict.keys()   # List of nodes
    node_num = len(node_list)       # Number of nodes
    for node_a in node_list:        
        Na = graph_dict[node_a]
        node_a_motifsum = 0
        for node_b in Na:
            Nb = graph_dict[node_b]
            inse = list(set(Na).intersection(set(Nb)))
            node_a_motifsum += len(inse)
            if node_a < node_b:
                em_dict = addEdgeMI(em_dict, (node_a,node_b), mia * len(inse) / 6)
            else:
                em_dict = addEdgeMI(em_dict, (node_b,node_a), mia * len(inse) / 6)
        if node_a_motifsum != 0:
            nm_dict = addNodeMI(nm_dict, node_a, mia * node_a_motifsum / 6)

    return caculate_MotifAandD(nm_dict, em_dict, node_num, Sparse)


def MfullconectionCounting(graph_dict, mia = 1.0, Sparse = False):
    """ 
    Caculate 4-order full-conection motif number.

    Input:
        graph_dict: (dictionary) the neighbor dictionary of each node in graph
        mia:        (float) motif information retention ratio (hyperparametric)
        Sparse:     (bool) need sparse matrix or not
    Output:
        motif_A:    (matrix/coo_matrix) motif adjancey matrix
        motif_D:    (matrix/coo_matrix) motif degree matrix
    """
    nm_dict, em_dict = {}, {}
    node_list = graph_dict.keys()   # List of nodes
    node_num = len(node_list)       # Number of nodes
    for node_a in node_list:        
        Na = graph_dict[node_a]
        node_a_motifsum = 0
        for node_b in Na:
            Nb = graph_dict[node_b]
            inse = list(set(Na).intersection(set(Nb)))
            VFlag = caculate_VFlag(Na, Nb, inse, node_num)
            for node_c in inse:
                Nc = graph_dict[node_c]
                for node_d in Nc:
                    if node_c > node_d:
                        continue
                    if VFlag[node_d] == 3:
                        node_a_motifsum += 1
                        if node_a < node_b:
                            em_dict = addEdgeMI(em_dict, (node_a,node_b), mia / 12)
                        else:
                            em_dict = addEdgeMI(em_dict, (node_b,node_a), mia / 12)
        if node_a_motifsum != 0:
            nm_dict = addNodeMI(nm_dict, node_a, mia * node_a_motifsum / 12)

    return caculate_MotifAandD(nm_dict, em_dict, node_num, Sparse)


def MSquareCounting(graph_dict, mia = 1.0, Sparse = False):
    """ 
    Caculate 4-order square motif number.

    Input:
        graph_dict: (dictionary) the neighbor dictionary of each node in graph
        mia:        (float) motif information retention ratio (hyperparametric)
        Sparse:     (bool) need sparse matrix or not
    Output:
        motif_A:    (matrix/coo_matrix) motif adjancey matrix
        motif_D:    (matrix/coo_matrix) motif degree matrix
    """
    nm_dict, em_dict = {}, {}
    node_list = graph_dict.keys()   # List of nodes
    node_num = len(node_list)       # Number of nodes
    for node_a in node_list:        
        Na = graph_dict[node_a]
        node_a_motifsum = 0
        for node_b in Na:
            Nb = graph_dict[node_b]
            inse = list(set(Na).intersection(set(Nb)))
            VFlag = caculate_VFlag(Na, Nb, inse, node_num)
            for node_c in Na:
                if VFlag[node_c] != 1 or node_c == node_b:
                    continue
                Nc = graph_dict[node_c]
                for node_d in Nc:
                    if VFlag[node_d] == 2 and node_d != node_a:
                        node_a_motifsum += 1
                        if node_a < node_b:
                            em_dict = addEdgeMI(em_dict, (node_a,node_b), mia / 8)
                        else:
                            em_dict = addEdgeMI(em_dict, (node_b,node_a), mia / 8)
        if node_a_motifsum != 0:
            nm_dict = addNodeMI(nm_dict, node_a, mia * node_a_motifsum / 8)
    return caculate_MotifAandD(nm_dict, em_dict, node_num, Sparse)

##########################################################################

if __name__ == "__main__":
    # Test code
    mode = 2
    if mode == 1:
        g = nx.Graph()
        # g.add_edges_from([(0,1), (1,2), (2,3), (3,0), (3,1)])
        # g.add_edges_from([(0,1), (0,2)])
        g = nx.dense_gnm_random_graph(10, 20)
        print(motiffeature(g, False))
    else:
        
        dataset_str = "fb_CMU_Carnegie49"
        
        totalMotifCounting(dataset_str)
        '''
        with open(PATH + "ind.{}.allx".format(dataset_str), 'rb') as f:
            allx = pkl.load(f, encoding='latin1')
        with open(PATH + "ind.{}.ally".format(dataset_str), 'rb') as f:
            ally = pkl.load(f, encoding='latin1')
        with open(PATH + "ind.{}.x".format(dataset_str), 'rb') as f:
            x = pkl.load(f, encoding='latin1')
        with open(PATH + "ind.{}.y".format(dataset_str), 'rb') as f:
            y = pkl.load(f, encoding='latin1')
        print(y)
        '''
        
