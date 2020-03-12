# Author:   Jin Xu
# Data:     2019-12-24
# Function: Pre-Processing the dataset

#--------------------------     import package    --------------------------#

import os
import re
import random
import codecs
import numpy as np
import networkx as nx
import scipy as sp

#--------------------------      global variable     --------------------------#

DIR_PATH = "C:\\Users\\HP\\Desktop\\Exp_TotalEmbeddedCombinationGCN\\"

#-------------------------- component function --------------------------#

def polblogs():
    """ Process the polblogs dataset"""

    # 1.1 Read the file
    original_file   = DIR_PATH + "origin_data\\polblogs\\polblogs.gml"
    node_pattern    = re.compile(r'id \d\d*')
    value_pattern   = re.compile(r'value \d\d*')
    source_pattern  = re.compile(r'source \d\d*')
    target_pattern  = re.compile(r'target \d\d*')
    with codecs.open(original_file, "r",encoding='utf-8', errors='ignore') as fdata:
        comtent = fdata.read()
        node_strlist   = re.findall(node_pattern, comtent)
        value_strlist  = re.findall(value_pattern, comtent)
        source_strlist = re.findall(source_pattern, comtent)
        target_strlist = re.findall(target_pattern, comtent)
    
    # 1.2 Get the original list which start from '0'
    origin_node_list, origin_label_list, source_list, target_list = [], [], [], []
    for item in node_strlist:
        origin_node_list.append(int(item.replace('id ','')))
    min_node, node_num = min(origin_node_list), len(origin_node_list)
    for i in range(0, node_num):
        origin_node_list[i] = origin_node_list[i] - min_node
    for item in value_strlist:
        origin_label_list.append(int(item.replace('value ','')))
    for item in source_strlist:
        source_list.append(int(item.replace('source ','')) - 1)
    for item in target_strlist:
        target_list.append(int(item.replace('target ','')) - 1)
    
    # 2.1 Get the node list
    OrphanedNode_count = 0
    node_list, label_list = [], []
    for i in range(0, node_num):
        node = origin_node_list[i]
        if node in source_list or node in target_list:
            node_list.append(node)
            label_list.append(origin_label_list[i])
        else:
            OrphanedNode_count += 1
    print(">> Origin node num: {} - Orphaned node num: {} = Now node num: {}."\
        .format(node_num, OrphanedNode_count, len(node_list)))
    node_num = len(node_list)

    # 2.2 Get the dictionary
    ID2label = {}
    for i in range(0, node_num):
        ID2label[node_list[i]] = label_list[i]
    node_index = list(range(0, node_num))
    random.shuffle(node_index)
    ID2index, index2ID = {}, {}
    for i in range(0, node_num):
        ID2index[node_list[i]] = node_index[i]
        index2ID[node_index[i]] = node_list[i]
    index2label = {}
    for i in range(0, node_num):
        index2label[i] =  ID2label[ index2ID[i] ]

    # 2.3 Build the edge list
    edge_list = []
    mutiedge_count = 0
    for i in range(0, len(source_list)):
        temp1 = ( ID2index[ source_list[i] ], ID2index[ target_list[i] ] )
        temp2 = ( ID2index[ target_list[i] ], ID2index[ source_list[i] ] )
        if temp1 not in edge_list and temp2 not in edge_list:
            edge_list.append(temp1)
        else:
            mutiedge_count += 1
    print(">> Origin edge num: {} - Muti edge num: {} = Now edge num: {}."\
        .format(len(source_list), mutiedge_count, len(edge_list)))

    # 2.4 Build the Graph
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    G = G.to_undirected()
    print(">> Build the Graph with {} nodes and {} edges."\
        .format(G.number_of_nodes(), G.number_of_edges()))
    
    # 3.0 Build the network file
    
    data_file   = DIR_PATH + "origin_data\\polblogs\\polblogs_Data.csv"
    label_file   = DIR_PATH + "origin_data\\polblogs\\polblogs_Label.csv"
    
    if os.path.exists(data_file) == True:
        os.remove(data_file)
    if os.path.exists(label_file) == True:
        os.remove(label_file)

    with open(data_file, "a") as f: 
        f.write("0,1\n")
        for edge in edge_list:
            f.write("{},{}\n".format(edge[0], edge[1]))
    with open(label_file, "a") as f: 
        f.write("0,1\n")
        for i in range(0, node_num):
            f.write("{},{}\n".format(i, index2label[i] + 1))
    print(">> Build file finish!")

    return


def football():
    """ Process the football dataset"""

    # 1.1 Read the file
    original_file   = DIR_PATH + "origin_data\\football\\football.gml"
    node_pattern    = re.compile(r'id \d\d*')
    value_pattern   = re.compile(r'value \d\d*')
    source_pattern  = re.compile(r'source \d\d*')
    target_pattern  = re.compile(r'target \d\d*')
    with codecs.open(original_file, "r",encoding='utf-8', errors='ignore') as fdata:
        comtent = fdata.read()
        node_strlist   = re.findall(node_pattern, comtent)
        value_strlist  = re.findall(value_pattern, comtent)
        source_strlist = re.findall(source_pattern, comtent)
        target_strlist = re.findall(target_pattern, comtent)
    
    # 1.2 Get the original list which start from '0'
    origin_node_list, origin_label_list, source_list, target_list = [], [], [], []
    for item in node_strlist:
        origin_node_list.append(int(item.replace('id ','')))
    min_node, node_num = min(origin_node_list), len(origin_node_list)
    for i in range(0, node_num):
        origin_node_list[i] = origin_node_list[i] - min_node
    for item in value_strlist:
        origin_label_list.append(int(item.replace('value ','')))
    for item in source_strlist:
        source_list.append(int(item.replace('source ','')))
    for item in target_strlist:
        target_list.append(int(item.replace('target ','')))
    
    # 2.1 Get the node list
    OrphanedNode_count = 0
    node_list, label_list = [], []
    for i in range(0, node_num):
        node = origin_node_list[i]
        if node in source_list or node in target_list:
            node_list.append(node)
            label_list.append(origin_label_list[i])
        else:
            OrphanedNode_count += 1
    print(">> Origin node num: {} - Orphaned node num: {} = Now node num: {}."\
        .format(node_num, OrphanedNode_count, len(node_list)))
    node_num = len(node_list)

    a = target_list
    print(a, len(a))
    os.system('pause')

    # 2.2 Get the dictionary
    ID2label = {}
    for i in range(0, node_num):
        ID2label[node_list[i]] = label_list[i]
    node_index = list(range(0, node_num))
    random.shuffle(node_index)
    ID2index, index2ID = {}, {}
    for i in range(0, node_num):
        ID2index[node_list[i]] = node_index[i]
        index2ID[node_index[i]] = node_list[i]
    index2label = {}
    for i in range(0, node_num):
        index2label[i] =  ID2label[ index2ID[i] ]

    # 2.3 Build the edge list
    edge_list = []
    mutiedge_count = 0
    for i in range(0, len(source_list)):
        temp1 = ( ID2index[ source_list[i] ], ID2index[ target_list[i] ] )
        temp2 = ( ID2index[ target_list[i] ], ID2index[ source_list[i] ] )
        if temp1 not in edge_list and temp2 not in edge_list:
            edge_list.append(temp1)
        else:
            mutiedge_count += 1
    print(">> Origin edge num: {} - Muti edge num: {} = Now edge num: {}."\
        .format(len(source_list), mutiedge_count, len(edge_list)))

    # 2.4 Build the Graph
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    G = G.to_undirected()
    print(">> Build the Graph with {} nodes and {} edges."\
        .format(G.number_of_nodes(), G.number_of_edges()))
    
    # 3.0 Build the network file
    
    data_file   = DIR_PATH + "origin_data\\football\\football_Data.csv"
    label_file   = DIR_PATH + "origin_data\\football\\football_Label.csv"
    
    if os.path.exists(data_file) == True:
        os.remove(data_file)
    if os.path.exists(label_file) == True:
        os.remove(label_file)

    with open(data_file, "a") as f: 
        f.write("0,1\n")
        for edge in edge_list:
            f.write("{},{}\n".format(edge[0], edge[1]))
    with open(label_file, "a") as f: 
        f.write("0,1\n")
        for i in range(0, node_num):
            f.write("{},{}\n".format(i, index2label[i] + 1))
    print(">> Build file finish!")

    return


def TerrorAttack():
    """ Process the TerrorAttack dataset"""
    
    # 1.1 Read the node file and build node dictionary
    label2type = {"http://counterterror.mindswap.org/2005/terrorism.owl#Arson":0,\
                  "http://counterterror.mindswap.org/2005/terrorism.owl#Bombing":1,\
                  "http://counterterror.mindswap.org/2005/terrorism.owl#Kidnapping":2,\
                  "http://counterterror.mindswap.org/2005/terrorism.owl#NBCR_Attack":3,\
                  "http://counterterror.mindswap.org/2005/terrorism.owl#other_attack":4,\
                  "http://counterterror.mindswap.org/2005/terrorism.owl#Weapon_Attack":5}
    name2id, id2name, id2type, id2feature = {}, {}, {}, {}

    original_nodefile   = DIR_PATH + "origin_data\\TerrorAttack\\terrorist_attack.nodes"
    nodeid = 0
    with open(original_nodefile, 'r') as f:
        while True:
            temp = f.readline()
            if not temp:
                break
            node_info = temp.split()
            # build the dictionary
            name2id[node_info[0]] = nodeid
            id2name[nodeid] = node_info[0]
            id2type[nodeid] = label2type[node_info[-1]]
            id2feature[nodeid] = node_info[1:-1]
            nodeid += 1
    
    # 1.2 Read the edge file and build the edge info list
    original_edgefile   = DIR_PATH + "origin_data\\TerrorAttack\\terrorist_attack_loc.edges"
    source, target = [], []
    with open(original_edgefile, 'r') as f:
        while True:
            temp = f.readline()
            if not temp:
                break
            edge_info = temp.split()
            source.append(name2id[edge_info[0]])
            target.append(name2id[edge_info[1]])
    original_edgefile   = DIR_PATH + "origin_data\\TerrorAttack\\terrorist_attack_loc_org.edges"
    with open(original_edgefile, 'r') as f:
        while True:
            temp = f.readline()
            if not temp:
                break
            edge_info = temp.split()
            source.append(name2id[edge_info[0]])
            target.append(name2id[edge_info[1]])
    
    # 2.0 Build the node and edge list
    node_list = list(range(0, nodeid))
    print(">> Build the node list with {} nodes.".format(nodeid))
    edge_list = []
    for i in range(0, len(source)):
        if (source[i], target[i]) not in edge_list and (target[i], source[i]) not in edge_list:
            edge_list.append((source[i], target[i]))
    print(">> Build the edge list with {} -> {} edges.".format(len(source), len(edge_list)))

    # 3.0 Build the network file
    data_file   = DIR_PATH + "origin_data\\TerrorAttack\\TerrorAttack_Data.csv"
    label_file  = DIR_PATH + "origin_data\\TerrorAttack\\TerrorAttack_Label.csv"
    feature_file= DIR_PATH + "origin_data\\TerrorAttack\\TerrorAttack_Feature.csv"
    
    if os.path.exists(data_file) == True:
        os.remove(data_file)
    if os.path.exists(label_file) == True:
        os.remove(label_file)
    if os.path.exists(feature_file) == True:
        os.remove(feature_file)

    with open(data_file, "a") as f: 
        f.write("0,1\n")
        for edge in edge_list:
            f.write("{},{}\n".format(edge[0], edge[1]))
    with open(label_file, "a") as f: 
        f.write("0,1\n")
        for i in range(0, nodeid):
            f.write("{},{}\n".format(i, id2type[i] + 1))
    feature_num = len(id2feature[0])
    with open(feature_file, "a") as f: 
        temp = list(range(0, feature_num+1))
        temp = [str(x) for x in temp]
        temp = ','.join(temp)
        f.write("{}\n".format(temp))
        for i in range(0, nodeid):
            temp = id2feature[i]
            temp = [str(x) for x in temp]
            temp = str(i) + ',' + ','.join(temp)
            f.write("{}\n".format(temp))
        
    print(">> Build file finish!")

    return

    
def www(allow_noid_author = False, allow_nokeyword_paper = False):
    """ Process the WWW dataset"""

    # 0.0 Read the file
    original_paper_file   = DIR_PATH + "origin_data\\WWW09-18\\WWW_papers_info_09_13.txt"
    original_author_file   = DIR_PATH + "origin_data\\WWW09-18\\WWW_authors_info_09_13.txt"
    # paper pattern
    author_pattern  = re.compile(r'\"authors\": \[[^\[]*\}\]')
    person_pattern  = re.compile(r'\{[^\}]*\}')
    name_pattern    = re.compile(r'\"name\": \"[^,\{\}]*\"')
    id_pattern    = re.compile(r'\"id\": \"[^,\{\}]*\"')
    keyword_pattern = re.compile(r'\"keywords\": \[[^\]]*\]')
    word_pattern    = re.compile(r'\"[^,]*\"')
    # author pattern
    tag_pattern     = re.compile(r'\"tags\": \[[^\]]*\}\]')
    t_pattern       = re.compile(r'\"t\": \"[^\"]*\"')
    

    # 1.0 Process the paper file
    keyword_name2index = {}
    keyword_index, author_index, paper_index, author_noid_counting, auther_all_counting = 0, 0, 0, 0, 0
    author_name2id, author_name2index, author_id2name = {}, {}, {}
    paper2keyword, keyword2paper = {}, {}
    paper2auther_name, autherindex2paper = {}, {}
    word_total_list, author_idlist, author_namelist = [], [], []

    with open(original_paper_file, 'r') as fdata:
        for eachline in fdata:
            # 1.1.1 Read the keyword
            keyword_strlist   = re.findall(keyword_pattern, eachline)
            if allow_nokeyword_paper == False:
                if len(keyword_strlist) == 0:
                    continue

            if len(keyword_strlist) == 0:
                keyword_strlist = ["\"keywords\": [\"None\"]"]
            temp = keyword_strlist[0]
            temp = temp.replace("\"keywords\": ", "")
            word_strlist      = re.findall(word_pattern, temp)
            for i in range(0, len(word_strlist)):
                word_strlist[i] = word_strlist[i].replace("\"", "")
            
            # 1.1.2 Process the word
            for word in word_strlist:
                if word not in word_total_list:
                    word_total_list.append(word)
                    keyword_name2index[word] = keyword_index
                    keyword_index += 1
                
                if keyword_name2index[word] not in keyword2paper.keys():
                    keyword2paper[keyword_name2index[word]] = [paper_index]
                else:
                    keyword2paper[keyword_name2index[word]].append(paper_index)
            
            # 1.2.1 Read the author
            author_strlist   = re.findall(author_pattern, eachline)
            person_strlist   = re.findall(person_pattern, author_strlist[0])
        
            if allow_noid_author == False:
                flag_allauthorhaveid = True
                for eachperson in person_strlist:
                    # get author id
                    author_id   = re.findall(id_pattern, eachperson)
                    if len(author_id) == 0:
                        flag_allauthorhaveid = False
                        break
                if flag_allauthorhaveid == False:
                    continue

            # 1.2.2 Process the author
            author_list = []
            for eachperson in person_strlist:
                auther_all_counting += 1
                # get author name
                author_name = re.findall(name_pattern, eachperson)
                if len(author_name) >= 2 or len(author_name) == 0:
                    raise Exception("[ERROR] have over 2 author name or 0 author in 1 line.\n        {}".format(eachline))
                else:
                    author_name = author_name[0]
                    author_name = author_name.replace("\"name\": ", "")
                    author_name = author_name.replace("\"", "")
                author_list.append(author_name)
                # get author id
                author_id   = re.findall(id_pattern, eachperson)
                if len(author_id) == 1:
                    author_id = author_id[0]
                    author_id = author_id.replace("\"id\": ", "")
                    author_id = author_id.replace("\"", "")
                elif len(author_id) == 0:
                    author_id = "0"
                else:
                    raise Exception("[ERROR] have over 2 author id in 1 line.\n        {}".format(eachline))
                # build the author dictionary
                if author_id not in author_idlist and author_id != "0":
                    author_idlist.append(author_id)
                    author_namelist.append(author_name)
                    author_name2id[author_name] = author_id
                    author_id2name[author_id] = author_name
                    author_name2index[author_name] = author_index
                    author_index += 1
                if  author_name not in author_namelist and author_id == "0":
                    author_namelist.append(author_name)
                    author_name2id[author_name] = author_id
                    author_name2index[author_name] = author_index
                    author_index += 1
                    author_noid_counting += 1
                
                if author_name in author_name2index.keys():
                    temp = author_name2index[author_name]
                elif author_id in author_id2name.keys():
                    temp = author_name2index[author_id2name[author_id]]
                else:
                    raise Exception("[ERROR] author_name or author_id can`t be found in dictionary.")

                if temp not in autherindex2paper.keys():
                    autherindex2paper[temp] = [paper_index]
                else:
                    autherindex2paper[temp].append(paper_index)

            # 1.3 Build paper dictionary
            paper2keyword[paper_index] = word_strlist
            paper2auther_name[paper_index] = author_list
            paper_index += 1
    # 1.4 Display
    print(">> Process paper finish![Opintion: allow_noid_author?-{}, allow_nokeyword_paper?-{}]".format(allow_noid_author, allow_nokeyword_paper))
    print("   Paper number = {}, Keyword number = {}".format(paper_index, keyword_index))
    print("   Author number = {}, where with {} no-ID author.".format(author_index, author_noid_counting))
    print("   Total authors = {}, reply times = {}".format(auther_all_counting, auther_all_counting - author_index))
    
    # 2.0 Process the author file
    tag_name2index = {}
    tag_index = 0
    with open(original_author_file, 'r') as fdata:
        for eachline in fdata:
            idstr  = re.findall(id_pattern, eachline)
            idstr = idstr[0].replace("\"id\": ", "")
            idno = idstr.replace("\"", "")
            # author id not in the paper file 
            if idno not in author_id2name.keys():
                continue

            tag_strlist = re.findall(tag_pattern, eachline)
            t_strlist = re.findall(t_pattern, tag_strlist[0])
            for i in range(0, len(t_strlist)):
                t_strlist[i] = t_strlist[i].replace("\"t\": ", "")
                t_strlist[i] = t_strlist[i].replace("\"", "")
            for tag in t_strlist:
                if tag not in tag_name2index.keys():
                    tag_name2index[tag] = tag_index
                    tag_index += 1
                
            
            print(idno, t_strlist)
    return

###################################################################################

if __name__ == "__main__":
    #polblogs()
    #football()
    #TerrorAttack()
    www()
