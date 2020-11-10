# MORE
 This is an Implementation of the MORE algorithm as described in our paper:
 
 Jin Xu, Shuo Yu, Ke Sun, Jing Ren, Ivan Lee, Shirui Pan, and Feng Xia. 2020. Multivariate Relations Aggregation Learning in Social Networks. In <i>Proceedings of the ACM/IEEE Joint Conference on Digital Libraries in 2020</i> (<i>JCDL '20</i>). Association for Computing Machinery, New York, NY, USA, 77–86. DOI:https://doi.org/10.1145/3383583.3398518
 
 The paper has been included in JCDL2020 and get the Vannevar Bush Best Paper Honorable Mention.
 
 ## Requirements
  - Tensorflow
  - networkx

 ## Theoretical Basis
  This algorithm uses the network motif to represent the rich multivariate relational structure in the network and incorporates it as a structural feature into the graph learning process. Specifically, by separately extracting the structural features and attribute features of the network data, combined with the aggregation learning process based on GCN or other network representation learning methods, we can obtain a more effective network representation tensor. In the paper, we apply the network representation tensor to the network node classification task through the softmax structure.
 
 ## Acknowledgements
 The original version of this code base was originally forked from https://github.com/tkipf/gcn/, and we owe many thanks to Thomas Kipf for making his code available.
