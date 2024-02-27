import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from PAGER.PAGER import PAGER
import random

PAGER = PAGER()
PAG_result = PAGER.run_pager(
    ['BRCA1','BRCA2'],
    source = ['KEGG_2021_HUMAN', 'WikiPathway_2021', 'BioCarta', 'Reactome_2021', 'Spike'],
    Type='All',
    minSize=1,
    maxSize=2000,
    similarity = 0.05,
    overlap = 1,
    nCoCo = 0,
    pValue = 0.05,
    FDR = 0.05
)
PAGint = PAGER.pathInt(list(PAG_result.GS_ID)) #Get table of GS relations
PAG_member = PAGER.pathMember(list(PAG_result.GS_ID)) #Get table of GS and their Genes

array = PAGint.to_numpy() #Turn into numpy array
np.savetxt('data.txt', array, fmt='%s', delimiter=', ') #Print the array into 'data.txt'
array2 = PAG_member.to_numpy() #Turn into numpy array
np.savetxt('data2.txt', array2, fmt='%s', delimiter=', ') #Print the array into 'data2.txt'

#Make graph and get degrees of each node
G = nx.Graph()
for row in array:
    num = 0
    num = random.random()
    if num <= float(row[5]):
        node1 = row[0]
        node2 = row[2]
        weight = float(row[5])

        G.add_edge(node1, node2, weight=weight)
print(G)

#Add the degree of each node    
#NOTE: Some nodes have a bi-directional relationship and some do not. Once it becomes a nx.Graph, all edges are one direction.
degree_dict = dict(G.degree())
for node in G.nodes():
    node_degree = degree_dict[node]
    degree_list = [node_degree]
    G.nodes[node]['degree_list'] = degree_list
'''
for node in G.nodes():
    degree_list = G.nodes[node]['degree_list']
    print(f"Node {node} has a degree list: {degree_list}")
'''

#Create a dictionary of gene symbols for each node
gene_symbols_dict = {}
for row in array2:
    node = row[0] 
    gene_symbol = row[1]  
    if node in gene_symbols_dict:
        gene_symbols_dict[node].add(gene_symbol)
    else:
        gene_symbols_dict[node] = {gene_symbol}

#Append the list of gene symbols to each node
for node in G.nodes():
    # Get the gene symbols for the node from the dictionary
    gene_symbols = list(gene_symbols_dict.get(node, []))
    # Append the list of gene symbols as a node attribute
    G.nodes[node]['gene_symbols'] = gene_symbols

#Append the list of shared gene symbols between each node to the node edge
for node1, node2 in G.edges():
    genes_node1 = gene_symbols_dict.get(node1, set())
    genes_node2 = gene_symbols_dict.get(node2, set())
    
    shared_genes = genes_node1.intersection(genes_node2)
    
    G.edges[node1, node2]['shared_genes'] = list(shared_genes)

#Create a visualization of the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.15, iterations=20)
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='skyblue', alpha=0.7)
edges = nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.15)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.axis('off')
# Save and show the graph
#plt.savefig("Graph_Creation.jpg")
plt.show()