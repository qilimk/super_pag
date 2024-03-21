import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import community as community_louvain
import random
import hdbscan
import time
import itertools
import seaborn as sns
import argparse


from scipy.stats import ttest_ind
from collections import Counter, deque
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import adjusted_rand_score,  normalized_mutual_info_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE   
from networkx.algorithms.community import girvan_newman
from sklearn.cluster import SpectralClustering

from communities_detection import plot_communities_w_names
from PAGER.PAGER import PAGER

PAGER = PAGER()
VERI_GRAGH_PATH = '2024_graph_w_verification.graphml'
WHOLE_GRAPH_PATH = '2024_GO_wholegraph.graphml'
META_DATA_PATH = 'filterpaginf2024.csv'   # 'pager_3_all_metadata.csv'


class GraphAnalysis:
    def __init__(self, G, root):
        self.G = G
        self.root = root
        self.parent = {root: None}  # Parent of each node
        self.level = {root: 0}  # Level of each node
        self._bfs()  # Perform BFS to populate parent and level information
        
    def _bfs(self):
        """Perform BFS to populate parent and level information."""
        queue = deque([self.root])
        while queue:
            current_node = queue.popleft()
            for neighbor in self.G.successors(current_node):
                if neighbor not in self.parent:  # If neighbor has not been visited
                    self.parent[neighbor] = current_node
                    self.level[neighbor] = self.level[current_node] + 1
                    queue.append(neighbor)
    
    def check_same_parent(self, node1, node2):
        """Check if two nodes have the same parent and determine their level."""
        if node1 not in self.parent or node2 not in self.parent:
            return None, None
        
        have_same_parent = self.parent[node1] == self.parent[node2]
        if have_same_parent:
            return have_same_parent, self.level[node1]  # Assuming they are at the same level
        else:
            return have_same_parent, (self.level[node1], self.level[node2])


def tsne_2d_plot(pags_embeddings, save_name='test'):

    tsne = TSNE(n_components=2, perplexity=5,  random_state=323)
    embeddings_2d = tsne.fit_transform(pags_embeddings)
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    plt.title("2D Visualization of Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(f"{save_name}.jpg")

# from communities_detection import run_eval
def evaluate_algorithm(true_labels, pred_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return ari, nmi

def run_eval(algorithm, G, true_labels, is_plot=False):
    
    if algorithm == 'louvain':
        partition = community_louvain.best_partition(G)
        title = "Louvain Method"
    elif algorithm == 'girvan_newman':
        gn_comm = next(girvan_newman(G))
        partition = {node: i for i, comm in enumerate(gn_comm) for node in comm}
        title = "Girvan-Newman Algorithm"
    elif algorithm == 'spectral_clustering':
        adj_matrix = nx.to_numpy_array(G)
        sc = SpectralClustering(n_clusters=3, affinity='precomputed', n_init=100, random_state=42)
        sc.fit(adj_matrix)
        nodes = list(G.nodes())
        partition = {nodes[i]: sc.labels_[i] for i in range(len(nodes))}
        title = "Spectral Clustering"
    else:
        raise ValueError("Invalid algorithm")
    
    # print(partition)
    
    labels = [partition[node] for node in G.nodes()]
    ari, nmi = evaluate_algorithm(true_labels, labels)

    if is_plot:
        print(f"{title}: ARI = {ari}, NMI = {nmi}")
        # plot_communities(G, partition, title)
        plot_communities_w_names(G, partition, f"{algorithm}_ari_{ari:0.2f}_nmi_{nmi:0.2f}")

    return ari, nmi



def plot_communities(G, partition, title=""):
    cmap = plt.colormaps.get_cmap('viridis')
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{title.replace(' ', '_')}_results.jpg")
    plt.close()  # Close the plot to avoid displaying it inline if not desired


def talbe_2_graph(file_path = 'topdown2018.txt'):

    G = nx.DiGraph()

    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):  # Enumerate to keep track of line numbers
            try:
                parent, child = line.strip().split(' ')
                if (parent, child) not in G.edges():  # Check to avoid adding duplicate edges
                    G.add_edge(parent, child)
            except ValueError:
                # Handle the error (e.g., wrong number of values to unpack)
                print(f"Skipping malformed line: {line.strip()}")
            except Exception as e:
                # Handle other possible errors
                print(f"An error occurred: {e}. Skipping line: {line.strip()}")

    nx.write_graphml(G, "graph.graphml")




def find_smallest_subgraph(G):
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]

    print("Root nodes:", roots)

    smallest_subgraph = nx.DiGraph()

    smallest_size = float('inf') 
    tree_sizes = {}
    for root in roots:

        # Perform DFS to find all nodes reachable from the root
        reachable_nodes = nx.dfs_tree(G, root)
        # The size of the tree is the number of reachable nodes
        tree_sizes[root] = len(reachable_nodes)
        if tree_sizes[root] < smallest_size:
            print(tree_sizes[root])
            smallest_size = tree_sizes[root]
            smallest_subgraph = nx.dfs_tree(G, root)


    print("Root nodes and the size of each tree:")
    for root, size in tree_sizes.items():
        print(f"Root: {root}, Size: {size}")

    print(f"Smallest Graph: {smallest_subgraph} ")

    return smallest_subgraph

def find_small_subtrees(G, max_size):
    small_subtrees = []

    # Iterate over all nodes in the graph
    for node in G.nodes():
        # Perform a limited BFS from each node
        visited = set()
        queue = [node]
        
        while queue and len(visited) < max_size:
            current_node = queue.pop(0)
            if current_node not in visited:
                visited.add(current_node)
                # Add neighbors that haven't been visited to the queue
                queue.extend([neighbor for neighbor in G.successors(current_node) if neighbor not in visited])
        
        # Check if the visited nodes form a subtree of the desired size
        if 1 < len(visited) < max_size:
            # Extract the subgraph and add it to the list
            small_subtrees.append(G.subgraph(visited).copy())
    
    return small_subtrees

# Function to find all subtrees with more than n nodes
def find_large_subtrees(G, min_size):
    large_subtrees = {}
    for node in G.nodes():
        subtree = nx.dfs_tree(G, node)
        if subtree.number_of_nodes() > min_size:
            large_subtrees[node] = subtree
    return large_subtrees


def test():
    n = 300

    graph_path = 'graph_w_verification.graphml'
    G = nx.read_graphml(graph_path)
    # smallest_subgraph = find_smallest_subgraph(G)
    large_subtrees = find_large_subtrees(G, n)

    print(f"Found {len(large_subtrees)} subtrees with fewer than {n} nodes.")
    for root_name, subtree in large_subtrees.items():
        # print(f"Subtree {root_name} contains {subtree.number_of_nodes()} nodes, out degree: {subtree.out_degree(root_name)} {len(subtree.edges())} edges.")
        if root_name == "GO:0006811":
            print('create a subtree for testing.')
            nx.write_graphml(subtree, "subgraph_GO_0006811.graphml")

            cycle = nx.find_cycle(G)
            print("Cycle detected:", cycle)


    print('finished')

def check_data():
    meta_data_path = 'pager_3_all_metadata.csv'
    df = pd.read_csv(meta_data_path)
    print(df.head())

    # talbe_2_graph(file_path = 'topdown2018.txt')
    # Path to your GraphML file
    graph_path = 'subgraph_GO_0006811.graphml'
    G = nx.read_graphml(graph_path)
    children_of_root = list(G.successors('GO:0034220'))
    print(children_of_root)

    list_of_nodes = list(G.nodes())
    print(len(list_of_nodes))
    counts = Counter(list_of_nodes)
    # print(counts)

    column_to_check = 'NAME'

    # Check if each name is a substring of any entry in the DataFrame column
    matches = {name: df[column_to_check].str.contains(name).any() for name in list_of_nodes}

    # Names found as substrings within the DataFrame column
    found_names = [name for name, found in matches.items() if found]

    # Names not found as substrings within the DataFrame column
    not_found_names = [name for name, found in matches.items() if not found]


    childrens_of_target = []
    # Print results
    print("Names found in the DataFrame:")
    for name in found_names:

        if node_belongs_to_tree(G, 'GO:0006811', name):
            childrens_of_target.append(name)

        if name in children_of_root:
            print(f"{name} is found.")

    print(f"childrens_of_target: {len(childrens_of_target)}")
    print(f"found_names: {len(found_names)}")
    with open('target_cluster.txt', 'w') as file:
        file.write(','.join(childrens_of_target))
    with open('found_nodes_list.txt', 'w') as file1:
        file1.write(','.join(found_names))



def plot_highlight_nodes(G, highlight_nodes):

    # Extract node colors based on whether they are in the highlight list or not
    node_colors = ['red' if node in highlight_nodes else 'skyblue' for node in G.nodes()]

    # Draw the graph
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw_networkx(G, pos, with_labels=False, node_color=node_colors, node_size=30, edge_color='gray')

    # Highlight edges connected to the highlighted nodes
    edge_colors = ['red' if (u in highlight_nodes or v in highlight_nodes) else 'black' for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors)

    # Show plot
    plt.title('Graph with Highlighted Nodes')
    plt.show()




def node_belongs_to_tree(graph, root, target):
    """
    Check if the target node belongs to the tree rooted at the given root node.
    
    Parameters:
    - graph: A networkx Graph representing the tree.
    - root: The root node of the tree.
    - target: The node to check for its presence in the tree.
    
    Returns:
    - True if the target node belongs to the tree rooted at the given root node, False otherwise.
    """
    # Use DFS to find if the target node is reachable from the root
    reachable_nodes = list(nx.dfs_preorder_nodes(graph, source=root))
    return target in reachable_nodes


def find_validate_set(graph_path = 'subgraph_for_test.graphml'):
    meta_data_path = 'pager_3_all_metadata.csv'
    df = pd.read_csv(meta_data_path)
    print(df.head())

    # Path to your GraphML file
    G = nx.read_graphml(graph_path)
    list_of_nodes = list(G.nodes())

    column_to_check = 'NAME'

    # Construct a regular expression pattern to match any of the elements in name_parts
    pattern = '|'.join(list_of_nodes)

    # Find rows where the name contains any of the substrings in name_parts
    matching_rows = df[df[column_to_check].str.contains(pattern, case=False, na=False)]
    desc_list = matching_rows['DESCRIPTION'].tolist()

    return matching_rows, desc_list



def clean_data():
    meta_data_path = 'filterpaginf2024.csv'
    df = pd.read_csv(meta_data_path)
    print(df.head())
    three_parts = ['biological_process', 'molecular_function', 'cellular_component']

    for part in three_parts:
        # Filter the DataFrame for rows where 'NAME' contains the current part
        filtered_df = df[df['NAME'].str.contains(part)]
    
        # Apply the lambda function to the filtered DataFrame and convert to list
        go_key_list = filtered_df['NAME'].apply(lambda x: x.split(' ')[0]).tolist()
        
        table_2_graph_with_verfication('topdown2024.txt', go_key_list, part)

def get_embedding(desc):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder.encode(desc)

def table_2_graph(file_path='topdown2024.txt'):

    G = nx.DiGraph()

    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                parent, child = line.strip().split('\t')
                G.add_edge(parent, child)

            except ValueError:
                # Handle the error (e.g., wrong number of values to unpack)
                print(f"Skipping malformed line {line_number}: {line.strip()}")
            except Exception as e:
                # Handle other possible errors
                print(f"Error on line {line_number}: {e}. Skipping line: {line.strip()}")

    nx.write_graphml(G, "2024_GO_wholegraph.graphml")



def table_2_graph_with_verfication(file_path, valid_nodes, part_name):
    G = nx.DiGraph()

    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                parent, child = line.strip().split('\t')
                # Verify both nodes are in the valid_nodes list before adding
                if parent in valid_nodes and child in valid_nodes:
                    if (parent, child) not in G.edges():  # Check to avoid adding duplicate edges
                        print(f'adding nodes {parent} and {child}')
                        G.add_edge(parent, child)
                else:
                    print(f"Skipping line {line_number}: Node not in valid_nodes list.")
            except ValueError:
                # Handle the error (e.g., wrong number of values to unpack)
                print(f"Skipping malformed line {line_number}: {line.strip()}")
            except Exception as e:
                # Handle other possible errors
                print(f"Error on line {line_number}: {e}. Skipping line: {line.strip()}")

    nx.write_graphml(G, f"2024_{part_name}_graph_w_verification.graphml")


def extract_groundtrue_dataset(graph_path = 'subgraph_GO:0140096_GO:0051239.graphml'):

    G = nx.read_graphml(graph_path)
    labels = {}
    for node in nx.descendants(G, 'GO:0140096'):
        labels['GO:0140096'] = 0
        labels[node] = 0
    for node in nx.descendants(G, 'GO:0051239'):
        labels['GO:0051239'] = 1
        labels[node] = 1
    return G, [labels.get(node, -1) for node in sorted(G.nodes())], labels
      
def graph_based_clustering_eval():
    G, true_labels = extract_groundtrue_dataset()
    print(f"The number of nodes: {len(G.nodes())}")
    G_undirected = G.to_undirected()
    for algorithm in ['girvan_newman',  'louvain', 'spectral_clustering']: #'girvan_newman',  'louvain', 
        ari, nmi = run_eval(algorithm, G_undirected, true_labels, is_plot=True)
        
        print(f"{algorithm}: ARI: {ari}, NMI: {nmi}")

def find_second_level_subtrees(G, root, M):
    # Level dictionary to keep track of the levels of nodes
    level = {root: 0}
    # Queue for BFS traversal, initialized with the root node
    queue = [root]
    # Second level nodes
    second_level_nodes = []

    while queue:
        # Get the next node from the queue
        current_node = queue.pop(0)
        current_level = level[current_node]

        # If the current node is on the second level, add it to the list
        if current_level == 2:
            second_level_nodes.append(current_node)

        # If the current node is beyond the second level, stop processing further
        if current_level > 2:
            break

        # Add neighbors of the current node to the queue and update their levels
        for neighbor in G.neighbors(current_node):
            if neighbor not in level:  # Ensure each node is processed only once
                level[neighbor] = current_level + 1
                queue.append(neighbor)

    # For each second level node, generate its subtree
    subtrees = {}
    for node in second_level_nodes:
        # Use BFS or DFS to get all nodes in the subtree rooted at 'node'
        subtree_nodes = nx.descendants(G, node)
        
        # Check if the subtree size meets the condition
        if len(subtree_nodes) < M:
            subtrees[node] = subtree_nodes

    return subtrees


def find_subtree_with_criteria(G, root, upper_num=100, lower_num=50):
    # Perform a BFS to find nodes at each level and compute subtree sizes
    queue = [(root, 0)]  # Each element is a tuple (node, level)
    level_nodes = {}  # {level: [nodes]}
    subtree_sizes = {}  # {node: size_of_subtree}

    while queue:
        current_node, level = queue.pop(0)
        # Compute the size of the subtree rooted at `current_node`
        subtree_size = len(nx.descendants(G, current_node)) + 1  # Including the node itself
        subtree_sizes[current_node] = subtree_size

        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(current_node)

        queue.extend((neighbor, level + 1) for neighbor in G.successors(current_node))
    # Filter for nodes that meet the minimum size requirement
    eligible_nodes = {node: size for node, size in subtree_sizes.items() if size <= upper_num and size > lower_num}

    # Find the subtree rooted at the lowest level that meets the size criterion
    for level in sorted(level_nodes.keys()):
        for node in level_nodes[level]:
            if node in eligible_nodes:
                return node, eligible_nodes[node]  # Return the node and the size of its subtree

    return None, None  # If no such subtree is found


def loadPAGERSubset():
    G, true_labels_list, true_label_dict = extract_groundtrue_dataset()

    df, desc_list = find_validate_set(graph_path = 'subgraph_GO:0140096_GO:0051239.graphml')

    df_no_root = df[~df['NAME'].str.contains('GO:0006811')]
    df_no_root['KEYNAME'] = df_no_root['NAME'].str.split(" ").str[0]


    PAGint = PAGER.pathInt(df_no_root['GS_ID']) #Get table of GS relations
    PAG_member = PAGER.pathMember(df_no_root['GS_ID']) #Get table of GS and their Genes

    array = PAGint.to_numpy() #Turn into numpy array
    np.savetxt('data.txt', array, fmt='%s', delimiter=', ') #Print the array into 'data.txt'
    array2 = PAG_member.to_numpy() #Turn into numpy array
    np.savetxt('data2.txt', array2, fmt='%s', delimiter=', ') #Print the array into 'data2.txt'

    #Make graph and get degrees of each node
    G = nx.Graph()
    for row in array:
        node1 = row[0]
        node2 = row[2]
        weight = float(row[5])
        G.add_edge(node1, node2, weight=weight)

    print(G)
    # Convert the NetworkX graph to an adjacency matrix (NumPy array)
    adj_matrix = nx.to_numpy_array(G)
    # Apply Spectral Clustering
    # n_clusters is the number of clusters you wish to find in the graph
    sc = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
    sc.fit(adj_matrix)
    partition = {node: label for node, label in zip(G.nodes(), sc.labels_)}
    labels = [partition[node] for node in G.nodes()]
    true_labels = [true_label_dict[df_no_root.loc[df_no_root['GS_ID'] == node, 'KEYNAME'].iloc[0]] for node in G.nodes()]

    sc_ari = adjusted_rand_score(true_labels, labels)
    print(f"sc ari: {sc_ari}")

    plot_communities_w_names(G, partition, "sc_pags_clustering_subgraph_GO:0140096_GO:0051239")
    
def filtered_outdegree_in_G(graph_path= '2024_biological_process_graph_w_verification.graphml', lower_bound=10, upper_bound= 100):
    G = nx.read_graphml(graph_path)
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]
    # node_out_degrees = G.out_degree()
    filtered_list = []

    node_out_degrees = {}
    for root in roots:
        root_out_degree = G.out_degree(root)
        node_out_degrees[root] = root_out_degree
        if root_out_degree < upper_bound and root_out_degree > lower_bound:
            filtered_list.append(root)
            print(f'{root}: {root_out_degree}')
    
    # ranked_nodes = sorted(node_out_degrees, key=lambda x: x[1], reverse=False)
    return filtered_list

def extract_subgraph_given_nodes(G, nodes_to_extract):

    # GO:0140096 and GO:0051239
    edges_to_add = []  # A list to store the subtree graphs

    for node in nodes_to_extract:
        # Find direct children of the node
        children = list(G.successors(node))
        
        # Create edges from the node to its direct children
        for child in children:
            edges_to_add.append((node, child))
        
        # Create a new graph for the subtree
    subgraph = nx.DiGraph(edges_to_add)
    
    nx.write_graphml(subgraph, f"subgraph_{'_'.join(nodes_to_extract)}.graphml")

# G = nx.read_graphml(GRAGH_PATH)
# nodes_to_extract = ['GO:0140096', 'GO:0051239']
# extract_subgraph_given_nodes(G, nodes_to_extract)
    

# random_roots = random.sample(roots, 3)

# for root in random_roots:
#     subtree_root, subtree_size = find_subtree_with_criteria(G, root)
#     if subtree_root:
#         print(f"Found a subtree rooted at '{subtree_root}' with {subtree_size} nodes.")
#     else:
#         print("No suitable subtree found.")
    
def find_edges_among_neighbors(G, target_vertex):
    """
    Finds the edges among the neighbors of a given target vertex, excluding edges to the target vertex itself.
    
    Parameters:
    - G: A NetworkX graph.
    - target_vertex: The vertex for which neighbors' connections are analyzed.
    
    Returns:
    - edges: A set of tuples where each tuple represents an edge between two neighbors of the target vertex.
    """
    # Find neighbors of the target vertex
    neighbors = set(nx.neighbors(G, target_vertex))
    print(f"the neighbors of the vertex {target_vertex}: {neighbors}")
    
    # Initialize an empty set to store edges among neighbors
    edges_among_neighbors = set()
    
    # Check for edges among the neighbors
    for neighbor in neighbors:
        # For each neighbor, find its neighbors and check if they are also neighbors of the target vertex
        for potential_neighbor_edge in nx.neighbors(G, neighbor):
            if potential_neighbor_edge in neighbors and neighbor < potential_neighbor_edge:
                # Add the edge if both nodes are neighbors of the target vertex (and avoid adding an edge twice)
                edges_among_neighbors.add((neighbor, potential_neighbor_edge))
    
    return edges_among_neighbors, neighbors


def read_m_type_file(txt_path="go_metadata/m_type_biological_process.txt"):
    return pd.read_csv(txt_path, sep='\t')

def convert_weighted_graph_2_non_weighted(G_weighted):
    # Create an empty non-weighted graph
    G_non_weighted = nx.Graph()

    # Iterate through the weighted edges and add them based on their weight probability
    for u, v, data in G_weighted.edges(data=True):
        if np.random.rand() <= data['weight']:  # data['weight'] is treated as the probability
            G_non_weighted.add_edge(u, v)

    return G_non_weighted





def find_node_levels(G, root, nodes):
    level = {root: 0}
    queue = deque([root])
    result = {}
    
    while queue:
        current_node = queue.popleft()
        for neighbor in G.neighbors(current_node):
            if neighbor not in level:  # If neighbor hasn't been assigned a level
                level[neighbor] = level[current_node] + 1
                queue.append(neighbor)
    
    # For each node in the list, save its level in the result dictionary
    for node in nodes:
        result[node] = level.get(node, None)  # Use None for nodes not found in the graph
    
    return result

def have_same_parent(G, node1, node2):
    # Initialize a dictionary to hold the parent of each node
    parents = {}

    # Populate the parents dictionary by iterating over the edges
    for parent, child in G.edges():
        parents[child] = parent

    # Check if both nodes are in the parents dictionary and have the same parent
    if node1 in parents and node2 in parents and parents[node1] == parents[node2]:
        return True
    else:
        return False
    
def all_have_different_parent(G, nodes):
    # Initialize a dictionary to hold the parent of each node
    parents = {}

    # Populate the parents dictionary by iterating over the edges
    for parent, child in G.edges():
        parents[child] = parent

    # Use a set to keep track of unique parents of the nodes in the list
    unique_parents = set(parents[node] for node in nodes if node in parents)

    # If the set of unique parents has exactly one element, all nodes have the same parent
    return len(unique_parents) == len(nodes)


# step 1: verify the relationship table with the meta data of GOA dataset
# clean_data()
# step 2: find out the level 2 nodes and the outdegree within the range of 50 to 100
# sort_outdegree_in_G()

def filter_exp_dataset(target_graph_path='2024_biological_process_graph_w_verification.graphml', taget_graph_root = 'GO:0008150', num_of_groups = 2):
    '''
    biological_process: 'GO:0008150'
    molecular_function: 'GO:0003674'
    cellular_component: 'GO:0005575'
    '''
    
    G_GOA = nx.read_graphml(WHOLE_GRAPH_PATH)
    filtered_roots = filtered_outdegree_in_G(target_graph_path)
    node_2_level_dict = find_node_levels(G_GOA, taget_graph_root, filtered_roots)

    level_2_nodes_dict = {}
    exp_candidates = []

    for key, value in node_2_level_dict.items():
        if value not in level_2_nodes_dict:
            level_2_nodes_dict[value] = [key]
        else:
            level_2_nodes_dict[value].append(key)

    for level, nodes in level_2_nodes_dict.items():
        group_list = list(itertools.combinations(nodes, num_of_groups))
        for nodes in group_list:
            if all_have_different_parent(G_GOA, nodes):
                exp_candidates.append(nodes)
            else:
                print(f"The {nodes} share the same parent!!")

    print(len(exp_candidates))
    return exp_candidates


def gen_gt_labels(graph_path, roots_list):

    G = nx.read_graphml(graph_path)
    labels_dict = {}
    all_gt_nodes = []

    # Assign labels starting from 0 for each root and its neighbors
    for label, root in enumerate(roots_list):
        labels_dict[root] = label
        all_gt_nodes.append(root)
        for node in nx.neighbors(G, root):
            # Avoid re-labeling a node if it has already been labeled
            if node not in labels_dict:
                labels_dict[node] = label
                all_gt_nodes.append(node)

    # Ensure uniqueness of nodes
    all_gt_nodes = list(set(all_gt_nodes))
    
    # Generate labels for all nodes, sorted to ensure consistent order
    labels = [labels_dict.get(node, -1) for node in sorted(all_gt_nodes)]

    return all_gt_nodes, labels, labels_dict


def find_gt_valid_set(valid_nodes):
    
    df = pd.read_csv(META_DATA_PATH)
    column_to_check = 'NAME'
    pattern = '|'.join(valid_nodes)
    matching_rows = df[df[column_to_check].str.contains(pattern, case=False, na=False)]
    desc_list = matching_rows['DESCRIPTION'].tolist()

    return matching_rows, desc_list


def get_normalized_embeddings(df, opt='desc'):
    """Extract and normalize embeddings."""
    df['NAMEDESC'] = df['NAME'].apply(lambda x: ' '.join(x.split(' ')[2:]) if len(x.split(' ')) > 2 else 'none')
     
    if opt == 'name':
        df['EMBEDDING'] = df['NAMEDESC'].apply(get_embedding)
    elif opt == 'desc':
        df['EMBEDDING'] = df['DESCRIPTION'].apply(get_embedding)
    elif opt == 'name+desc':
        df['EMBEDDING'] = (df['NAMEDESC'] + ' ' + df['DESCRIPTION']).apply(get_embedding)
    else:
        print(f"The option {opt} is invalid..")

    embeddings = np.stack(df['EMBEDDING'].values)
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def perform_clustering(algo, name, embeddings, true_labels):
    """Fit clustering algorithm and print ARI score."""
    start_time = time.time()
    predictions = algo.fit_predict(embeddings)
    ari_score = adjusted_rand_score(true_labels, predictions)
    end_time = time.time()
    duration = end_time - start_time
    print(f"ARI Score {name}: {ari_score}, time spent: {duration}")

    return ari_score, duration

def eval_m_type_relation_clustering(graph_path, roots_list):

    df = read_m_type_file("go_metadata/m_type_biological_process.txt")
    valid_nodes, true_labels, true_labels_dict = gen_gt_labels(graph_path, roots_list)
    print(len(valid_nodes))
    nodes_dict = {f"Root{label}": root for label, root in enumerate(roots_list)}
    filtered_df = df[df['GS_A_ID'].isin(valid_nodes) & df['GS_B_ID'].isin(valid_nodes)]

    G = nx.Graph()
    for _, row in filtered_df.iterrows():
        G.add_edge(row['GS_A_ID'], row['GS_B_ID'], weight=row['SIMILARITY'])

    # G_non_weighted = convert_weighted_graph_2_non_weighted(G)

    # connection_list = find_connections(G_non_weighted)

    true_labels = [true_labels_dict[node] for node in G.nodes()]

    results = []

    for algorithm in ['girvan_newman',  'louvain', 'spectral_clustering']:  #'girvan_newman',  'louvain', 
        ari, nmi = run_eval(algorithm, G, true_labels, is_plot=True)
        result_dict = {'Algorithm': algorithm, 'ARI Score': ari, 'NMI': nmi, 'Sample size': len(valid_nodes)}
        result_dict.update(nodes_dict)
        results.append(result_dict)

    return results

def find_connections(G):

    connection_list = []
    for node in G.nodes():
        # target_vertex = int(node_id)
        edges, neighbors = find_edges_among_neighbors(G, node)
        number_of_neighbors = len(neighbors)
        connection = len(edges) *2 / (number_of_neighbors*(number_of_neighbors-1))
        connection_list.append(connection)

    return connection_list



def eval_llm_clustering(graph_path, roots_list):
    
    valid_nodes, true_labels, _ = gen_gt_labels(graph_path, roots_list)
    print(len(valid_nodes))
    nodes_dict = {f"Root{label}": root for label, root in enumerate(roots_list)}
    df, _ = find_gt_valid_set(valid_nodes)

    # Data preprocessing
    df['KEYNAME'] = df['NAME'].str.split(" ").str[0]
    
    # Sort and normalize embeddings
    df_sorted = df.sort_values(by='KEYNAME')
    pags_embeddings = get_normalized_embeddings(df_sorted)

    # Clustering and evaluation
    clustering_algorithms = {
        'K-Means': KMeans(n_clusters=2),
        'Agglomerative': AgglomerativeClustering(n_clusters=2),
        'DBSCAN': DBSCAN(eps=0.1, min_samples=40),
        'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    }

    results = []

    # Perform clustering with each algorithm
    for name, algo in clustering_algorithms.items():
        print({f"Start algo {name} evaluation..."})
        ari_score, duration = perform_clustering(algo, name, pags_embeddings, true_labels)
        result_dict = {'Algorithm': name, 'ARI Score': ari_score, 'Duration': duration, 'Sample size': len(valid_nodes)}
        result_dict.update(nodes_dict)
        results.append(result_dict)
    
    return results



def ari_results_box_plots(csv_file_path = 'clustering_3_groups_results.csv'):
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title("ARI Scores by Clustering Algorithm")
    boxplot = df.boxplot(column='ARI Score', by='Algorithm', grid=False)
    plt.suptitle('')
    plt.xlabel('Algorithm')
    plt.ylabel('ARI Score')
    plt.xticks(rotation=60)
    plt.savefig(f"{csv_file_path.split('.')[0]}.jpg")

def ari_comparison_box_plots(csv_file_path1, csv_file_path2):

    # Read the CSV files into DataFrames
    df1 = pd.read_csv(csv_file_path1)
    df2 = pd.read_csv(csv_file_path2)

    # Add a column to distinguish the data sets
    df1['DataSet'] = 'Two Groups Dataset'
    df2['DataSet'] = 'Three Groups Dataset'

    # Concatenate the DataFrames
    df_combined = pd.concat([df1, df2], ignore_index=True)
    df_filtered = df_combined[df_combined['Algorithm'] != 'DBSCAN']

    algorithms = df_filtered['Algorithm'].unique()
    algo_p_value = {}

    for algo in algorithms:

        df_scores = df_filtered[df_filtered['Algorithm'] == algo]
        # Compute t-test between the two groups for the selected algorithm
        group1_scores = df_scores[df_scores['DataSet'] == 'Two Groups Dataset']['ARI Score']
        group2_scores = df_scores[df_scores['DataSet'] == 'Three Groups Dataset']['ARI Score']
        t_stat, p_value = ttest_ind(group1_scores, group2_scores, equal_var=False) 
        algo_p_value[algo] = p_value
        print(f"T-test for {algo}: t-statistic = {t_stat}, p-value = {p_value}")

    # Plotting using seaborn for better control over aesthetics
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=df_filtered, x='DataSet', y='ARI Score', hue='Algorithm', width=0.6)
    plt.title("ARI Scores Comparison")
    # plt.xlabel('Data Set')
    plt.ylabel('ARI Score')
    
    # Get current y-axis ticks and labels
    ticks = ax.get_yticks()

    # Define the maximum value you want to display on the y-axis
    max_display_value = 1.1

    # Generate new tick labels, replace those above max_display_value with an empty string
    new_tick_labels = [str(round(tick, 1)) if tick <= max_display_value else '' for tick in ticks]

    # Apply the new tick labels
    ax.set_yticklabels(new_tick_labels)


    # plt.xticks(rotation=60)
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')

    hue_order = ax.get_legend_handles_labels()[1]
    n_hues = len(hue_order)

    # Get the current color palette, assuming it has at least as many colors as there are hues
    palette = sns.color_palette()[:n_hues]

    # Map the hue order to the colors
    hue_colors = dict(zip(hue_order, palette))

    y = df_filtered['ARI Score'].max()
    h = 0.1
    # Optionally, print or return the color for each hue level (Algorithm)
    for idx, (algo, col) in enumerate(hue_colors.items()):
        plt.plot([ -0.2+idx*0.2, -0.2+idx*0.2 , 0.8 + idx*0.2 , 0.8 + idx*0.2 ], 
                 [y+0.05, y+0.2+h*idx, y+0.2+ h*idx, y+0.05], 
                 lw=1.5, c=col)
        plt.text(0.35+idx*0.2, y+0.2+h*idx, f"p = {algo_p_value[algo]:.4f}", ha='center', va='bottom', color=col)
        print(f"Algorithm: {algo}, Color: {col}")

    # Save the plot
    plt.savefig("ari_scores_comparison.jpg", bbox_inches='tight')
    plt.show()
    

# Example usage
# ari_comparison_box_plots('clustering_results.csv', 'clustering_3_groups_results.csv')

def plot_algorithm_comparison_with_ttest_and_line(csv_file_path1, csv_file_path2, algorithm_name):
    # Load the datasets
    df1 = pd.read_csv(csv_file_path1)
    df2 = pd.read_csv(csv_file_path2)
    
    # Add an identifier for the dataset
    df1['DataSet'] = 'Set 1'
    df2['DataSet'] = 'Set 2'
    
    # Combine the datasets
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    # Filter by the algorithm of interest
    df_filtered = df_combined[df_combined['Algorithm'] == algorithm_name]
    
    # Compute t-test between the two groups for the selected algorithm
    group1_scores = df_filtered[df_filtered['DataSet'] == 'Set 1']['ARI Score']
    group2_scores = df_filtered[df_filtered['DataSet'] == 'Set 2']['ARI Score']
    t_stat, p_value = ttest_ind(group1_scores, group2_scores, equal_var=False)  # Assumes unequal variances
    print(f"T-test for {algorithm_name}: t-statistic = {t_stat}, p-value = {p_value}")
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='DataSet', y='ARI Score', data=df_filtered)
    plt.title(f"ARI Scores Comparison for {algorithm_name}")
    plt.xlabel('Data Set')
    plt.ylabel('ARI Score')
    
    # Drawing the linked line and showing the p-value
    y, h, col = df_filtered['ARI Score'].max() + 0.05, 0.02, 'k'
    plt.plot([0, 0, 1, 1], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text(0.5, y+h, f"p = {p_value:.4f}", ha='center', va='bottom', color=col)
    
    plt.tight_layout()
    plt.show()

# plot_algorithm_comparison_with_ttest_and_line('clustering_results.csv', 'clustering_3_groups_results.csv', 'K-Means')



# ari_results_box_plots()

def main():

    parser = argparse.ArgumentParser(description ='LLM embedding clustering')
    parser.add_argument('--num', type=int, default=2, help ='the number of groups for the evaluation')
    
    args = parser.parse_args()
    bp_graph_path ='2024_biological_process_graph_w_verification.graphml'
    num_of_groups = args.num
    exp_candidates = filter_exp_dataset(num_of_groups=num_of_groups)

    results = []
    for roots in exp_candidates[:10]:
        print(f"Roots: {roots}.")
        # results.extend(eval_llm_clustering(bp_graph_path, roots)) 
        results.extend(eval_m_type_relation_clustering(bp_graph_path, roots))   

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'm-type_clustering_{num_of_groups}_groups_results.csv', index=False)

# read_m_type_file()
    
if __name__ == "__main__":
    main() 





        




