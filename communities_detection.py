# import matplotlib.pyplot as plt
# import networkx as nx
# import community as community_louvain
# from networkx.algorithms.community import girvan_newman
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# import numpy as np


# def plot_communities(G, partition, title=""):
#     cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)
#     plt.figure(figsize=(8, 8))
#     pos = nx.spring_layout(G, seed=42)
#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
#                            cmap=cmap, node_color=list(partition.values()))
#     nx.draw_networkx_edges(G, pos, alpha=0.5)
#     plt.title(title)
#     plt.axis('off')
#     plt.savefig(f"{title.replace(' ', '_')}_results.jpg")
#     # plt.show()

# # Evaluation Metrics
# def evaluate(true_labels, pred_labels):
#     ari = adjusted_rand_score(true_labels, pred_labels)
#     nmi = normalized_mutual_info_score(true_labels, pred_labels)
#     return ari, nmi



# # Parameters for the SBM
# sizes = [50, 50]  # 3 communities of size 30 each
# p_within = 0.3  # Probability of connecting nodes within a community
# p_between = 0.1  # Probability of connecting nodes between communities
# # prob_matrix = [[p_within, p_between, p_between], 
# #                [p_between, p_within, p_between], 
# #                [p_between, p_between, p_within]]

# prob_matrix = [[p_within, p_between], 
#                [p_between, p_within]]

# # Generate the synthetic graph
# G = nx.stochastic_block_model(sizes, prob_matrix, seed=42)

# # Original community labels for visualization
# original_communities = {node: i//50 for i, node in enumerate(G.nodes())}

# # Ground truth labels
# true_labels = [i // 50 for i in range(sum(sizes))]  


# partition_louvain = community_louvain.best_partition(G)
# louvain_labels = [partition_louvain[node] for node in G.nodes()]

# # Taking only the first level of division for simplicity
# gn_comm = next(girvan_newman(G))
# partition_gn = {}
# for i, comm in enumerate(gn_comm):
#     for node in comm:
#         partition_gn[node] = i
#         gn_labels = [0] * len(G.nodes())

# # Initialize labels
# gn_labels = [0] * len(G.nodes())  
# for i, comm in enumerate(gn_comm):
#     for node in comm:
#         gn_labels[node] = i 

# # Convert to adjacency matrix
# adj_matrix = nx.to_numpy_array(G)
# # Apply spectral clustering
# sc = SpectralClustering(n_clusters=2, affinity='precomputed', n_init=100, random_state=42)
# sc.fit(adj_matrix)
# partition_sc = {i: sc.labels_[i] for i in range(len(G.nodes()))}
# sc_labels = sc.labels_




# ari_louvain, nmi_louvain = evaluate(true_labels, louvain_labels)
# ari_gn, nmi_gn = evaluate(true_labels, gn_labels)
# ari_sc, nmi_sc = evaluate(true_labels, sc_labels)

# print(f"Louvain Method: ARI = {ari_louvain}, NMI = {nmi_louvain}")
# print(f"Girvan-Newman Algorithm: ARI = {ari_gn}, NMI = {nmi_gn}")
# print(f"Spectral Clustering: ARI = {ari_sc}, NMI = {nmi_sc}")



# # Visualize the original and detected communities
# plot_communities(G, original_communities, "Original Communities")
# plot_communities(G, partition_louvain, "Louvain Method")
# plot_communities(G, partition_gn, "Girvan-Newman Algorithm")
# plot_communities(G, partition_sc, "Spectral Clustering")
# print("Finished!!!")


import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
from networkx.algorithms.community import girvan_newman
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

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

def evaluate(true_labels, pred_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return ari, nmi

def main(algorithm):
    # Parameters for the SBM
    sizes = [50, 50]
    p_within = 0.3
    p_between = 0.1
    prob_matrix = [[p_within, p_between], [p_between, p_within]]
    G = nx.stochastic_block_model(sizes, prob_matrix, seed=42)
    true_labels = [i // 50 for i in range(sum(sizes))]
    
    if algorithm == 'louvain':
        partition = community_louvain.best_partition(G)
        title = "Louvain Method"
    elif algorithm == 'girvan_newman':
        gn_comm = next(girvan_newman(G))
        partition = {node: i for i, comm in enumerate(gn_comm) for node in comm}
        title = "Girvan-Newman Algorithm"
    elif algorithm == 'spectral_clustering':
        adj_matrix = nx.to_numpy_array(G)
        sc = SpectralClustering(n_clusters=2, affinity='precomputed', n_init=100, random_state=42)
        sc.fit(adj_matrix)
        partition = {i: sc.labels_[i] for i in range(len(G.nodes()))}
        title = "Spectral Clustering"
    else:
        raise ValueError("Invalid algorithm")
    
    labels = [partition[node] for node in G.nodes()]
    ari, nmi = evaluate(true_labels, labels)
    print(f"{title}: ARI = {ari}, NMI = {nmi}")
    
    plot_communities(G, partition, title)

# Example usage:
main('louvain')
main('girvan_newman')
main('spectral_clustering')
