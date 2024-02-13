import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
from networkx.algorithms.community import girvan_newman
from sklearn.cluster import SpectralClustering
import numpy as np

# Parameters for the SBM
sizes = [30, 30, 30]  # 3 communities of size 30 each
p_within = 0.3  # Probability of connecting nodes within a community
p_between = 0.1  # Probability of connecting nodes between communities
prob_matrix = [[p_within, p_between, p_between], 
               [p_between, p_within, p_between], 
               [p_between, p_between, p_within]]

# Generate the synthetic graph
G = nx.stochastic_block_model(sizes, prob_matrix, seed=42)

# Original community labels for visualization
original_communities = {node: i//30 for i, node in enumerate(G.nodes())}

partition_louvain = community_louvain.best_partition(G)

# Taking only the first level of division for simplicity
gn_comm = next(girvan_newman(G))
partition_gn = {}
for i, comm in enumerate(gn_comm):
    for node in comm:
        partition_gn[node] = i

# Convert to adjacency matrix
adj_matrix = nx.to_numpy_array(G)
# Apply spectral clustering
sc = SpectralClustering(n_clusters=3, affinity='precomputed', n_init=100, random_state=42)
sc.fit(adj_matrix)
partition_sc = {i: sc.labels_[i] for i in range(len(G.nodes()))}

def plot_communities(G, partition, title=""):
    cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{title.replace(' ', '_')}_results.jpg")
    # plt.show()

# Visualize the original and detected communities
plot_communities(G, original_communities, "Original Communities")
plot_communities(G, partition_louvain, "Louvain Method")
plot_communities(G, partition_gn, "Girvan-Newman Algorithm")
plot_communities(G, partition_sc, "Spectral Clustering")
print("Finished!!!")
