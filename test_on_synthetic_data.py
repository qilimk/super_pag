from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN

import networkx as nx

G = nx.davis_southern_women_graph()  # Example graph
communities = nx.community.greedy_modularity_communities(G)

# Compute positions for the node clusters as if they were themselves nodes in a
# supergraph using a larger scale factor
supergraph = nx.cycle_graph(len(communities))
superpos = nx.spring_layout(G, scale=50, seed=429)

# Use the "supernode" positions as the center of each node cluster
centers = list(superpos.values())
pos = {}
for center, comm in zip(centers, communities):
    pos.update(nx.spring_layout(nx.subgraph(G, comm), center=center, seed=1430))

# Nodes colored by cluster
for nodes, clr in zip(communities, ("tab:blue", "tab:orange", "tab:green")):
    nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_color=clr, node_size=100)
nx.draw_networkx_edges(G, pos=pos)

plt.tight_layout()
plt.savefig("test.png", format="PNG")
plt.show()
print("Test is finished!!")






# Dataset characteristics
# n_samples = 1000  # Total number of samples
# n_features = 2    # Number of features (for easy visualization)
# n_clusters = 4    # Number of clusters
# cluster_std = 2.0 # Standard deviation of clusters

# # Generate synthetic dataset
# X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=cluster_std, random_state=42)

# Visualize the dataset
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Synthetic Dataset Visualization')
# plt.show()



# K-means clustering

# kmeans = KMeans(n_clusters=4)
# kmeans.fit(X)
# y_kmeans = kmeans.predict(X)

# # Visualize the clusters
# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
# plt.title('K-Means Clustering')
# plt.show()


# # Hierarchical clustering

# Apply Hierarchical Clustering
# Z = linkage(X, 'ward')

# # Plot dendrogram
# plt.figure(figsize=(10, 7))
# dendrogram(Z)
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Data points')
# plt.ylabel('Euclidean distances')
# plt.show()

# # DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# # Generate synthetic data
# X, _ = make_moons(n_samples=200, noise=0.1, random_state=19)

# # Apply DBSCAN
# dbscan = DBSCAN(eps=0.3, min_samples=5)
# y_dbscan = dbscan.fit_predict(X)

# # Visualize the clusters
# plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis')
# plt.title('DBSCAN Clustering')
# plt.show()

# Mean Shift
# # Estimate bandwidth
# bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

# # Apply Mean Shift
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# ms.fit(X)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_

# # Visualize the clusters
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
# plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, alpha=0.5)
# plt.title('Mean Shift Clustering')
# plt.show()

