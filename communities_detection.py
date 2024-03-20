import argparse
import textwrap
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
from networkx.algorithms.community import girvan_newman
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ALGORITHMS = ['louvain', 'girvan_newman', 'spectral_clustering']

def plot_communities_w_names(G, partition, title=""):
    cmap = plt.get_cmap('Pastel1')  # Corrected method to get colormap
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Draw node labels
    labels = {node: str(node) for node in G.nodes()}  # Create a label for each node (using the node itself as the label)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)  # Adjust font_size as needed
    
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{title.replace(' ', '_')}_results.jpg", dpi=300)  # Adjusted DPI for better image quality

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

def evaluate_algorithm(true_labels, pred_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return ari, nmi


def generate_sbm_graph(sizes, p_within, p_between):
    # Dynamically create the probability matrix for connections
    prob_matrix = [[p_within if i == j else p_between for j in range(len(sizes))] for i in range(len(sizes))]
    # Generate the graph
    G = nx.stochastic_block_model(sizes, prob_matrix, seed=323)
    true_labels = []
    for label_index, size in enumerate(sizes):
        true_labels.extend([label_index] * size)
    return G, true_labels


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
        sc = SpectralClustering(n_clusters=2, affinity='precomputed', n_init=100, random_state=42)
        sc.fit(adj_matrix)
        partition = {i: sc.labels_[i] for i in range(len(G.nodes()))}
        title = "Spectral Clustering"
    else:
        raise ValueError("Invalid algorithm")
    
    # print(partition)
    
    labels = [partition[node] for node in G.nodes()]
    ari, nmi = evaluate_algorithm(true_labels, labels)

    if is_plot:
        print(f"{title}: ARI = {ari}, NMI = {nmi}")
        plot_communities(G, partition, title)

    return ari, nmi

# Function to plot heatmap for given metric ('ARI' or 'NMI')
def plot_heatmap_for_metric(metric, results_df, prefix = "sbm_50_50"):
    for alg in ALGORITHMS:
        # Filter DataFrame for the current algorithm
        alg_data = results_df[results_df['Algorithm'] == alg]
        
        # Pivot the data to get a matrix where rows are p_within, columns are p_between, and values are the metric
        pivot_table = alg_data.pivot("p_within", "p_between", metric)
        
        # Plotting
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"{alg.capitalize()} - {metric}")
        plt.ylabel('p_within')
        plt.xlabel('p_between')
        plt.savefig(f"{prefix}_{alg}_{metric}_heatmap.jpg")
        plt.show()

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


def eval_cross_all_for_3_methods(args):
    exp = f"sbm_{'_'.join(str(item) for item in args.sizes)}"
    save_csv_name = f'{exp}_clustering_results.csv'

    results = []
    for alg in ALGORITHMS:
        for i in range(10):
            p_within = (i+1) / 10
            for j in range(i+1):
                p_between= (j+1) / 10
                G, true_labels = generate_sbm_graph(args.sizes, p_within, p_between)
                print(f"{alg}: {p_within}, {p_between}")
                ari, nmi = run_eval(alg, G, true_labels)
                # Append results to the DataFrame
                results.append({'Algorithm': alg, 'p_within': p_within, 'p_between': p_between, 'ARI': ari, 'NMI': nmi})
    

    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_csv_name, index=False)
    return exp,save_csv_name

def plot_heatmaps(clustering_result_file, exp):
    results_df = pd.read_csv(clustering_result_file)
    for metric in ['ARI', 'NMI']:
        plot_heatmap_for_metric(metric, results_df, exp)


def generate_all_sbm_graphs(sizes=None):
    if sizes is None:
        sizes = [50, 50]

    exp = f"sbm_{'_'.join(str(item) for item in sizes)}"
    save_csv_name = f'{exp}_distribution_of_connection_results.csv'

    community_dict = {}
    columns = ['p_within', 'p_between', 'node', 'connection']
    connection_df = pd.DataFrame(columns=columns)

    for i in range(10):
        p_within = (i+1) / 10
        for j in range(i+1):
            p_between= (j+1) / 10
            G, true_labels = generate_sbm_graph(sizes, p_within, p_between)

            # Generate community labels based on sizes
            start = 0
            for label, size in enumerate(sizes):
                for node in range(start, start + size):
                    community_dict[node] = label
                start += size

            for node_id in range(G.number_of_nodes()):
                # Choose a target vertex
                target_vertex = int(node_id)

                # Find the existing edges among the neighbors of the target vertex
                edges, neighbors = find_edges_among_neighbors(G, target_vertex)
                number_of_neighbors = len(neighbors)
                result = {'p_within': p_within, 'p_between': p_between, 'node': target_vertex, 'connection': len(edges) *2 / (number_of_neighbors*(number_of_neighbors-1))}
                connection_df = connection_df.append(result, ignore_index=True)

                print(f"Edges among neighbors of vertex {target_vertex}: {edges}")

            plot_communities_w_names(G, community_dict, f"{exp}_p_within_{p_within}_p_between_{p_between}")

    connection_df.to_csv(save_csv_name, index=False)

def plot_all_connection_dist(df, exp):

    desired_width_px = 1200
    desired_height_px = 800
    dpi = 100  # Desired DPI

    # Calculate figure size in inches
    figsize_inches = (desired_width_px / dpi, desired_height_px / dpi)

    df['node'] = df['node'].astype(int)
    for i in range(10):
        p_within = (i+1) / 10
        for j in range(i+1):
            p_between= (j+1) / 10
            
            filtered_df = df[(df['p_within'] == p_within) & (df['p_between'] == p_between)][['node', 'connection']]
            plt.figure(figsize=figsize_inches)
            sns.displot(filtered_df, x="connection", kind="kde", fill=True)
            # sns.barplot(x="node", y="connection", data=filtered_df, palette="viridis")
            plt.xlim(0, 1)
            title_text = f"Distribution of Connection Values for p_within={p_within} and p_between={p_between}"
            wrapped_title = textwrap.fill(title_text, width=50)
            plt.title(wrapped_title)
            plt.xlabel("Connection Strength")
            plt.ylabel("Density")
            plt.tight_layout()
            plt.savefig(f"{exp}_p_within_{p_within}_p_between_{p_between}_distribution.jpg", dpi=dpi)
            plt.close()

def main():

    parser = argparse.ArgumentParser(description ='Community detection!!')
    parser.add_argument('--alg', type=str, default="louvain", help ='the algorithm to be evaluated.')
    parser.add_argument('--sizes', nargs='+', type=int, help='List of sizes (integers).')
    parser.add_argument('--p_within', type=int, default=0.5, help ='the probability of edge creation within a cluster.')
    parser.add_argument('--p_between', type=int, default=0.1, help ='the probability for edge creation between the clusters. ')
    args = parser.parse_args()

    # multiple evaluations 
    print(args.sizes)
    # exp, save_csv_name = eval_cross_all_for_3_methods(args)
    # plot_heatmaps(save_csv_name, exp)

    exp = f"sbm_{'_'.join(str(item) for item in args.sizes)}"
    save_csv_name = f'{exp}_distribution_of_connection_results.csv'
    
    generate_all_sbm_graphs(args.sizes)
    df = pd.read_csv(save_csv_name)
    plot_all_connection_dist(df, exp)



if __name__ == "__main__":

    main() 

# Example usage:
# python communities_detection.py --sizes 50 50 --p_within 0.5 --p_between
