class WeightedGraph:
    def __init__(self):
        self.graph = {}

    def node_exists(self, node):
        """Check if a node exists in the graph."""
        return node in self.graph

    def edge_exists(self, node1, node2):
        """Check if an edge exists between node1 and node2."""
        return self.node_exists(node1) and node2 in self.graph[node1]

    def add_node(self, node):
        """Add a node to the graph."""
        if not self.node_exists(node):
            self.graph[node] = {}

    def add_edge(self, node1, node2, weight):
        """Add an edge between node1 and node2 with the given weight."""
        if not self.edge_exists(node1, node2):
            # If the nodes don't exist, add them to the graph
            self.add_node(node1)
            self.add_node(node2)

            # Add node2 as a neighbor of node1 and vice versa
            self.graph[node1][node2] = weight
            # self.graph[node2][node1] = weight  # Assuming it's an undirected graph

    def __str__(self):
        return str(self.graph)