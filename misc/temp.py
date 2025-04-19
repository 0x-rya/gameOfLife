import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class EdgeAttr:

    def __init__(self, delay: int, node1: tuple[int, int], lkv1: int, node2: tuple[int, int], lkv2: int) -> None:
        
        self.delay = delay
        self.last_known_values = {
            node1: lkv1,
            node2: lkv2
        }

    def __str__(self) -> str:
        
        return str((self.delay, str(self.last_known_values)))

    def getLkv(self, node: tuple[int, int]) -> int:

        result = self.last_known_values.get(node, None)

        if result != None:
            return result

        raise Exception(f"Invalid Node {node} for the edge with valid nodes {self.last_known_values}")

def genInitConfig(seed, width, height, density: float) -> list[list[int]]:

    random.seed(seed)
    size = width * height
    if not 0 <= density <= 1:
        raise ValueError("Density must be between 0 and 1.")

    num_ones = int(size * density)
    arr = [1] * num_ones + [0] * (size - num_ones)
    random.shuffle(arr)
    arr = np.array(arr).reshape((height, width))

    return arr.tolist()

def create_graph_from_grid(grid):
    rows, cols = len(grid), len(grid[0])
    G = nx.Graph()
    
    # Add nodes
    for r in range(rows):
        for c in range(cols):
            G.add_node((r, c), value=grid[r][c])
    
    # Define 8 possible moves (including diagonals)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),         (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    # Add edges based on 8-neighbor connectivity
    for r in range(rows):
        for c in range(cols):
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    G.add_edge((r, c), (nr, nc), weight=EdgeAttr(random.randint(1, 4), (r, c), G.nodes[(r, c)]["value"], (nr, nc), G.nodes[(nr, nc)]["value"]))
    
    return G

# Example grid
grid = genInitConfig(5000, 5, 5, 0.5)

G = create_graph_from_grid(grid)

def draw_graph(G):
    pos = {(r, c): (c, -r) for r, c in G.nodes()}  # Position nodes according to grid layout
    
    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500)
    
    # Draw edge labels (weights)
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

# Draw the graph
pos = {(r, c): (c, -r) for r, c in G.nodes()}  # Position nodes according to their grid coordinates
draw_graph(G)
plt.show()

