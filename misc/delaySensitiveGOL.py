import random
import numpy as np
import networkx as nx
from pathlib import Path
from collections import abc
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib.animation as animation

class NodeAttr:

    def __init__(self, max_delay: int, value: int) -> None:
        
        self.max_delay = max_delay
        self.prev_values = [value] * max_delay

    def __repr__(self):
        
        return str((self.max_delay, self.prev_values))

    def add_new_value(self, value: int) -> None:

        if len(self.prev_values) >= self.max_delay:
            self.prev_values.pop(0)
        self.prev_values.append(value)

    def get_curr_value(self) -> int:

        return self.prev_values[-1]

    def get_delayed_value(self, delay: int) -> int:

        return self.prev_values[-delay]

class EdgeAttr:

    def __init__(self, delay: int, node1: tuple[int, int], lkv1: int, node2: tuple[int, int], lkv2: int) -> None:
        
        self.delay = delay
        self.last_known_values = {
            node1: lkv1,
            node2: lkv2
        }

    def __str__(self) -> str:
        
        return str(self.delay) + "\n" + str(self.last_known_values)

    def getLkv(self, node: tuple[int, int]) -> int:

        result = self.last_known_values.get(node, None)

        if result != None:
            return result

        raise Exception(f"Invalid Node {node} for the edge with valid nodes {self.last_known_values}")

    def getNeighbourValue(self, graph: nx.Graph, neighbourNode: tuple[int, int], tau: float) -> int:

        if random.random() <= tau:
            return self.getLkv(neighbourNode)
        else:
            cell: NodeAttr = graph.nodes[neighbourNode]["value"]
            return cell.get_delayed_value(self.delay)

class delaySensitiveGOL():

    def __init__(
            self,
            sizeXY: tuple[int, int],
            timeStamps: int,
            initConfig: str | None,
            rule: str,
            seed: int,
            density: float,
            alpha: float,               ## what fraction of the population gets updated
            delay: int,                 ## maximum possible delay between two neighbours
            tau: float,                 ## probabilistic loss of information
            averageTimeStamp: int,
            extremes: tuple[int, int]
        ) -> None:

        if not isinstance(sizeXY, abc.Iterable):
            raise Exception(f"Size has to be an iterable and not of type {type(sizeXY)}")
        
        if len(sizeXY) != 2:
            raise Exception("Size can only be of length 2")

        self.imgs: list[list[list[int]]] = []
        self.width = sizeXY[0]
        self.height = sizeXY[1]
        self.seed = seed
        self.alpha = alpha
        self.delay = delay
        self.tau = tau
        self.avts = averageTimeStamp
        self.timeStamps = timeStamps
        if not self.validateRule(rule):
            raise Exception("Invalid Rule.\nRule should be of format Bl,m,n,.../Sl,n,m\nExample: B3/S2,3...")

        self.density = []
        self.extremes = extremes

        if initConfig != None:
            self.automata = self.readInitConfig(initConfig)

        else:
            self.automata = self.genInitConfig(density)

        self.viewState = self.create_graph_from_grid()

    def create_graph_from_grid(self):

        grid = self.automata
        rows, cols = len(grid), len(grid[0])
        G = nx.Graph()
        
        # Add nodes
        for r in range(rows):
            for c in range(cols):
                G.add_node((r, c), value=NodeAttr(self.delay, grid[r][c]))
        
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
                        G.add_edge((r, c), (nr, nc), weight=EdgeAttr(random.randint(1, self.delay), (r, c), G.nodes[(r, c)]["value"].get_curr_value(), (nr, nc), G.nodes[(nr, nc)]["value"].get_curr_value()))
        
        return G

    def draw_graph(self, G):
        pos = {(r, c): (c, -r) for r, c in G.nodes()}  # Position nodes according to grid layout
        
        # Draw nodes and edges
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500)
        
        # Draw edge labels (weights)
        edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    def readInitConfig(self, initConfig: str) -> list[list[int]]:
        
        try:
            fp = open(initConfig, "r")
            grid = fp.readlines()
            grid = [row.strip("\n") for row in grid]

            if len(grid) != self.height:
                raise Exception(f"Input file has height of {len(grid)} but expected to be {self.height}")

            finalCols = []
            for rows in grid:
                finalRows = []
                
                cols = rows.split(",")

                if len(cols) != self.width:
                    raise Exception(f"Input file has width of {len(cols)} but expected to be {self.width}")

                for val in cols:

                    if int(val) != 0 and int(val) != 1:
                        raise Exception("initConfig file can only contain 0s and 1s")

                    finalRows.append(int(val))
                
                finalCols.append(finalRows)
            
            return finalCols

        except FileNotFoundError:
            raise Exception("Could not find the initConfig file")
        
        except TypeError:
            raise Exception("File contains non-numeric input")

    def genInitConfig(self, density: float) -> list[list[int]]:

        random.seed(self.seed)
        size = self.width * self.height
        if not 0 <= density <= 1:
            raise ValueError("Density must be between 0 and 1.")

        num_ones = int(size * density)
        arr = [1] * num_ones + [0] * (size - num_ones)
        random.shuffle(arr)
        arr = np.array(arr).reshape((self.height, self.width))

        return arr.tolist()

    def validateRule(self, rule: str) -> bool:

        self.rulesDict = {
                
                ## traditional life game
                "life": "B3/S2,3",
                
                ## low-density life-like games
                "flock_life": "B3/S1,2",
                "honey_life": "B3,8/S2,3,8",
                "2x2_life": "B3,6/S1,2,5",
                "eight_life": "B3/S2,3,8",
                "high_life": "B3,6/S2,3",
                "pedestrian_life": "B3,8/S2,3",
                "lowdeath_life": "B3,6,8/S2,3,8",
                
                ## high-density life-like games
                "dry_life": "B3,7/S2,3",
                "drigh_life": "B3,6,7/S2,3",
                "B356/S23": "B3,5,6/S2,3",
                "B356/S238": "B3,5,6/S2,3,8",
                "B3568/S23": "B3,5,6,8/S2,3",
                "B3568/S238": "B3,5,6,8/S2,3,8",
                "B3578/S23": "B3,5,7,8/S2,3",
                "B3578/S237": "B3,5,7,8/S2,3,7",
                "B3578/S238": "B3,5,7,8/S2,3,8"
            }

        try:
            birth, survival = rule.split("/")
            
            birth = birth.strip("B").split(",")
            survival = survival.strip("S").split(",")

            birth = [int(x) for x in birth]
            survival = [int(x) for x in survival]

            self.rule = rule
            self.birth = birth
            self.survival = survival
            return True

        except:
            if rule in self.rulesDict.keys():
                return self.validateRule(self.rulesDict[rule])
            else:
                return False

    def getNeighbours(self, x: int, y: int) -> list[int]:
        
        # if x == 0 and y == 0:

        #     return [self.automata[y + 1][x], self.automata[y][x + 1], self.automata[y + 1][x + 1]]

        # elif x == 0 and y == self.height - 1:

        #     return [self.automata[y - 1][x], self.automata[y][x + 1], self.automata[y - 1][x + 1]]

        # elif x == self.width - 1 and y == 0:

        #     return [self.automata[y][x - 1], self.automata[y + 1][x], self.automata[y + 1][x - 1]]

        # elif x == self.width - 1 and y == self.height - 1:

        #     return [self.automata[y - 1][x], self.automata[y][x - 1], self.automata[y - 1][x - 1]]

        # elif x == 0 and 0 < y < self.height - 1:

        #     return [self.automata[y + 1][x], self.automata[y - 1][x], self.automata[y - 1][x + 1], self.automata[y][x + 1], self.automata[y + 1][x + 1]]

        # elif x == self.width - 1 and 0 < y < self.height - 1:

        #     return [self.automata[y + 1][x], self.automata[y - 1][x], self.automata[y - 1][x - 1], self.automata[y][x - 1], self.automata[y + 1][x - 1]]

        # elif y == 0 and 0 < x < self.width - 1:

        #     return [self.automata[y][x - 1], self.automata[y][x + 1], self.automata[y + 1][x - 1], self.automata[y + 1][x], self.automata[y + 1][x + 1]]

        # elif y == self.height - 1 and 0 < x < self.width - 1:

        #     return [self.automata[y][x - 1], self.automata[y][x + 1], self.automata[y - 1][x - 1], self.automata[y - 1][x], self.automata[y - 1][x + 1]]

        # else:

        #     return [self.automata[y - 1][x - 1], self.automata[y - 1][x], self.automata[y - 1][x + 1], self.automata[y][x - 1],
        #             self.automata[y + 1][x - 1], self.automata[y + 1][x], self.automata[y + 1][x + 1], self.automata[y][x + 1]]

        grid = self.automata
        rows, cols = len(grid), len(grid[0])

        # Define 8 possible moves (including diagonals)
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),         (0, 1),
                      (1, -1), (1, 0), (1, 1)]

        # Add edges based on 8-neighbor connectivity
        neighbours = []
        for dr, dc in directions:
            nr, nc = x + dr, y + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                vs: EdgeAttr = self.viewState[(y, x)][(nc, nr)]["weight"]
                neighbours.append(vs.getNeighbourValue(self.viewState, (nc, nr), self.tau))
        
        return neighbours

    def updateAutomata(self) -> None:

        nextIter = np.zeros(shape=(self.height, self.width))
        newValues = {}
        for w in range(self.width):
            for h in range(self.height):
                if random.random() <= self.alpha:
                    nextIter[h][w] = self.getNextIter(w, h)
                else:
                    nextIter[h][w] = self.automata[h][w]
                newValues[(w, h)] = nextIter[h][w]
                # self.viewState.nodes[(w, h)]["value"].add_new_value(nextIter[h][w])

        self.automata = nextIter.tolist()
        for key, value in newValues.items():
            self.viewState.nodes[key]["value"].add_new_value(value)

    def calcDensity(self) -> float:

        ones = 0
        for row in self.automata:
            ones += sum(row)

        density = ones/(self.height * self.width)
        return density
    
    def record(self, save: bool | str) -> None:

        frames = []
        fig = plt.figure()
        for i in range(len(self.imgs)):

            frames.append([plt.imshow(self.imgs[i], animated=True, cmap='Greys')])
        
        ani = animation.ArtistAnimation(fig, frames, interval=800, blit=True)
        plt.show()
        if isinstance(save, str):
            ani.save(f"outputs/{save}/game.mp4")

    def getNextIter(self, x: int, y: int) -> int:

        aliveNeighbours = sum(self.getNeighbours(x, y))
        if self.automata[y][x] == 0:
            ## will be 1 only if exactly three neighbours are alive
            if aliveNeighbours in self.birth:
                return 1
            else:
                return 0
        else:
            ## will be 1 if 2-3 neighbours are alive
            if aliveNeighbours in self.survival:
                return 1
            else:
                return 0

    def simulate(self, vis: bool, save: bool | str):

        self.imgs.append(self.automata)
        self.density.append(self.calcDensity())
        for ts in range(self.timeStamps):
            self.updateAutomata()
            # if ts >= self.avts:
            self.density.append(self.calcDensity())
            self.imgs.append(self.automata)
        if isinstance(save, str):
            Path(f"outputs/{save}/").mkdir(parents=True, exist_ok=True)     ## make sure the directory exists
            self.save_seed(save)                                            ## save seed for reproducibility's sake
            self.save_density(save)
        if vis:
            self.record(save=save)

    def getAvgDensity(self) -> float:

        return sum(self.density[:self.avts]) / len(self.density[:self.avts])

    def save_seed(self, save: str) -> None:
        
        with open(f"outputs/{save}/seed.txt", "w") as fd:

            fd.write(str(self.seed))

    def save_density(self, save: str) -> None:

        ## plot the density in a graph
        plt.plot(self.density)
        plt.savefig(f"outputs/{save}/density.png")

if __name__ == "__main__":
    
    parser = ArgumentParser(prog="alpha-Asynchronous Game of Life", epilog="Either provide an initconfig file or density along with a rule.")
    
    DEFAULTS = {
            "WIDTH": 50,
            "HEIGHT": 50,
            "AVTS": 1000,
            "TIMESTAMPS": 1500,
            "INITCONFIG": None,
            "RULE": "B3/S2,3",
            "DENSITY": 0.5,
            "ALPHA": 1,
            "DELAY": 1,
            "TAU": 0,
            "EXTREMES": "0,1",
            "SAVEFILE": False,
            "SEED": None
        }

    parser.add_argument("-D", "--defaults", action="store_true")
    parser.add_argument("-W", "--width", default=DEFAULTS["WIDTH"], type=int)
    parser.add_argument("-H", "--height", default=DEFAULTS["HEIGHT"], type=int)
    parser.add_argument("-A", "--average-timestamp", default=DEFAULTS["AVTS"], type=int)
    parser.add_argument("-t", "--timestamps", default=DEFAULTS["TIMESTAMPS"], type=int)
    parser.add_argument("-i", "--initconfig", default=DEFAULTS["INITCONFIG"], type=str)
    parser.add_argument("-r", "--rule", default=DEFAULTS["RULE"], type=str)
    
    parser.add_argument("-d", "--density", default=DEFAULTS["DENSITY"], type=float)
    parser.add_argument("-a", "--alpha", default=DEFAULTS["ALPHA"], type=float)
    parser.add_argument("-l", "--delay", default=DEFAULTS["DELAY"], type=int)
    parser.add_argument("-T", "--tau", default=DEFAULTS["TAU"], type=float)
    parser.add_argument("-e", "--extremes", default=DEFAULTS["EXTREMES"], type=str, help="To pass only one extreme: `,upper_bound` or `lower_bound,`\nExample: `,.9889` ; `.2243,`")

    parser.add_argument("-s", "--seed", default=DEFAULTS["SEED"], type=int)
    parser.add_argument("-V", "--visualize", action="store_true")
    parser.add_argument("-S", "--save", default=DEFAULTS["SAVEFILE"], type=str)

    args = parser.parse_args()

    ## --------------- HELPER FUNCTIONS ----------------- ##

    def print_defaults() -> None:
        
        for k, v in DEFAULTS.items():
            print(k, ":", v)

    def generate_seed() -> int:

        return int(random.random() * 10000)

    def parse_extremes() -> tuple[int, int]:

        exts = args.extremes.split(",")
        retExts = []
        if len(exts) != 2:
            raise Exception("Extremes must be passed as a string of two values separated by `,`.\n\nDefault: `0,1`")

        try:
            retExts.append(float(exts[0]))
        except:
            if exts[0] == "":
                retExts.append(0)
            else:
                raise Exception("Extremes must be passed as a string of two floats separated by `,`.\nDefault: `0,1`")

        try:
            retExts.append(float(exts[1]))
        except:
            if exts[1] == "":
                retExts.append(1)
            else:
                raise Exception("Extremes must be passed as a string of two floats separated by `,`.\nDefault: `0,1`")

        return tuple(retExts)

    ## --------------- SETUP CONDITIONS ----------------- ##

    if args.defaults:
        print_defaults()
        exit(0)

    if args.seed == None:
        args.seed = generate_seed()

    if args.average_timestamp > args.timestamps:
        raise Exception("The timestamp to start counting the average from has to be smaller than the total number of timestamps")

    ## --------------- START FROM HERE! ----------------- ##

    conway = delaySensitiveGOL((args.width, args.height),
                        args.timestamps, args.initconfig, args.rule, args.seed,
                        args.density,
                        args.alpha, args.delay, args.tau, args.average_timestamp, parse_extremes())
    conway.simulate(vis=args.visualize, save=args.save)
    
    ## average density
    print(conway.density)
    print(sum(conway.density)/len(conway.density))
