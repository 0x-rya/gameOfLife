import random
import numpy as np
from pathlib import Path
from collections import abc
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib.animation as animation

class GameOfLife:

    def __init__(
            self,
            sizeXY: tuple[int, int],
            timeStamps: int,
            initConfig: str | None,
            rule: str,
            seed: int,
            density: float,
            averageTimeStamp: int,
            extremes: tuple[int, int],   ## what value in replacement of 0, 1 is to be considered for phase transition
            maxDelay: int = 1,           ## New parameter for maximum delay
            tau: float = 0.0             ## New parameter for probabilistic information loss
        ) -> None:
        
        if not isinstance(sizeXY, abc.Iterable):
            raise Exception(f"Size has to be an iterable and not of type {type(sizeXY)}")
        
        if len(sizeXY) != 2:
            raise Exception("Size can only be of length 2")

        # Validate tau (information loss probability)
        if not 0 <= tau <= 1:
            raise ValueError("Tau (information loss probability) must be between 0 and 1")
        
        self.imgs: list[list[list[int]]] = []
        self.width = sizeXY[0]
        self.height = sizeXY[1]
        self.seed = seed
        self.avts = averageTimeStamp
        self.timeStamps = timeStamps
        self.maxDelay = maxDelay
        self.tau = tau  # Information loss probability
        
        if not self.validateRule(rule):
            raise Exception("Invalid Rule.\nRule should be of format Bl,m,n,.../Sl,n,m\nExample: B3/S2,3...")

        self.density = []
        self.extremes = extremes

        # Initialize the random number generator with the provided seed
        random.seed(self.seed)
        
        # Create a history buffer to store past states for delayed interactions
        # We need to store states up to maxDelay steps in the past
        self.history = []
        
        # Initialize the delay matrix - this stores the delay between each pair of cells
        self.initializeDelayMatrix()
        
        # Initialize the last known state matrix - for information loss compensation
        self.last_known_states = {}
        
        if initConfig != None:
            self.automata = self.readInitConfig(initConfig)
        else:
            self.automata = self.genInitConfig(density)
        
        # Initialize history with initial state
        self.history = [self.automata.copy() for _ in range(maxDelay)]
        
        # Initialize last known states with initial state
        self.initializeLastKnownStates()

    def initializeLastKnownStates(self) -> None:
        """Initialize the last known states for every cell's neighbors"""
        for y in range(self.height):
            for x in range(self.width):
                neighbors = self.getNeighborCoordinates(x, y)
                for nx, ny in neighbors:
                    # Key as ((y, x), (ny, nx)) to identify the relationship between cells
                    self.last_known_states[((y, x), (ny, nx))] = self.automata[ny][nx]

    def initializeDelayMatrix(self) -> None:
        """Initialize the delay matrix for neighbor interactions"""
        # Create a dictionary to store delays between each pair of cells
        # delay_matrix[((y1, x1), (y2, x2))] = delay between cell (x1,y1) and cell (x2,y2)
        
        self.delay_matrix = {}
        
        # For each cell, set delay with its neighbors
        for y in range(self.height):
            for x in range(self.width):
                neighbors = self.getNeighborCoordinates(x, y)
                for nx, ny in neighbors:
                    # Only process each pair once
                    pair_key = ((y, x), (ny, nx))
                    reverse_key = ((ny, nx), (y, x))
                    
                    if pair_key not in self.delay_matrix and reverse_key not in self.delay_matrix:
                        # Random delay between 1 and maxDelay (inclusive)
                        delay = random.randint(1, self.maxDelay)
                        # Store the delay for both directions (must be the same)
                        self.delay_matrix[pair_key] = delay
                        self.delay_matrix[reverse_key] = delay

    def getNeighborCoordinates(self, x: int, y: int) -> list[tuple[int, int]]:
        """Get the coordinates of neighboring cells"""
        neighbors = []
        
        # Check for the 8 possible neighbors, considering edge cases
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # Skip the cell itself
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                
                # Check if the neighbor is within the grid boundaries
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        
        return neighbors

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

    def getDelayedNeighbourStates(self, x: int, y: int) -> list[int]:
        """Get the states of neighbors with appropriate delay and probabilistic information loss"""
        delayedStates = []
        
        neighbors = self.getNeighborCoordinates(x, y)
        
        for nx, ny in neighbors:
            # Define the cell relationship
            pair_key = ((y, x), (ny, nx))
            
            # Get the delay between this cell and its neighbor
            delay = self.delay_matrix[pair_key]
            
            # Determine if information loss occurs for this interaction
            information_loss = random.random() < self.tau
            
            if information_loss:
                # Use the last known state if information is lost
                delayedStates.append(self.last_known_states[pair_key])
            else:
                # Otherwise, get the neighbor's state from the appropriate time in history
                historyIndex = min(delay - 1, len(self.history) - 1)
                
                if historyIndex < len(self.history):
                    current_state = self.history[historyIndex][ny][nx]
                else:
                    # If we don't have enough history yet, use the oldest state we have
                    current_state = self.history[-1][ny][nx]
                
                # Update the last known state for this neighbor
                self.last_known_states[pair_key] = current_state
                delayedStates.append(current_state)
        
        return delayedStates

    def updateAutomata(self) -> None:
        # Calculate the next state based on current state and history
        nextIter = np.zeros(shape=(self.height, self.width))
        for w in range(self.width):
            for h in range(self.height):
                nextIter[h][w] = self.getNextIter(w, h)
        
        # Update the history buffer - push the current state into history
        self.history.insert(0, self.automata.copy())
        if len(self.history) > self.maxDelay:
            self.history.pop()
            
        # Set the current state to the calculated next state
        self.automata = nextIter.tolist()

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
        # Get states of neighbors with their respective delays and potential information loss
        aliveNeighbours = sum(self.getDelayedNeighbourStates(x, y))
        
        if self.automata[y][x] == 0:
            # Will be 1 only if the right number of neighbors are alive
            if aliveNeighbours in self.birth:
                return 1
            else:
                return 0
        else:
            # Will be 1 if the right number of neighbors are alive
            if aliveNeighbours in self.survival:
                return 1
            else:
                return 0

    def simulate(self, vis: bool, save: bool | str):
        self.imgs.append(self.automata.copy())
        self.density.append(self.calcDensity())
        
        for ts in range(self.timeStamps):
            self.updateAutomata()
            self.density.append(self.calcDensity())
            self.imgs.append(self.automata.copy())
            
        if isinstance(save, str):
            Path(f"outputs/{save}/").mkdir(parents=True, exist_ok=True)
            self.save_seed(save)
            self.save_delay_matrix(save)
            self.save_parameters(save)
            self.save_density(save)
            
        if vis:
            self.record(save=save)

    def getAvgDensity(self) -> float:
        return sum(self.density[:self.avts]) / len(self.density[:self.avts])

    def save_seed(self, save: str) -> None:
        with open(f"outputs/{save}/seed.txt", "w") as fd:
            fd.write(str(self.seed))
            
    def save_delay_matrix(self, save: str) -> None:
        """Save the delay matrix for reproducibility"""
        with open(f"outputs/{save}/delay_matrix.txt", "w") as fd:
            for key, value in self.delay_matrix.items():
                fd.write(f"{key}: {value}\n")
                
    def save_parameters(self, save: str) -> None:
        """Save simulation parameters for reproducibility"""
        with open(f"outputs/{save}/parameters.txt", "w") as fd:
            fd.write(f"Rule: {self.rule}\n")
            fd.write(f"Max Delay: {self.maxDelay}\n")
            fd.write(f"Information Loss Probability (Tau): {self.tau}\n")
            fd.write(f"Grid Size: {self.width}x{self.height}\n")
            fd.write(f"Initial Density: {self.density[0]}\n")
            fd.write(f"Total Timestamps: {self.timeStamps}\n")

    def save_density(self, save: str) -> None:
        # Plot the density in a graph
        plt.figure(figsize=(10, 6))
        plt.plot(self.density)
        plt.title("Population Density Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Density")
        plt.grid(True)
        plt.savefig(f"outputs/{save}/density.png")

if __name__ == "__main__":
    
    parser = ArgumentParser(prog="Game of Life", epilog="Either provide an initconfig file or density along with a rule.")
    
    DEFAULTS = {
            "WIDTH": 50,
            "HEIGHT": 50,
            "AVTS": 1000,
            "TIMESTAMPS": 1500,
            "INITCONFIG": None,
            "RULE": "B3/S2,3",
            "DENSITY": 0.5,
            "EXTREMES": "0,1",
            "SAVEFILE": False,
            "SEED": None,
            "MAXDELAY": 1,
            "TAU": 0.0
        }

    parser.add_argument("-D", "--defaults", action="store_true")
    parser.add_argument("-W", "--width", default=DEFAULTS["WIDTH"], type=int)
    parser.add_argument("-H", "--height", default=DEFAULTS["HEIGHT"], type=int)
    parser.add_argument("-A", "--average-timestamp", default=DEFAULTS["AVTS"], type=int)
    parser.add_argument("-t", "--timestamps", default=DEFAULTS["TIMESTAMPS"], type=int)
    parser.add_argument("-i", "--initconfig", default=DEFAULTS["INITCONFIG"], type=str)
    parser.add_argument("-r", "--rule", default=DEFAULTS["RULE"], type=str)
    
    parser.add_argument("-l", "--looping-boundary", action="store_true")
    parser.add_argument("-d", "--density", default=DEFAULTS["DENSITY"], type=float)
    parser.add_argument("-e", "--extremes", default=DEFAULTS["EXTREMES"], type=str, help="To pass only one extreme: `,upper_bound` or `lower_bound,`\nExample: `,.9889` ; `.2243,`")

    parser.add_argument("-s", "--seed", default=DEFAULTS["SEED"], type=int)
    parser.add_argument("-V", "--visualize", action="store_true")
    parser.add_argument("-S", "--save", default=DEFAULTS["SAVEFILE"], type=str)
    
    # Delay parameter
    parser.add_argument("-M", "--max-delay", default=DEFAULTS["MAXDELAY"], type=int, help="Maximum delay for neighbor interactions (minimum is always 1)")
    
    # Information loss parameter
    parser.add_argument("-T", "--tau", default=DEFAULTS["TAU"], type=float, help="Probability of information loss (0.0 = no loss, 1.0 = complete loss)")

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
        
    # Validate the max_delay parameter
    if args.max_delay < 1:
        raise Exception("Maximum delay must be at least 1")
        
    # Validate the tau parameter
    if not 0 <= args.tau <= 1:
        raise Exception("Tau (information loss probability) must be between 0 and 1")

    ## --------------- START FROM HERE! ----------------- ##

    conway = GameOfLife((args.width, args.height),
                        args.timestamps, args.initconfig, args.rule, args.seed,
                        args.density, args.average_timestamp, parse_extremes(),
                        args.max_delay, args.tau)  # Add the max_delay and tau parameters
                        
    conway.simulate(vis=args.visualize, save=args.save)
    
    # Print average density
    print("Final density values:", conway.density[-5:])
    print("Average density:", sum(conway.density)/len(conway.density))
    
    # Print information loss statistics if applicable
    if args.tau > 0:
        print(f"Simulation ran with information loss probability (tau): {args.tau}")
        print("Higher values of tau increase the reliance on memory of previously observed states.")