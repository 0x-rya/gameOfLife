import random
import numpy as np
from pathlib import Path
from copy import deepcopy
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
            looping_boundary: bool,
            density: float,
            alpha: float,               ## what fraction of the population gets updated
            averageTimeStamp: int,
            extremes: tuple[int, int]   ## what value in replacement of 0, 1 is to be considered for phase transition
        ) -> None:
        
        if not isinstance(sizeXY, abc.Iterable):
            raise Exception(f"Size has to be an iterable and not of type {type(sizeXY)}")
        
        if len(sizeXY) != 2:
            raise Exception("Size can only be of length 2")

        self.imgs: list[list[list[int]]] = []
        self.width = sizeXY[0]
        self.height = sizeXY[1]
        self.seed = seed
        self.avts = averageTimeStamp
        self.timeStamps = timeStamps
        if not self.validateRule(rule):
            raise Exception("Invalid Rule.\nRule should be of format Bl,m,n,.../Sl,n,m\nExample: B3/S2,3...")

        self.alpha = alpha
        self.density = []
        self.extremes = extremes

        if initConfig != None:
            self.automata = self.readInitConfig(initConfig)

        else:
            self.automata = self.genInitConfig(density)

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
            return False

    def getNeighbours(self, x: int, y: int) -> list[int]:
        
        neighbours = []
        widthRange, heightRange = [-1, 0, 1], [-1, 0, 1]
        
        if x == 0:
            widthRange.remove(-1)
        if x == self.width - 1:
            widthRange.remove(1)

        if y == 0:
            heightRange.remove(-1)
        if y == self.height - 1:
            heightRange.remove(1)

        for w in widthRange:
            for h in heightRange:
                if w == 0 and h == 0:
                    continue
                neighbours.append(self.automata[y+h][x+w])

        return neighbours

    def updateAutomata(self) -> None:

        nextIter = deepcopy(self.automata)
        for w in range(self.width):
            for h in range(self.height):
                if random.random() <= self.alpha:
                    nextIter[h][w] = self.getNextIter(w, h)

        self.automata = nextIter

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

            frames.append([plt.imshow(self.imgs[i], animated=True)])
        
        ani = animation.ArtistAnimation(fig, frames, interval=800, blit=True)
        if isinstance(save, str):
            ani.save(f"outputs/{save}/game.mp4")
        plt.show()

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
        for ts in range(self.timeStamps):
            self.updateAutomata()
            if ts >= self.avts:
                self.density.append(self.calcDensity())
            self.imgs.append(self.automata)
        if isinstance(save, str):
            Path(f"outputs/{save}/").mkdir(parents=True, exist_ok=True)     ## make sure the directory exists
            self.save_seed(save)                                            ## save seed for reproducibility's sake
        if vis:
            self.record(save=save)

    def save_seed(self, save: str) -> None:
        
        with open(f"outputs/{save}/seed.txt", "w") as fd:

            fd.write(str(self.seed))

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
            "ALPHA": 1,
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
    
    parser.add_argument("-l", "--looping-boundary", action="store_true")
    parser.add_argument("-d", "--density", default=DEFAULTS["DENSITY"], type=float)
    parser.add_argument("-a", "--alpha", default=DEFAULTS["ALPHA"], type=float)
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

    conway = GameOfLife((args.width, args.height),
                        args.timestamps, args.initconfig, args.rule, args.seed,
                        args.looping_boundary, args.density,
                        args.alpha, args.average_timestamp, parse_extremes())
    conway.simulate(vis=args.visualize, save=args.save)
    
    ## average density
    print(sum(conway.density)/len(conway.density))
    print(conway.density)

    ## plot the density in a graph
    plt.plot(conway.density)
    plt.show()
