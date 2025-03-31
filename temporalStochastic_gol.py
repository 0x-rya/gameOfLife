import random
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from alpha_gol import alphaGOL

class temporalStochasticGOL(alphaGOL):

    def __init__(
            self,
            sizeXY: tuple[int, int],
            timeStamps: int,
            initConfig: str | None,
            rule: tuple[str, str],
            seed: int,
            density: float,
            alpha: float,               ## what fraction of the population gets updated
            lmbda: float,               ## what fraction of the population gets rule 1
            averageTimeStamp: int,
            extremes: tuple[int, int]
        ) -> None:

        super().__init__(sizeXY, timeStamps, initConfig, rule[0], seed, density, alpha, averageTimeStamp, extremes)
        self.lmbda = lmbda

        if not self.validateRule(rule[1]):  ## rule[0] already tested in super().__init__()
            raise Exception("Invalid Rule.\nRule should be of format Bl,m,n,.../Sl,n,m\nExample: B3/S2,3...")

        self.rule = rule

        try:
            birth, survival = rule[0].split("/")
            
            birth = birth.strip("B").split(",")
            survival = survival.strip("S").split(",")

            birth = [int(x) for x in birth]
            survival = [int(x) for x in survival]

            self.rule1 = rule
            self.birth1 = birth
            self.survival1 = survival
            
            birth, survival = rule[1].split("/")
            
            birth = birth.strip("B").split(",")
            survival = survival.strip("S").split(",")

            birth = [int(x) for x in birth]
            survival = [int(x) for x in survival]

            self.rule2 = rule
            self.birth2 = birth
            self.survival2 = survival

            noErr = True

        except:
                noErr = False

        if not noErr:
            raise Exception("Invalid Rule.\nRule should be of format Bl,m,n,.../Sl,n,m\nExample: B3/S2,3...")

    def getNextIter(self, x: int, y: int) -> int:

        aliveNeighbours = sum(self.getNeighbours(x, y))
        if self.automata[y][x] == 0:
            ## will be 1 only if exactly three neighbours are alive
            if self.first:
                if aliveNeighbours in self.birth1:
                    return 1
                else:
                    return 0
            else:
                if aliveNeighbours in self.birth2:
                    return 1
                else:
                    return 0
        else:
            ## will be 1 if 2-3 neighbours are alive
            if self.first:
                if aliveNeighbours in self.survival1:
                    return 1
                else:
                    return 0
            else:
                if aliveNeighbours in self.survival1:
                    return 1
                else:
                    return 0

    def simulate(self, vis: bool, save: bool | str):

        self.imgs.append(self.automata)
        self.density.append(self.calcDensity())
        for ts in range(self.timeStamps):
            if random.random() <= self.lmbda:
                self.first = True
            else:
                self.first = False
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

if __name__ == "__main__":
    
    parser = ArgumentParser(prog="Stochastic Game of Life", epilog="Either provide an initconfig file or density along with a rule.")
    
    DEFAULTS = {
            "WIDTH": 50,
            "HEIGHT": 50,
            "AVTS": 1000,
            "TIMESTAMPS": 1500,
            "INITCONFIG": None,
            "RULE": "B3/S2,3",
            "DENSITY": 0.5,
            "ALPHA": 1,
            "LAMBDA": 1,
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
    parser.add_argument("-l", "--lmbda", default=DEFAULTS["LAMBDA"], type=float)
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

    conway = temporalStochasticGOL((args.width, args.height),
                        args.timestamps, args.initconfig, args.rule, args.seed,
                        args.density,
                        args.alpha, args.lmbda, args.average_timestamp, parse_extremes())
    conway.simulate(vis=args.visualize, save=args.save)
    
    ## average density
    print(conway.density)
    print(sum(conway.density)/len(conway.density))
