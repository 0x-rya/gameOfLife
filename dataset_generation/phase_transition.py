import sys
import random
from enum import Enum
from pathlib import Path
import alphaGOL as gol
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt

if __name__ == "__main__":

    def log(text: str) -> None:

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - {text}")

    def generate_seed() -> int:

        return int(random.random() * 10000)

    def generate_random_rule(seed: int) -> str:
        
        birth_string = "B"
        survival_string = "/S"
        
        random.seed(seed)

        valid_bNeighbours = random.randint(1, 8)
        log(f"Number of Valid Neighbours for Birth: {valid_bNeighbours}")
        valid_sNeighbours = random.randint(1, 8)
        log(f"Number of Valid Neighbours for Survival: {valid_sNeighbours}")

        bSet = set()
        while len(bSet) < valid_bNeighbours:
            bSet.add(str(random.randint(1, 8)))

        sSet = set()
        while len(sSet) < valid_sNeighbours:
            sSet.add(str(random.randint(1, 8)))

        birth_string += ",".join(sorted(list(bSet)))
        survival_string += ",".join(sorted(list(sSet)))
        final_string = birth_string + survival_string

        return final_string

    DEFAULTS = {
            "WIDTH": 50,
            "HEIGHT": 50,
            "AVTS": 2000,
            "TIMESTAMPS": 2500,
            "INITCONFIG": None,
            "LOOPING_BOUNDS": False,
            "DENSITY": 0.5,
            "ALPHA": 1,
            "EXTREMES": "0.05,0.95",
            "SAVEFILE": False,
            "MAX_ITERS": 10
    }

    args = sys.argv[1:]
    if len(args) != 0 and len(args) != 1:
        
        print(len(args))
        print(f"Incorrect usage.\nCorrect Usage:\n\t\tpython3 {sys.argv[0]}\n\t\tpython3 {sys.argv[0]} N_ITERS")
        exit(0)

    DEFAULTS["SEED"] = 1433 # generate_seed()
    log(f"No seed provided, generating random seed: {DEFAULTS['SEED']}")

    if len(args) == 1:

        DEFAULTS["MAX_ITERS"] = int(args[0])

    elif len(args) != 0:
        
        print(f"Incorrect usage.\nCorrect Usage:\n\t\tpython3 {sys.argv[0]}\n\t\tpython3 {sys.argv[0]} N_ITERS")
        exit(0)

    def parse_extremes(extremeString: str) -> tuple[int, int]:

        exts = extremeString.split(",")
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

    def detect_phase_transitions(data: list[float], min: float = 0.05, max: float = 0.95, bump: float = 0.2) -> list[tuple[int, str, str]]:

        class TransitionType(Enum):

            NoTransition        = 0
            UpperAToITransition = 1
            UpperIToATransition = 2
            LowerAToITransition = 3
            LowerIToATransition = 4

        class PhaseType(Enum):

            UpperActive     = 0
            UpperInactive   = 1
            Inactive        = 2
            LowerInactive   = 3
            LowerActive     = 4

        def classifyPhase(value: float) -> PhaseType:

            if max <= value <= 1:
                return PhaseType.UpperActive
            elif max - bump < value < max:
                return PhaseType.UpperInactive
            elif min + bump <= value <= max - bump:
                return PhaseType.Inactive
            elif min < value < min + bump:
                return PhaseType.LowerInactive
            elif 0 <= value <= min:
                return PhaseType.LowerActive
            else:
                raise Exception(f"Invalid Input {value} not in range [0, 1]")

        def classifyPT(oldPhase: PhaseType, newPhase: PhaseType) -> TransitionType:

            if oldPhase == PhaseType.UpperActive and newPhase == PhaseType.Inactive:
                return TransitionType.UpperAToITransition
            elif oldPhase == PhaseType.Inactive and newPhase == PhaseType.UpperActive:
                return TransitionType.UpperIToATransition
            elif oldPhase == PhaseType.LowerActive and newPhase == PhaseType.Inactive:
                return TransitionType.LowerAToITransition
            elif oldPhase == PhaseType.Inactive and newPhase == PhaseType.LowerActive:
                return TransitionType.LowerIToATransition
            else:
                return TransitionType.NoTransition

        returnList = []
        prevPhase = classifyPhase(data[0])
        for i, currValue in enumerate(data):

            currPhase = classifyPhase(currValue)
            phaseTransition = classifyPT(prevPhase, currPhase)
            if phaseTransition != TransitionType.NoTransition:
                returnList.append((i, prevPhase, currPhase))
                prevPhase = currPhase

        return returnList

    def plot_phase_transitions_with_bump(data: list[float], alpha_scale: list[float], transitions: list, rule: str, lower_threshold = 0.05, upper_threshold = 0.95, min_bump = 1.5):

        plt.figure(figsize=(10, 6))
        plt.plot(alpha_scale, data,marker='o', label="Data Values")
        plt.axhline(y=lower_threshold, color='green', linestyle='--', label="Lower Threshold")
        plt.axhline(y=upper_threshold, color='red', linestyle='--', label="Upper Threshold")
        
        # Plot bump-adjusted lines
        plt.axhline(y=lower_threshold + min_bump, color='blue', linestyle='--', label="Lower Threshold + Bump")
        plt.axhline(y=upper_threshold - min_bump, color='orange', linestyle='--', label="Upper Threshold - Bump")

        # Mark transitions
        for idx, prev_phase, curr_phase in transitions:
            plt.scatter(alpha_scale[idx], data[idx], color='purple', zorder=5, 
                        label=f"Transition to {curr_phase}" if idx == transitions[0][0] else None)

        # Add labels and legend
        plt.title(f"{rule} - Phase Transitions in Data with Bump Lines")
        plt.xlabel("Alpha")
        plt.ylabel("Average Density")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"outputs/{rule.replace('/', '-')}/plot")
        # plt.show()
        plt.close()

    random.seed(DEFAULTS["SEED"])
    usedRules = set()
    rules = []
    alphas = [i/20 for i in range(1, 21)]

    # Iterate through possible numbers of variables (1 to 8)
    for b_vars in range(1, 5):  # Number of variables in B part
        for s_vars in range(1, 5):  # Number of variables in S part
            # Generate all combinations for B part
            b_combinations = list(product(range(1, 9), repeat=b_vars))
            
            # Generate all combinations for S part
            s_combinations = list(product(range(1, 9), repeat=s_vars))
            
            # Create rules for each combination
            for b_combo in b_combinations:
                for s_combo in s_combinations:
                    # Format B part
                    b_part = 'B' + ','.join(map(str, b_combo))
                    
                    # Format S part
                    s_part = 'S' + ','.join(map(str, s_combo))
                    
                    # Combine parts
                    rule = f"{b_part}/{s_part}"
                    rules.append(rule)

    def get_random_rule() -> str:

        rule = rules.pop(random.randint(0, len(rules) - 1))
        usedRules.add(rule)

        return rule

    while len(rules) > 0:

        randSeed = generate_seed()
        randRule = get_random_rule() # generate_random_rule(randSeed)
        dPath = Path(f"outputs/{randRule.replace('/', '-')}/")
        while randRule in usedRules or dPath.exists():
            log("Generated Rule already tested, re-generating rule")
            randSeed = generate_seed()
            randRule = generate_random_rule(randSeed)
            dPath = Path(f"outputs/{randRule.replace('/', '-')}/")
        usedRules.add(randRule)
        dPath.mkdir(parents=True, exist_ok=True)     ## make sure the directory exists
        fp = open(f"outputs/{randRule.replace('/', '-')}/alphas.txt", "w")
        log(f"Random Seed and Rule for {len(usedRules)}/{DEFAULTS['MAX_ITERS']} iteration: {randSeed} & {randRule}")

        avg_density = []
        for alpha in alphas:
            
            ## creating game of life
            conway = gol.alphaGOL((DEFAULTS["WIDTH"], DEFAULTS["HEIGHT"]),
                                    DEFAULTS["TIMESTAMPS"], DEFAULTS["INITCONFIG"], randRule, randSeed,
                                    DEFAULTS["DENSITY"],
                                    alpha, DEFAULTS["AVTS"], parse_extremes(DEFAULTS["EXTREMES"]))

            ## simulating game of life
            log(f"Running Conway Simulation for alpha - {alpha}")
            conway.simulate(False, False)

            avg_den = conway.getAvgDensity()
            log(f"Calculated Average Density for alpha - {alpha}: {avg_den}")
            fp.write(f"{alpha} {avg_den}\n")
            avg_density.append(avg_den)

        fp.close()
        bump = 2
        pts = detect_phase_transitions(avg_density)
        fPath = Path("outputs/dataset.txt")
        if fPath.exists():
            fp = open("outputs/dataset.txt", "a")
        else:
            fp = open("outputs/dataset.txt", "w")
        fp.write(f"{randRule}\t{'Yes' if len(pts) != 0 else 'No'}\t{len(pts)}\n")
        log(f"Detected Phase Transitions: {pts}")

        log("Plotting alpha vs avg density")
        plot_phase_transitions_with_bump(avg_density, alphas, pts, randRule)
