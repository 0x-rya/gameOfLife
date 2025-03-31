import sys
import random
from enum import Enum
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import combinations
import spatialStochastic_gol as gol
from itertools import combinations

DEFAULTS = {
    "WIDTH": 50,
    "HEIGHT": 50,
    "AVTS": 2000,
    "TIMESTAMPS": 2500,
    "INITCONFIG": None,
    "LOOPING_BOUNDS": False,
    "DENSITY": 0.5,
    "LMBDA": 1,
    "EXTREMES": "0.05,0.95",
    "SAVEFILE": False,
    "MAX_ITERS": 10
}

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

def plot_phase_transitions_with_bump(data: list[float], lmbda_scale: list[float], transitions: list, rule: str, lower_threshold = 0.05, upper_threshold = 0.95, min_bump = 0.2):
    plt.figure(figsize=(10, 6))
    plt.plot(lmbda_scale, data,marker='o', label="Data Values")
    plt.axhline(y=lower_threshold, color='green', linestyle='--', label="Lower Threshold")
    plt.axhline(y=upper_threshold, color='red', linestyle='--', label="Upper Threshold")

    # Plot bump-adjusted lines
    plt.axhline(y=lower_threshold + min_bump, color='blue', linestyle='--', label="Lower Threshold + Bump")
    plt.axhline(y=upper_threshold - min_bump, color='orange', linestyle='--', label="Upper Threshold - Bump")

    # Mark transitions
    for idx, prev_phase, curr_phase in transitions:
        plt.scatter(lmbda_scale[idx], data[idx], color='purple', zorder=5,
                    label=f"Transition to {curr_phase}" if idx == transitions[0][0] else None)

    # Add labels and legend
    plt.title(f"{rule} - Phase Transitions in Data with Bump Lines")
    plt.xlabel("Lmbda")
    plt.ylabel("Average Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"outputs/spatial-stochastic/{rule.replace('/', '-')}/plot")
    # plt.show()
    plt.close()

def generate_rule_pairs(num_pairs=1000):
    # Generate all possible rules first
    all_rules = []
    
    # Generate rules for B and S combinations
    for b_count in range(1, 9):
        b_combinations = list(combinations(range(1, 9), b_count))
        
        for s_count in range(1, 9):
            s_combinations = list(combinations(range(1, 9), s_count))
            
            for b_combo in b_combinations:
                for s_combo in s_combinations:
                    # Format B part
                    b_part = 'B' + ','.join(map(str, b_combo))
                    
                    # Format S part
                    s_part = 'S' + ','.join(map(str, s_combo))
                    
                    # Combine parts
                    rule = f"{b_part}/{s_part}"
                    all_rules.append(rule)
    
    # Randomly sample unique pairs
    rule_pairs = set()
    attempts = 0
    max_attempts = num_pairs * 10  # Prevent infinite loop
    
    while len(rule_pairs) < num_pairs and attempts < max_attempts:
        # Randomly choose two different rules
        rule1 = random.choice(all_rules)
        rule2 = random.choice(all_rules)
        
        # Ensure rules are different
        if rule1 != rule2:
            # Sort to prevent duplicate reverse pairs
            pair = tuple(sorted([rule1, rule2]))
            rule_pairs.add(pair)
        
        attempts += 1
    
    return list(rule_pairs)

# Optional: if you want to limit or sample the pairs
def sample_rule_pairs(n=1000):
    pairs = generate_rule_pairs(n)
    return pairs

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 0 and len(args) != 1:
        print(len(args))
        print(f"Incorrect usage.\nCorrect Usage:\n\t\tpython3 {sys.argv[0]}\n\t\tpython3 {sys.argv[0]} N_ITERS")
        exit(0)

    DEFAULTS["SEED"] = 1433  # or generate_seed()
    log(f"No seed provided, generating random seed: {DEFAULTS['SEED']}")

    if len(args) == 1:
        DEFAULTS["MAX_ITERS"] = int(args[0])
    elif len(args) != 0:
        print(f"Incorrect usage.\nCorrect Usage:\n\t\tpython3 {sys.argv[0]}\n\t\tpython3 {sys.argv[0]} N_ITERS")
        exit(0)

    random.seed(DEFAULTS["SEED"])

    # Generate rule pairs
    sample_pairs = sample_rule_pairs(300000)

    usedRules = set()
    lmbdas = [i/20 for i in range(1, 21)]

    while len(sample_pairs) > 0:
        # Select a random rule pair
        randSeed = generate_seed()
        randRule = random.choice(sample_pairs)
        sample_pairs.remove(randRule)

        # Convert tuple to string for directory and file naming
        rule_str = f"{randRule[0]}__{randRule[1]}".replace('/', '-')
        dPath = Path(f"outputs/spatial-stochastic/{rule_str}/")

        # Regenerate if rule pair already used or directory exists
        while rule_str in usedRules or dPath.exists():
            log(f"Generated Rule Pair {rule_str} already tested, re-generating")
            
            # If no more pairs left, break
            if not sample_pairs:
                break
            
            randRule = random.choice(sample_pairs)
            sample_pairs.remove(randRule)
            rule_str = f"{randRule[0]}__{randRule[1]}".replace('/', '-')
            dPath = Path(f"outputs/spatial-stochastic/{rule_str}/")

        # Create directory and track used rules
        usedRules.add(rule_str)
        dPath.mkdir(parents=True, exist_ok=True)

        # Open file for writing lambda and density results
        fp = open(f"outputs/spatial-stochastic/{rule_str}/lmbdas.txt", "w")
        log(f"Processing Rule Pair {len(usedRules)}/{DEFAULTS['MAX_ITERS']}: {randRule}")

        avg_density = []
        for lmdba in lmbdas:
            # Creating game of life with both rules
            conway = gol.spatialStochasticGOL(
                (DEFAULTS["WIDTH"], DEFAULTS["HEIGHT"]),
                DEFAULTS["TIMESTAMPS"],
                DEFAULTS["INITCONFIG"],
                randRule,  # Pass the entire rule pair
                randSeed,
                DEFAULTS["DENSITY"],
                1,
                lmdba,
                DEFAULTS["AVTS"],
                parse_extremes(DEFAULTS["EXTREMES"])
            )

            # Simulating game of life
            log(f"Running Conway Simulation for lmbda - {lmdba}")
            conway.simulate(False, False)

            avg_den = conway.getAvgDensity()
            log(f"Calculated Average Density for lmbda - {lmdba}: {avg_den}")
            fp.write(f"{lmdba} {avg_den}\n")
            avg_density.append(avg_den)

        fp.close()

        # Detect phase transitions
        bump = 2
        pts = detect_phase_transitions(avg_density)

        # Write to dataset
        fPath = Path("outputs/spatial-stochastic/dataset.txt")
        with open(fPath, "a" if fPath.exists() else "w") as fp:
            fp.write(f"{randRule[0]}\t{randRule[1]}\t{'Yes' if len(pts) != 0 else 'No'}\t{len(pts)}\n")

        log(f"Detected Phase Transitions: {pts}")
        log("Plotting lmbda vs avg density")
        plot_phase_transitions_with_bump(avg_density, lmbdas, pts, rule_str)
