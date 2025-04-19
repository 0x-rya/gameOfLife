from itertools import product

def generate_all_rules():
    """
    Generate all possible rules where the number of variables 
    for both B and S parts ranges from 1 to 8.
    """
    rules = []
    
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
    
    return rules

# Generate and print all rules
all_rules = generate_all_rules()
print(f"Total number of rules: {len(all_rules)}")

# Optional: Print first 20 rules to get a sense of the variety
print("\nFirst 20 rules:")
for rule in all_rules[:20]:
    print(rule)

# Optional: Save all rules to a file
with open('all_rules.txt', 'w') as f:
    for rule in all_rules:
        f.write(rule + '\n')

# Optional: Provide a rule count breakdown
rule_breakdown = {}
for b_vars in range(1, 9):
    for s_vars in range(1, 9):
        rules_count = len([r for r in all_rules if f"B{','.join(['1']*b_vars)},..." in r and f"S{','.join(['1']*s_vars)},..." in r])
        rule_breakdown[(b_vars, s_vars)] = rules_count

print("\nRule count breakdown:")
for (b_vars, s_vars), count in rule_breakdown.items():
    print(f"B variables: {b_vars}, S variables: {s_vars} - {count} rules")
