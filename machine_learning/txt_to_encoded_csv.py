import sys

def parse_rule(rule):
    b_part, s_part = rule.split('/')
    b_nums = list(map(int, b_part[1:].split(',')))
    s_nums = list(map(int, s_part[1:].split(',')))
    return b_nums, s_nums

def encode_bits(nums):
    bits = ['0'] * 8
    for num in nums:
        if 1 <= num <= 8:
            bits[num - 1] = '1'
    return ''.join(bits)

def encode_line(line):
    parts = line.strip().split('\t')
    
    if len(parts) == 3:
        rule, pt, num_pt = parts
        b, s = parse_rule(rule)

        bin_b = encode_bits(b)
        bin_s = encode_bits(s)

        full_binary = f"{bin_b}{bin_s}"
        pt_binary = '1' if pt.strip().lower() == 'yes' else '0'

        return f"{full_binary},{pt_binary},{num_pt.strip()}"
    
    elif len(parts) == 4:

        rule1, rule2, pt, num_pt = parts
        b1, s1 = parse_rule(rule1)
        b2, s2 = parse_rule(rule2)

        bin_b1 = encode_bits(b1)
        bin_s1 = encode_bits(s1)
        bin_b2 = encode_bits(b2)
        bin_s2 = encode_bits(s2)

        full_binary = f"{bin_b1}{bin_s1}{bin_b2}{bin_s2}"
        pt_binary = '1' if pt.strip().lower() == 'yes' else '0'

        return f"{full_binary},{pt_binary},{num_pt.strip()}"
    
    else:
        return None  # malformed line

def process_file(input_path, output_path=None):
    with open(input_path, 'r') as infile:
        lines = infile.readlines()

    encoded_lines = [encode_line(line) for line in lines if encode_line(line)]

    if output_path:
        with open(output_path, 'w') as outfile:
            for encoded in encoded_lines:
                outfile.write(encoded + '\n')
    else:
        for encoded in encoded_lines:
            print(encoded)

# Example usage
if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) != 2:
        print(f"Invalid Usage. Correct Usage `{sys.argv[0]} input_file.txt output_file.csv`")
        exit()

    process_file(args[0], args[1])  # Change to your actual filenames

