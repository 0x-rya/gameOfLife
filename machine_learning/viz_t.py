import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV
df = pd.read_csv("temporal_data.csv")

# Replace with your actual encoded column names
encoded_columns = ['B1_encoded', 'S1_encoded', 'B2_encoded', 'S2_encoded']

def row_to_bits(row):
    row_bits = []
    for col in encoded_columns:
        row_bits.extend([int(bit) for bit in row[col].strip('[').strip(']').split(' ')])
    return row_bits

# Convert to bit arrays
df['bit_array'] = df.apply(row_to_bits, axis=1)

# Separate output_binary == 1 and == 0
ones_df = df[df['output_binary'] == 1]
zeros_df = df[df['output_binary'] == 0]

# Build numpy arrays
ones_bits = np.stack(ones_df['bit_array'].values)
zeros_bits = np.stack(zeros_df['bit_array'].values)

# Count how many times each bit column is 1 in output_binary == 1
bit_sums = ones_bits.sum(axis=0)

# Print total columns and bit-wise counts
total_columns = ones_bits.shape[1]
print(f"Total number of columns (bits): {total_columns}")

for i, count in enumerate(bit_sums):
    print(f"Bit column {i:02}: {int(count)} ones")

# Check zero-count columns in output_binary == 1
print("\nChecking zero-count columns in output_binary == 1:")

for i, count in enumerate(bit_sums):
    if count == 0:
        # Count how many rows in output_binary == 0 have a 1 in this bit position
        count_in_zeros = int(zeros_bits[:, i].sum())
        print(f"Bit column {i:02}: 0 ones in output_binary==1, but {count_in_zeros} ones in output_binary==0")

# Plot ones bits
plt.figure(figsize=(4, len(ones_bits) * 0.1))  # Smaller square size
plt.imshow(ones_bits, cmap='gray_r', vmin=0, vmax=1, aspect='auto')
plt.axis('off')
plt.tight_layout()
plt.savefig("temporal_data_visualization_ones.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Plot zeroes bits
plt.figure(figsize=(4, len(zeros_bits) * 0.1))  # Smaller square size
plt.imshow(zeros_bits, cmap='gray_r', vmin=0, vmax=1, aspect='auto')
plt.axis('off')
plt.tight_layout()
plt.savefig("temporal_data_visualization_zeros.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()