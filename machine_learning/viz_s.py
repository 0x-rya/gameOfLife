import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv("spatial_data.csv")

# Filter rows where output_binary == 1
filtered_df = df[df['output_binary'] == 1]

# Replace with your actual encoded column names
encoded_columns = ['B1_encoded', 'S1_encoded', 'B2_encoded', 'S2_encoded']

# Convert binary strings to arrays of 0s and 1s
def binary_string_to_array(s: str):
    return np.array([int(bit) for bit in s.strip('[').strip(']').split(' ')])

# Build all row images as 1x32 arrays
row_images = []

for _, row in filtered_df.iterrows():
    row_bits = []
    for col in encoded_columns:
        row_bits.extend(binary_string_to_array(row[col]))
    row_images.append(row_bits)

# Convert to numpy array for imshow
image = np.array(row_images)

# Plot
plt.figure(figsize=(4, len(image) * 0.1))  # Smaller square size
plt.imshow(image, cmap='gray', vmin=0, vmax=1, aspect='auto')
plt.axis('off')
plt.tight_layout()
plt.savefig("spatial_data_visualization.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Count how many times each bit column has a 1
bit_sums = image.sum(axis=0)

# Print stats
total_columns = image.shape[0]
print(f"Total number of columns (bits): {total_columns}")
for i, count in enumerate(bit_sums):
    print(f"Bit column {i:02}: {int(count)} ones")