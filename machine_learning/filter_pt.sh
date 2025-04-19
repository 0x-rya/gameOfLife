#!/bin/bash

dir=$1
input_file="${dir}/dataset.txt"
output_file="${dir}/filtered_dataset.txt"

# Extract all "Yes" rules
grep "	Yes	" "$input_file" > temp_yes.txt

# Count the number of "Yes" rules
total_yes=$(wc -l < temp_yes.txt)

# Calculate how many "No" rules to select so that they make up 50-60% of the final dataset
percentage=$(( 45 + RANDOM % 16 ))  # Random value between 45 and 55
num_no_to_select=$(( total_yes * percentage / (100 - percentage) ))
echo "${percentage}% -- ${num_no_to_select} no's selected"

# Extract random "No" rules
grep "	No	" "$input_file" | shuf -n "$num_no_to_select" > temp_no.txt

# Combine both into the output file
cat temp_yes.txt temp_no.txt | shuf > "$output_file"

# Cleanup
rm temp_yes.txt temp_no.txt

echo "Filtered dataset created: $output_file"

