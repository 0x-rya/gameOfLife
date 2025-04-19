#!/bin/bash

DIR="machine_learning"

# create dataset folders if they don't exist
mkdir -p dataset/alpha
mkdir -p dataset/spatial-stochastic
mkdir -p dataset/temporal-stochastic

# fetch data from remote host where it is being harvested
echo "Fetching data from remote host where it is being harvested..."
scp -i ../cloud/ssh-key-2025-03-19.key	\
	ubuntu@129.154.254.90:/home/ubuntu/home/outputs/dataset.txt \
	dataset/alpha/dataset.txt
scp  -i ../cloud/ssh-key-2025-03-19.key	\
	ubuntu@129.154.254.90:/home/ubuntu/new_thesis/outputs/spatial-stochastic/dataset.txt \
	dataset/spatial-stochastic/dataset.txt
scp  -i ../cloud/ssh-key-2025-03-19.key	\
	ubuntu@129.154.254.90:/home/ubuntu/new_thesis/outputs/temporal-stochastic/dataset.txt \
	dataset/temporal-stochastic/dataset.txt
echo "Done"

# filter down the dataset
echo "Filtering down the dataset... "
./$DIR/filter_pt.sh dataset/alpha
./$DIR/filter_pt.sh dataset/spatial-stochastic
./$DIR/filter_pt.sh dataset/temporal-stochastic
echo "Done"

# encode the txt into binary csv
echo -n "Encoding the filtered dataset into csv... "
python3 $DIR/txt_to_encoded_csv.py							\
	dataset/alpha/filtered_dataset.txt							\
	dataset/alpha/encoded_filtered_dataset.csv
python3 $DIR/txt_to_encoded_csv.py							\
	dataset/spatial-stochastic/filtered_dataset.txt				\
	dataset/spatial-stochastic/encoded_filtered_dataset.csv
python3 $DIR/txt_to_encoded_csv.py							\
	dataset/temporal-stochastic/filtered_dataset.txt			\
	dataset/temporal-stochastic/encoded_filtered_dataset.csv
echo "Done"

# run the ml models
echo ""
echo "Runnign ML Models on ALPHA"
python3 $DIR/classify_ml.py dataset/alpha/encoded_filtered_dataset.csv

echo ""
echo "Running ML Models on SPATIAL STOCHASTIC"
python3 $DIR/classify_ml.py dataset/spatial-stochastic/encoded_filtered_dataset.csv

echo ""
echo "Running ML Models on TEMPORAL STOCHASTIC"
python3 $DIR/classify_ml.py dataset/temporal-stochastic/encoded_filtered_dataset.csv
