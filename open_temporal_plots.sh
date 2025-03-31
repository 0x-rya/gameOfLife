
#!/bin/bash

if [[ ! -z $DISPLAY ]]; then

	for i in $(cat ~/aarya-thesis/outputs/dataset.txt | grep Yes | awk '{print $1}' | sed 's/\//-/g'); do
		open ~/aarya-thesis/outputs/temporal-stochastic/$i/plot.png
	done

else
	echo "No Display found. Please run the script from a GUI and not SSH connection."
fi
