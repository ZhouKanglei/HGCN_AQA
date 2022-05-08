#!/usr/bin/env bash
date;
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate AQA
conda info --envs


for class_idx in {1..6}
do
	echo $class_idx
	python main.py --phase train --exp_name ours --model models.MSGCN --device 32 --benchmark Seven --resume True --class_idx $class_idx 
done

