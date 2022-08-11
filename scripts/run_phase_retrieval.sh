#!/bin/bash

python solve.py --num_iterations=10000 --num_repeats=4 --algorithm=ER;

python solve.py --num_iterations=10000 --num_repeats=4 --algorithm=HIO;
