#bash

python -m cProfile $1 | grep -E 'seconds|output'