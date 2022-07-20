# Toroidal-PSDA
A probabilistic scoring backend for length-normalized embeddings. 

This repo is private for now. We will open source it later.

## Install
Dependencies are numpy, scipy and [PYLLR](https://github.com/bsxfan/PYLLR).

To install, put the root (the folder that contains the package tpsda) on your python path.

## Demo

- A working demo is here:
<https://github.com/bsxfan/Toroidal-PSDA/blob/main/tpsda/toroidal/toroid_vs_cosred.py>.
It can be run as a script. It makes synthetic data and demonstrates training and scoring.

- Further insight into the model and the training em-algorithm can be gained by running this demo script:
<https://github.com/bsxfan/Toroidal-PSDA/blob/main/tpsda/toroidal/test_em.py>.
It plots low-dimensional data on an interactive rotatable globe (if your plotting backend allows). 
