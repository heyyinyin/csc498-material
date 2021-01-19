#!/bin/bash

# install python 3.8 with anaconda
wget --no-check-certificate https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ./anaconda.sh
chmod u+x ./anaconda.sh
./anaconda.sh
~/anaconda3/bin/conda create -y -n py38-test python=3.8
~/anaconda3/bin/conda activate py38

# install some prereqs
~/anaconda3/bin/conda install -n py38-test jupyter matplotlib numpy

# install torch
~/anaconda3/bin/conda install -n py38-test pytorch

source activate py38-test

# install gym
pip install gym
