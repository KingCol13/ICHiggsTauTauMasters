#!/bin/sh
virtualenv --python=python3.6 htt
source htt/bin/activate
pip install --upgrade pip
pip install tensorflow-gpu
pip install uproot
pip install numpy
pip install pylorentz
pip install lbn
pip install matplotlib
python --version
