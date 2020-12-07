#!/bin/sh
source htt/bin/activate
export LD_LIBRARY_PATH=/vols/cms/ktc17/cuda/lib64:$LD_LIBRARY_PATH
python helloTF.py
