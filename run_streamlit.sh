#!/bin/bash

source /opt/anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate dhh-comm-env

cd /Users/ericwang/git/dhh-comm

/opt/anaconda3/envs/dhh-comm-env-py312/bin/streamlit run app.py --server.runOnSave=true