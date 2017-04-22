#! /bin/bash

# ===================================================================
#  Title  : YCL NIPS evaluation script
#  Author : Lee Je Yeol, Haanju Yoo
#  Date   : 2017-04-22
# ===================================================================
#
# [NOTICE]
#
# This file is designed to run under the folder containing network
# file. It will be copied to that folder automatically when you run 
# 'train.sh'.
#
# ===================================================================

echo "=========================="
echo " Start evaluation"
echo "=========================="

# auto entered variables
model_path=$PWD
model=${model_path##*/}

# model file name postfix can be '_latest.pth' or '_epoch_[NUMBER].pth'
model_file_name_postfix='_latest.pth'


matlab -nodesktop -r "evaluate $model_path"

echo "=========================="
echo " Evaluation is done"
echo "=========================="

#()()
#('')