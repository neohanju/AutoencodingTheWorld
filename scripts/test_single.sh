#! /bin/bash

# ===================================================================
#  Title  : YCL NIPS testing script
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
# 
# [REQUIREMENTS]
#
# 1. Environment variable
#
# Before to run, please add the environment variable 'YCL_DATA_ROOT'
# , which indicates the root folder containing all folders of dataset,
# in your ~/.bashrc :
# 	export YCL_DATA_ROOT="--your dataset path--"
# That path would contains subfolders like 'avenue', 'ped_1', etc.
#
# 2. Path to main directory
# 
# This script is assumed to run at the subfolder containing network
# file which is [MAIN_DIR]/[RESULT_DIR]/[NETWORK_DIR]. If it is not,
# please modify 'main_dir_relative_path' to the folder containing
# 'train.py' and 'test.py'.
# 
# ===================================================================

# enter option
dataset="avenue|enter|exit"  # avenue | ped1 | ped2 | enter | exit, and also support 'all'
display=true  # set this when you want to see the result with 'Visdom' package
debug_print=false

# model file name postfix can be '_latest.pth' or '_epoch_[NUMBER].pth'
model_file_name_postfix='_latest.pth'

# enter sequence type
# (!) Default is "False" and only change to "True" when testing a specific model manually.
# (!) Violations can cause double evaluation problems.
do_evaluate=false

echo "=========================="
echo " Start testing"
echo "=========================="

# check the location of this script
BASEDIR=$(dirname "$0")
echo "Script is plased at : $BASEDIR"
main_dir_relative_path="$BASEDIR/../.."  # MAIN_DIR / training_result / NETWORK_DIR

# auto generate options
if [ $dataset = "all" ]; then
	$dataset="avenue|ped1|ped2|enter|exit"
fi
OPT_DATA_ROOT=$YCL_DATA_ROOT
OPT_SAVE_NAME=$model"_"$STARTTIME"_"${dataset//|/-}"_"$HOSTNAME
OPT_SAVE_PATH="$result_path/$OPT_SAVE_NAME"
OPT_MODEL_PATH="$BASEDIR/${BASEDIR##*/}$model_file_name_postfix"

# boolean options
if [ $display = true  ]; then
	OPT_DISPLAY="--display"
else
	OPT_DISPLAY=""
fi
if [ $debug_print = true ]; then
	OPT_DEBUG_PRINT="--debug_print"
else
	OPT_DEBUG_PRINT=""
fi

OPT_STRING="--model_path $OPT_MODEL_PATH --dataset $dataset --data_root $OPT_DATA_ROOT $OPT_DISPLAY $OPT_DEBUG_PRINT"
# echo "Test option string is : $OPT_STRING"

#run test.py
CMD_STRING="python $main_dir_relative_path/test.py $OPT_STRING"
echo
echo $CMD_STRING
echo
$CMD_STRING

echo "=========================="
echo " Testing is done"
echo "=========================="

#auto evaluate
if [ $do_evaluate = true ]; then	
	bash evaluate_single.sh
fi

#()()
#('')