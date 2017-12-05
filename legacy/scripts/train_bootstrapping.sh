#! /bin/bash

# ===================================================================
#  Title  : YCL NIPS training script
#  Author : Lee Je Yeol, Haanju Yoo
#  Date   : 2017-04-22
# ===================================================================
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
# This script is assumed to run at the subfolder named 'script' under
# main source code folder contains 'train.py' and 'test.py'. If it is
# not, please modify 'main_dir_relative_path' variable.
# 
# ===================================================================


# enter option
model="AE-LTR"
dataset="avenue"  # avenue | ped1 | ped2 | enter | exit, and also supports 'all'
batch_size="160"
epochs="2000"
save_interval="50"
margin_sigma="1.5"
resample_interval="10"
num_gpu="1"
gpu_ids=""
pretrained_model_path=""
#pretrained_model_path="/home/neohanju/Workspace/Github/VAE_regularization/training_result/VAE_20170501-002802_avenue-ped1-ped2-enter-exit_Ace/VAE_20170501-002802_avenue-ped1-ped2-enter-exit_Ace_latest.pth"

display=true
debug_print=false

display_interval="1"

# enter processing type
do_test=false
do_evaluate=false

echo "=========================="
echo " Start training"
echo "=========================="

# path things
BASEDIR=$(pwd)
# echo "Script is plased at : $BASEDIR"
if [ "${pwd##*/}" = "scripts" ]; then
    main_dir_path=$(dirname $BASEDIR)
else
    main_dir_path=$BASEDIR
fi

result_path="$main_dir_path/training_result"  # all networks will be saved in here
# echo "Result path : $result_path"

# for naming of experiment
HOSTNAME=$(hostname)
STARTTIME=$(date +"%Y%m%d-%H%M%S")

# auto generate options
if [ $dataset = "all" ]; then
    dataset="avenue ped1 ped2 enter exit"
fi
OPT_DATA_ROOT=$YCL_DATA_ROOT
OPT_SAVE_NAME=$model"_"$STARTTIME"_"${dataset// /-}"_"$HOSTNAME
OPT_SAVE_PATH="$result_path/$OPT_SAVE_NAME"
# echo "OPT_SAVE_NAME is $OPT_SAVE_NAME"
# echo "OPT_SAVE_PATH is $OPT_SAVE_PATH"

# make save directory
if [ ! -d $result_path ]; then
    echo "[WARNING] Save path doesn't exist. Creating now"
    mkdir $result_path
fi
if [ ! -d $OPT_SAVE_PATH ]; then    
    mkdir $OPT_SAVE_PATH
else
    echo "[WARNING] ${OPT_SAVE_PATH##*/} already exists. Result will be overwritten"
fi
#copy test script for single test and single evaluate
cp "$main_dir_path/scripts/test_single.sh" "$OPT_SAVE_PATH/test_single.sh"
cp "$main_dir_path/scripts/evaluate_single.sh" "$OPT_SAVE_PATH/evaluate_single.sh"

# options
if [ "$pretrained_model_path" = "" ]; then
    OPT_LOAD_MODEL=""
else
    OPT_LOAD_MODEL="--load_model_path $pretrained_model_path"
fi
if [ $display = true  ]; then
	OPT_DISPLAY="--display --display_interval $display_interval"
else
	OPT_DISPLAY=""
fi
if [ $debug_print = true ]; then
	OPT_DEBUG_PRINT="--debug_print"
else
	OPT_DEBUG_PRINT=""
fi
if [ "$gpu_ids" = "" ]; then
    OPT_GPU_IDS="--gpu_ids $gpu_ids"
else
    OPT_GPU_IDS=""
fi

# option command
OPT_STRING="--model $model --dataset $dataset --data_root $OPT_DATA_ROOT --save_path $OPT_SAVE_PATH --save_name $OPT_SAVE_NAME --margin_sigma $margin_sigma --resample_interval $resample_interval --epochs $epochs $OPT_DISPLAY $OPT_DEBUG_PRINT --num_gpu $num_gpu --batch_size $batch_size --save_interval $save_interval $OPT_GPU_IDS $OPT_LOAD_MODEL"

#run train.py
CMD_STRING="python $main_dir_path/train_bootstrapping.py $OPT_STRING"
echo
echo $CMD_STRING
echo
$CMD_STRING |& tee $OPT_SAVE_PATH"/"$OPT_SAVE_NAME"_training.log"

echo "=========================="
echo " Training is done"
echo "=========================="

#auto test
if [ $do_test = true ]; then	
	bash "$OPT_SAVE_PATH/test_single.sh"
fi

#auto evaluate
if [ $do_evaluate = true ]; then
	bash "$OPT_SAVE_PATH/evaluate_single.sh"
fi

#()()
#('')
