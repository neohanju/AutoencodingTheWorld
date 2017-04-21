# by Lee Je Yeol
# YCL NIPS
# train module

# plz add env :
# export YCL_DATA_ROOT="--your dataset path--"


# enter option
model="AE_LTR"
dataset="avenue-enter-exit"
display="False"

# enter sequence type
autotest="True"
autoevaluate="True"

# if you set autotest to True, you can set dataset for test too.
autotest_dataset="avenue-enter-exit"


# auto entered variables
HOSTNAME=$(hostname)
STARTTIME=$(date +"%Y%m%d-%H%M%S")
data_root=$YCL_DATA_ROOT
save_path="./training_result/"$model"_"$STARTTIME"_"$dataset-$HOSTNAME

#make save directory
if [ ! -d "training_result" ]
then
    echo "training_result doesn't exist. Creating now"
    mkdir "./training_result"
else
    echo "training_result exists"
fi

if [ ! -d "${save_path##*./}" ]
then
    echo "${save_path##*./}"" doesn't exist. Creating now"
    mkdir ./"${save_path##*./}"
else
    echo "${save_path##*./}"" File exists"
fi


#run train.py
#python train.py --model $model --dataset $dataset --data_root $data_root --save_path $save_path --display --debug_print 2>&1 | tee $save_path"/terminal.log"

echo "train done"

#copy test script for single test and single evaluate
cp "./test_single.sh" $save_path"/test_single.sh"
cp "./evaluate_single.sh" $save_path"/evaluate_single.sh"

#auto test
if [ $autotest = True ] ; then
	cd $save_path
	sh test_single.sh
fi

#auto evaluate
if [ $autoevaluate = True ] ; then
	sh evaluate_single.sh
fi