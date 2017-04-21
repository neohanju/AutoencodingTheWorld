# by Lee Je Yeol
# YCL NIPS
# test module

# plz add env :
# export YCL_DATA_ROOT="--your dataset path--"

# enter option
dataset="avenue-enter-exit"

# enter sequence type
# (!) Default is "False" and only change to "True" when testing a specific model manually.
# (!) Violations can cause double evaluation problems.
autoevaluate="False"



# auto entered variables
model_path=$PWD
data_root=$YCL_DATA_ROOT

#run test.py
#python ../../test.py --model_path $model_path --dataset $dataset --data_root $data_root

echo "test done"

#auto evaluate
if [ $autoevaluate = True ] ; then
	cd ../..
	sh evaluate.sh
fi