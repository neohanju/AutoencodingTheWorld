# by Lee Je Yeol
# YCL NIPS
# evaluate module


# auto entered variables
model_path=$PWD

model=${model_path##*/}


matlab -r "test_evaluate $model_path"

echo "evaluate done"