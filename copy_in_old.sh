#!/bin/bash
# set -x
source_path=$1
dest_path=$2
max_processes=$3
max_files=$4
data_type=$5
if [[ -f "copy_commands.txt" ]]; then
    rm "copy_commands.txt"
    #echo copy command file already exists. Exiting.
fi

# make destination dirs
dir_name=$(basename ${source_path})
mkdir -p ${dest_path}/${dir_name}
mkdir -p ${dest_path}/${dir_name}/train/
mkdir -p ${dest_path}/${dir_name}/val/

# Set internal dir structure
cfg_file_source=${source_path}/cfg.yaml
train_dir_source=${source_path}/train/
val_dir_source=${source_path}/val/

# Create arrays of source and destination paths
train_shards=($(gfal-ls ${train_dir_source}))
val_shards=($(gfal-ls ${val_dir_source}))
train_dest_list=($(printf "%0.s${dest_path}/${dir_name}/train/ " $(seq 1 ${#train_shards[@]})))
val_dest_list=($(printf "%0.s${dest_path}/${dir_name}/val/ " $(seq 1 ${#val_shards[@]})))
file_list_source_train=( "${train_shards[@]/#/${train_dir_source}}" )
file_list_dest_train=( "${train_dest_list[@]}" )
file_list_source_val=( "${val_shards[@]/#/${val_dir_source}}" )
file_list_dest_val=( "${val_dest_list[@]}" )

# Write copy commands to file
max_train_files=${max_files}
max_train_files=$(( ${#file_list_source_train[@]} < ${max_train_files} ? ${#file_list_source_train[@]} : ${max_train_files} ))
max_val_files=${max_files}
max_val_files=$(( ${#file_list_source_val[@]} < ${max_val_files} ? ${#file_list_source_val[@]} : ${max_val_files} ))

echo "xrdcp -r ${cfg_file_source} ${dest_path}/${dir_name}" >> copy_commands.txt
for i in $(seq 0 $((${max_train_files}-1))); do 
    echo "xrdcp -r ${file_list_source_train[$i]} ${file_list_dest_train[$i]}" >> copy_commands.txt
done
for i in $(seq 0 $((${max_val_files}-1))); do 
    echo "xrdcp -r ${file_list_source_val[$i]} ${file_list_dest_val[$i]}" >> copy_commands.txt
done

cat copy_commands.txt
# Start copy process with N threads
cat copy_commands.txt | xargs -I CMD --max-procs=${max_processes} bash -c CMD