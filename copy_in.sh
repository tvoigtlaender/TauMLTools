#!/bin/bash
set -x
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
mkdir -p ${dest_path}/${dir_name}/${data_type}/

# Set internal dir structure
dir_source=${source_path}/${data_type}/

# Create arrays of source and destination paths
shards=($(gfal-ls ${dir_source}))
dest_list=($(printf "%0.s${dest_path}/${dir_name}/${data_type}/ " $(seq 1 ${#shards[@]})))
file_list_source=( "${shards[@]/#/${dir_source}}" )

# Copy cfg file from first in list of train files
if [[ "${data_type}" == "train" ]]; then
    cfg_file_source=${file_list_source[0]}/cfg.yaml
    echo "xrdcp -r ${cfg_file_source} ${dest_path}/${dir_name}" >> copy_commands.txt
fi

# Write copy commands to file
max_files=$(( ${#file_list_source[@]} < ${max_files} ? ${#file_list_source[@]} : ${max_files} ))

for i in $(seq 0 $((${max_files}-1))); do 
    echo "xrdcp -r ${file_list_source[$i]} ${dest_list[$i]}" >> copy_commands.txt
done

# cat copy_commands.txt
# Start copy process with N threads
cat copy_commands.txt | xargs -I CMD --max-procs=${max_processes} bash -c CMD