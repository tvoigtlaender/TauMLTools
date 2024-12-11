#!/usr/bin/env bash
# set -x
action() {
  if [[ "{{environment}}" == *"conda"* ]] && [[ ! "{{comp_facility}}" == *"TOpAS"* ]]; then
    echo "Will use conda inside {{conda_path}}"
    __conda_setup="$('{{conda_path}}/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
      eval "$__conda_setup"
      conda activate tau-ml
    else
      if [ -f "{{conda_path}}/etc/profile.d/conda.sh" ]; then
        . "{{conda_path}}/etc/profile.d/conda.sh"
      else
        export PATH="{{conda_path}}/bin:$PATH"
      fi
    fi
    unset __conda_setup
  fi

  if [[ "{{comp_facility}}" == *"ETP"* ]]; then
    export HOME=${_CONDOR_JOB_IWD}
    tar -xzf TauMLTools*.tar.gz -C tmp
    cd tmp/TauMLTools
    source env.sh docker
    #Copy in data with 64 subprocesses
    bash copy_in.sh {{data_dir_train}} ../ 64 {{max_files_train}} train
    bash copy_in.sh {{data_dir_val}} ../ 64 {{max_files_val}} val
    #export X509_USER_PROXY=${HOME}/voms.proxy
    # git clone https://gitlab.etp.kit.edu/jeppelt/checkpointer.git
    # cd checkpointer
    # pip install  -e .
    # #export PYTHONPATH=${PYTHONPATH}:$(pwd)
    # cd -
  else
    source "{{analysis_path}}/env.sh" "{{environment}}"
  fi
}
action
