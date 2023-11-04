#!/usr/bin/env bash

action() {
  if [[ "{{environment}}" == *"conda"* ]] && [[ ! "{{comp_facility}}" == *"TOpAS"* ]]; then
    echo "Will use conda inside {{conda_path}}"
    __conda_setup="$('{{conda_path}}/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
      eval "$__conda_setup"
    else
      if [ -f "{{conda_path}}/etc/profile.d/conda.sh" ]; then
        . "{{conda_path}}/etc/profile.d/conda.sh"
      else
        export PATH="{{conda_path}}/bin:$PATH"
      fi
    fi
    unset __conda_setup
  fi

  if [[ "{{comp_facility}}" == *"TOpAS"* ]]; then
    export HOME=${_CONDOR_JOB_IWD}
    tar -xzf TauMLTools*.tar.gz -C tmp
    cd tmp/TauMLTools
    source env.sh docker
    export X509_USER_PROXY=${HOME}/voms.proxy
    xrdcp -r {{data_dir}} ../
  else
    source "{{analysis_path}}/env.sh" "{{environment}}"
  fi
}
action
