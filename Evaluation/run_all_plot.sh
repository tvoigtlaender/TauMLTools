#!/bin/bash
# run_id=$1
# Script to run common settings for tau with GluGluHToTauTau_M125, e/mu with DYJetsToLL_M-50-amcatnloFXFX_ext2 and jet with TTToSemiLeptonic samples
# Involved mlflow runs should be set manually in the respective config files

# Evaluate all prediction samples

# echo python evaluate_performance.py vs_type=e dataset_alias=tau_ggH_e_DY run_id=${run_id}
# python evaluate_performance.py vs_type=e dataset_alias=tau_ggH_e_DY run_id=${run_id}
# echo python evaluate_performance.py vs_type=mu dataset_alias=tau_ggH_mu_DY run_id=${run_id}
# python evaluate_performance.py vs_type=mu dataset_alias=tau_ggH_mu_DY run_id=${run_id}
# echo python evaluate_performance.py vs_type=jet dataset_alias=tau_ggH_jet_TT run_id=${run_id}
# python evaluate_performance.py vs_type=jet dataset_alias=tau_ggH_jet_TT run_id=${run_id}


# Plot all performances
python plot_roc.py vs_type=e dataset_alias=tau_ggH_e_DY
python plot_roc.py vs_type=mu dataset_alias=tau_ggH_mu_DY
python plot_roc.py vs_type=jet dataset_alias=tau_ggH_jet_TT

