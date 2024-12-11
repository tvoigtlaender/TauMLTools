from glob import glob
import json
import os
from remote_glob import remote_glob


def check_match_and_save(path_local, path_remote, output_file):
    absolute_output = os.path.abspath(output_file)
    assert path_local.count("*")==1 and path_remote.count("*")==1, "Unsuited input path provided"


    files_local = sorted(glob(path_local))
    file_names = [n.split("/")[-3] for n in files_local]
    expected_files_remote = [path_remote.replace("*", n) for n in file_names]
    files_remote = remote_glob(path_remote)
    matching_files = sorted(list(set(expected_files_remote) & set(files_remote)))

    assert len(matching_files)==len(expected_files_remote), "Expected and present files not matching"

    json_data = {l: r for l,r in zip(files_local, expected_files_remote)}

    print(f"Writing filemap to {absolute_output}")
    with open(absolute_output, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

# path_local = '/work/tvoigtlaender/checkpointing/TauMLTools/data/Training/2024_aug_short_recompute_V1.2/files/mlruns/220469896414000777/634aeb7306184e7eb82973e236ea967a/artifacts/predictions/GluGluHToTauTau_M125/RootToTF/*/predictions.h5'
# path_remote = 'root://eosuser.cern.ch//eos/cms/store/group/phys_tau/TauML/prod_2018_v2/full_tuples/GluGluHToTauTau_M125/*.root'
# output_file = "pred_input_filemap.json"