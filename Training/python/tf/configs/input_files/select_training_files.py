import numpy as np
from mass_copy import remote_glob
import yaml

path = "root://cmsdcache-kit-disk.gridka.de//store/user/tvoigtlaender/TAUML/january_2025_V2/ShuffleMergeSpectral_*"
cfg_file = "root://cmsdcache-kit-disk.gridka.de//store/user/tvoigtlaender/TAUML/january_2025_V2/cfg.yaml"

all_files = remote_glob(path)


rng = np.random.default_rng(seed=1234)
rng.shuffle(all_files)

cutoff = 450

# 450 files
train_files = all_files[:cutoff]
# 49 files
val_files = all_files[cutoff:]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

num_per_train = 50
num_runs = 10
for run_i in range(num_runs):
    rng.shuffle(train_files)
    for chunk_i, files in enumerate(chunks(train_files, num_per_train)):
        with open(f"used_train_files_ensemble_run{run_i}_chunk{chunk_i}.yaml", "w") as f:
            yaml.dump({"cfg": cfg_file, "val": val_files, "train": files}, f)

used_train_files_scaling = {"val": val_files, "train": {}}
run_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 450]
for run_n in run_sizes:
    tmp = rng.choice(train_files, run_n, replace=False)
    with open(f"used_train_files_scaling_{run_n}files.yaml", "w") as f:
        yaml.dump({"cfg": cfg_file, "val": val_files, "train": tmp.tolist()}, f)

with open(f"all_train_files.yaml", "w") as f:
    yaml.dump({"cfg": cfg_file, "val": val_files, "train": train_files}, f)
        

