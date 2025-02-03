import numpy as np
from mass_copy import remote_glob
import yaml

path = "root://cmsdcache-kit-disk.gridka.de//store/user/tvoigtlaender/TAUML/january_2025_V1/ShuffleMergeSpectral_*"

all_files = remote_glob(path)


rng = np.random.default_rng(seed=1234)
rng.shuffle(all_files)

# 450 files
train_files = all_files[:450]
# 49 files
val_files = all_files[450:]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

num_per_train = 50
num_runs = 10
used_train_files_ensemble = {"val": val_files, "train": {}}
for run_i in range(num_runs):
    rng.shuffle(train_files)
    tmp_dict = {f"chunk_{chunk_i}": files for chunk_i, files in enumerate(chunks(train_files, num_per_train))}
    used_train_files_ensemble["train"][f"run_{run_i}"] = tmp_dict
with open("used_train_files_ensemble.yaml", "w") as f:
    yaml.dump(used_train_files_ensemble, f)

used_train_files_scaling = {"val": val_files, "train": {}}
run_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 450]
for run_n in run_sizes:
    tmp = rng.choice(train_files, run_n, replace=False)
    used_train_files_scaling["train"][f"{run_n}_files"] = tmp.tolist()
with open("used_train_files_scaling.yaml", "w") as f:
    yaml.dump(used_train_files_scaling, f)
