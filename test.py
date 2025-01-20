

import dask
import dask.dataframe as dd
import uproot
# import tensorflow as tf
import numpy as np

# Function to load a chunk of ROOT data
def load_chunk(file_name, tree_name, branches, entry_start, entry_stop):
    with uproot.open(file_name) as file:
        tree = file[tree_name]
        arrays = tree.arrays(branches, entry_start=entry_start, entry_stop=entry_stop, library="np")
    return arrays

# Create Dask DataFrame
def create_dask_dataframe(file_name, tree_name, branches, chunk_size=10000):
    with uproot.open(file_name) as file:
        tree = file[tree_name]
        total_entries = tree.num_entries

    chunks = []
    for start in range(0, total_entries, chunk_size):
        stop = min(start + chunk_size, total_entries)
        delayed_chunk = dask.delayed(load_chunk)(file_name, tree_name, branches, start, stop)
        chunks.append(delayed_chunk)
    print(chunks)

    ddf = dd.from_delayed([dask.delayed(chunk) for chunk in chunks])
    return ddf


# Parameters for the ROOT file
file_name = "ShuffleMergeSpectral_0.root"
tree_name = "taus"
branches = ['tau_pt', 'tau_eta', 'tau_mass']  # Example branches
chunk_size = 50000

# Create a Dask DataFrame
ddf = create_dask_dataframe(file_name, tree_name, branches, chunk_size)


# Generator function to yield data from Dask in batches
def dask_generator(ddf, x_columns, y_column):
    for batch in ddf.partitions:
        # Compute the batch (lazy evaluation)
        data = batch.compute()
        X = np.array(data[x_columns], dtype=np.float32)
        y = np.array(data[y_column], dtype=np.float32)
        yield X, y
        
# dataset = tf.data.Dataset.from_generator(
#     lambda: dask_generator(ddf, x_columns, y_column),
#     output_signature=output_signature
# )
        
iterer = iter(dask_generator(ddf, branches[:-1],branches[-1:]))
a, b = next(iterer)
print(a,b)