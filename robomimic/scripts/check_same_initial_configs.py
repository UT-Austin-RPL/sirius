import h5py
import sys
import numpy as np 

dataset_path = sys.argv[1]

f = h5py.File(dataset_path, "r")
data = f["data"]
demos = list(data.keys())

initial_state_lst = []

def same_state(s1, s2):
    return np.all(np.equal(s1, s2))
    
for d in demos:
    this_s = data[d]["states"][()][0]
    for s in initial_state_lst:
        if same_state(s, this_s):
            print("same state")
    initial_state_lst.append(this_s)

#########################################

sec_dataset_path = sys.argv[2]
f = h5py.File(sec_dataset_path, "r")
sec_data = f["data"]
sec_demos = list(sec_data.keys())

for d in sec_demos:
    this_s = sec_data[d]["states"][()][0]
    for s in initial_state_lst:
        if same_state(s, this_s):
            print("same state")
    initial_state_lst.append(this_s)


