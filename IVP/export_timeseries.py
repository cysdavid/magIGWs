"""
Export 1D timeseries of all fields to pickle file

Usage:
    export_timeseries.py <sim_name>
"""

from docopt import docopt
import pickle
import numpy as np
import pathlib
import glob
import h5py
from dedalus.tools.general import natural_sort

x_value = 0.375
Z_value = 0.048

args = docopt(__doc__)
sim_name = args['<sim_name>']

file_dir = pathlib.Path('data').absolute()
file_dir = file_dir.joinpath(sim_name)
files = glob.glob(str(file_dir.joinpath("*.h5")))
files = natural_sort(files)

# Account for restarts, if any
start_it_list = []
for j,filename in enumerate(files):
    with h5py.File(filename, mode='r') as file:
        it = file['scales']['iteration'][0]
        start_it_list.append(it)
        if j == len(files)-1:
            end_it = file['scales']['iteration'][-1]

# Get x and Z
filename = files[0]
with h5py.File(filename, mode='r') as file:
    p_field = file['tasks']['p']
    x = np.array(p_field.dims[1]['x'])
    Z = np.array(p_field.dims[2]['Z'])

x_idx = np.argmin(np.abs(x-x_value))
Z_idx = np.argmin(np.abs(Z-Z_value))

t_series = []
p_series = []
u_series = []
v_series = []
w_series = []
rho_series = []
Bx_series = []
By_series = []
Bz_series = []
KE_series = []
ME_series = []
E_series = []

for j,filename in enumerate(files):
    with h5py.File(filename, mode='r') as file:
        for i in range(0,file['tasks']['p'].shape[0]):
            it = file['scales']['iteration'][i]
            if not ((j < len(files)-1) and (it >= start_it_list[j+1])):
                t_series.append(file['tasks']['p'].dims[0]['sim_time'][i])
                p_series.append(file['tasks']['p'][i,x_idx,Z_idx])
                u_series.append(file['tasks']['u'][i,x_idx,Z_idx])
                v_series.append(file['tasks']['v'][i,x_idx,Z_idx])
                w_series.append(file['tasks']['w'][i,x_idx,Z_idx])
                rho_series.append(file['tasks']['rho'][i,x_idx,Z_idx])
                Bx_series.append(file['tasks']['Bx'][i,x_idx,Z_idx])
                By_series.append(file['tasks']['By'][i,x_idx,Z_idx])
                Bz_series.append(file['tasks']['Bz'][i,x_idx,Z_idx])
                KE_series.append(file['tasks']['KE'][i,0].real)
                ME_series.append(file['tasks']['ME'][i,0].real)
                E_series.append(file['tasks']['E'][i,0].real)

t_series = np.array(t_series)
p_series = np.array(p_series)
u_series = np.array(u_series)
v_series = np.array(v_series)
w_series = np.array(w_series)
rho_series = np.array(rho_series)
Bx_series = np.array(Bx_series)
By_series = np.array(By_series)
Bz_series = np.array(Bz_series)
KE_series = np.array(KE_series)
ME_series = np.array(ME_series)
E_series = np.array(E_series)

# Export data
fname = sim_name+f"-timeseries_x={str(x_value).replace('.','p')}_Z={str(Z_value).replace('.','p')}.hdf5"
export_path = file_dir.joinpath(fname)

export_dict = {'x':x_value, 'Z':Z_value, 't':t_series, 'p':p_series, 'u':u_series, 'v':v_series, 'w':w_series, 'rho':rho_series, 'Bx':Bx_series, 'By':By_series, 'Bz':Bz_series, 
               'KE':KE_series, 'ME':ME_series, 'E':E_series}

with h5py.File(export_path, "w") as f:
    for key in list(export_dict.keys()):
        dset = f.create_dataset(key, data=export_dict[key])