"""
Export last snapshot to pickle file

Usage:
    export.py <sim_name>
"""

from docopt import docopt
import pickle
import numpy as np
import pathlib
import glob
import h5py
from dedalus.tools.general import natural_sort

args = docopt(__doc__)
sim_name = args['<sim_name>']

file_dir = pathlib.Path('data').absolute()
file_dir = file_dir.joinpath(sim_name)
files = glob.glob(str(file_dir.joinpath("*.h5")))
files = natural_sort(files)

filename = files[-1]
with h5py.File(filename, mode='r') as file:
    p_field = file['tasks']['p']
    u_field = file['tasks']['u']
    v_field = file['tasks']['v']
    w_field = file['tasks']['w']
    rho_field = file['tasks']['rho']
    Bx_field = file['tasks']['Bx']
    By_field = file['tasks']['By']
    Bz_field = file['tasks']['Bz']
    p = p_field[-1]
    u = u_field[-1]
    v = v_field[-1]
    w = w_field[-1]
    rho = rho_field[-1]
    Bx = Bx_field[-1]
    By = By_field[-1]
    Bz = Bz_field[-1]
    t = np.array(p_field.dims[0]['sim_time'])
    x = np.array(p_field.dims[1]['x'])
    Z = np.array(p_field.dims[2]['Z'])

export_path = file_dir.joinpath(sim_name+'-last.hdf5')

export_dict = {'t':t, 'x':x, 'Z':Z, 'p':p, 'u':u, 'v':v, 'w':w, 'rho':rho, 'Bx':Bx, 'By':By, 'Bz':Bz}

with h5py.File(export_path, "w") as f:
    for key in list(export_dict.keys()):
        dset = f.create_dataset(key, data=export_dict[key])