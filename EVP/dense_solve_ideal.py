"""
Find eigenvalues of the ideal system at uniformly spaced values of Z from Z=0 to Z=LZ using Dedalus' dense solver.

Usage:
    dense_solve_ideal.py <Nx> <NZ>
"""

from docopt import docopt
import numpy as np
import pickle
from dedalus import public as d3
import logging
logger = logging.getLogger(__name__)
import pathlib
import json

args = docopt(__doc__)

# Simulation base name
base_name = "evp"

# Numerical Parameters
Nx = int(args['<Nx>'])
NZ = int(args['<NZ>'])
dealias = 1
dtype = np.complex128

# Physical parameters
om = 1
Lx, LZ = (1., 1./4)
ky = 0
eta = 0
Fr = 0.025
Gamma = 0.1
dNdZ = 0
kb = 2*np.pi
Z_list = np.linspace(0, LZ, NZ)

# Simulation name
sim_name = base_name + f"_LZ={LZ:.3e}_ky={ky/np.pi:.2e}pi_Gamma{Gamma:.2e}_eta{eta:.1e}".replace('.','p') + f"_{Nx}_{NZ}"

# Save directories
data_path = pathlib.Path('data').absolute()
if not data_path.exists():
    data_path.mkdir()
    
subfolders_path = data_path.joinpath(sim_name)
if not subfolders_path.exists():
    subfolders_path.mkdir()

params_path = pathlib.Path('params').absolute()
if not params_path.exists():
    params_path.mkdir()
save_params_path = params_path.joinpath(sim_name+".json")


params_dict = {'om':om, 'Nx': Nx, 'NZ': NZ, 'Lx': Lx, 'LZ': LZ, 'ky': ky, 'Gamma': Gamma, 'dNdZ': dNdZ, 'kb': kb, 'eta': eta, 'Fr': Fr}
params_json = json.dumps(params_dict, indent = 4)
with open(save_params_path, "w") as outfile: 
    outfile.write(params_json)

# Create bases and domain
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.ComplexFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)
x = dist.local_grid(xbasis)

Z = dist.Field()

# Fields
## Eigenfunctions
u = dist.Field(name='u', bases=xbasis)
v = dist.Field(name='v', bases=xbasis)
p = dist.Field(name='p', bases=xbasis)
## Eigenvalue
kz2 = dist.Field()

# Substitutions
dx = lambda A: d3.Differentiate(A, xcoord)
dy = lambda A: 1j*ky*A
vAx = dist.Field(name='vAx', bases=xbasis)
vAz = dist.Field(name='vAz', bases=xbasis)
vAx['g'] = 0.5*np.sin(kb*x)
vAz['g'] = 0.5*np.cos(kb*x)
vAx = vAx * 2/np.exp(kb*Z)
vAz = vAz * 2/np.exp(kb*Z)

# Buoyancy frequency
N2 = ((2 + 2*dNdZ*Z)/(2 + dNdZ*LZ))**2

# Problem
problem = d3.EVP([u, v, p], eigenvalue=kz2, namespace=locals())
problem.add_equation("-((1j*kz2*om*p)/N2) + dx(u) + dy(v) = 0")
problem.add_equation("u*(om**2 - Gamma**2*kz2*vAz**2) + 1j*om*dx(p) + eta/Fr**2 *(1j*kz2*om*u - kz2*dx(p)) = 0")
problem.add_equation("v*(om**2 - Gamma**2*kz2*vAz**2) + 1j*om*dy(p) + eta/Fr**2 * (1j*kz2*om*v - kz2*dy(p)) = 0")

# Solver
solver = problem.build_solver()
eval_list = []
p_list = []
u_list = []
v_list = []
left_eigfxn_list = []
right_eigfxn_list = []

for it,Zi in enumerate(Z_list):
    if it>=0:
        Z['g'] = Zi
        solver.solve_dense(solver.subproblems[0], rebuild_matrices=True, left=True)

        # Get eigenvalues
        evals = solver.eigenvalues
        print(f'Z level {it}/{len(Z_list)}')
        print("[",evals[0],",",evals[1],",",evals[2],",...]")
        eval_list.append(evals[:]) 

        # Get corresponding eigenfunctions
        p_sublist = []
        u_sublist = []
        v_sublist = []
        for ev in evals:
            solver.set_state(np.argmin(np.abs(solver.eigenvalues - ev)), solver.subproblems[0].subsystems[0])
            parr = np.copy(p['g'])
            uarr = np.copy(u['g'])
            varr = np.copy(v['g'])
            p_sublist.append(parr)
            u_sublist.append(uarr)
            v_sublist.append(varr)
        p_list.append(p_sublist)
        u_list.append(u_sublist)
        v_list.append(v_sublist)

        # Save at each level in Z
        Z_save_dict = {'Z': Zi, 'kz2': evals[:], 'p': p_sublist, 'u': u_sublist, 'v': v_sublist, 'params': params_dict}
        Z_path = subfolders_path.joinpath(f"Zind-{it:04}.pickle")
        with open(Z_path, 'wb') as handle:
            pickle.dump(Z_save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save data
save_dict = {'Z': Z_list, 'kz2': eval_list, 'p': p_list, 'u': u_list, 'v': v_list, 'params': params_dict}

save_data_path = data_path.joinpath(sim_name+'.pickle')
with open(save_data_path, 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)