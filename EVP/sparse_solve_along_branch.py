"""
Find eigenvalues branches by incrementing Z and using Dedalus' sparse solver.

Usage:
    sparse_solve_along_branch.py <target_sparse_name> <parity> <N_modes>
"""

from docopt import docopt
import numpy as np
import pickle
import matplotlib.pyplot as plt

from dedalus import public as d3

import pathlib
import json
import h5py
import sys

import eigenproblem

# Parameters
args = docopt(__doc__)
target_sparse_name = args['<target_sparse_name>']
extra_terms = ["x_diffusion_indep_eta","y_diffusion","z_diffusion"]
Nx = 1024
NZ_total = 2*1536
v_parity = args['<parity>'] # 'sin' or 'cos'
resolved_threshold = 1e-3
N_modes = int(args['<N_modes>'])
direction = 1 # 1 is upwards, -1 is downwards
match_imag_sign = True # Whether the sign of each eigenvalue's imaginary part should match that of the previous eigenvalue
hyper_eta = 0 # coefficient for hyper-diffusion, only used if "x_hyperdiffusion" in extra_terms
eta_x = 5e-5 # coefficient for diffusion in x direction, independent of coefficient in other directions
eta_override = 1e-8 
LZ_override = 1/4

# Import target eval
target_sparse_path = pathlib.Path("data").joinpath(target_sparse_name)
target_sparse_dict = pickle.load(open(target_sparse_path, 'rb'))
target_Nx = len(target_sparse_dict['v'])
params_dict = target_sparse_dict['params']
if eta_override != None:
    eta = eta_override
    target_sparse_dict['eta'] = eta
    params_dict['eta'] = eta
else:
    eta = target_sparse_dict['eta']
if LZ_override != None:
    params_dict['LZ'] = LZ_override
    target_sparse_dict['params']['LZ'] = LZ_override
params_dict["hyper_eta"] = hyper_eta
params_dict["eta_x"] = eta_x
x = np.linspace(0,1,Nx)
Z = np.linspace(0,params_dict['LZ'],NZ_total)
Z_shift = params_dict['LZ']/NZ_total

# Simulation name
base_name = "sparse_along_branch"
ky,Gamma,Fr = [params_dict[key] for key in ['ky','Gamma','Fr']]
if extra_terms != None:
    sim_name = base_name + f"_target_{target_sparse_name.split(".")[0]}_{v_parity}_ky{ky/np.pi:.2e}pi_Gamma{Gamma:.2e}_eta{eta:.1e}".replace('.','p') + f"_Fr{Fr:.1e}".replace('.','p') + f"_{Nx}_{NZ_total}+"+"+".join(extra_terms)
else:
    sim_name = base_name + f"_target_{target_sparse_name.split(".")[0]}_{v_parity}_ky{ky/np.pi:.2e}pi_Gamma{Gamma:.2e}_eta{eta:.1e}".replace('.','p') + f"_Fr{Fr:.1e}".replace('.','p') + f"_{Nx}_{NZ_total}"

# Save directories
data_path = pathlib.Path('data').absolute()
if not data_path.exists():
    data_path.mkdir()

h5_path = data_path.joinpath(f"{sim_name}.hdf5")
# Check if h5 file exists
if h5_path.is_file():
    print(f"{str(h5_path)} exists. Overwrite (y/n)?")
    if input() != 'y':
        sys.exit()
# Delete if existing
h5_path.unlink(missing_ok=True)

params_path = pathlib.Path('params').absolute()
if not params_path.exists():
    params_path.mkdir()
save_params_path = params_path.joinpath(sim_name+".json")

full_params_dict = {'params':params_dict, 'target_sparse_name':target_sparse_name, 'extra_terms':extra_terms, 'Nx':Nx, 'NZ_total':NZ_total, 'v_parity':v_parity, 'resolved_threshold':resolved_threshold, 'N_modes':N_modes, 'direction':direction}
full_params_json = json.dumps(full_params_dict, indent = 4)
with open(save_params_path, "w") as outfile: 
    outfile.write(full_params_json)

# Define sparse solve function
def sparse_solve(eval, Nx, Z_level, params_dict, eta, N_modes=1):
    params_dict['eta'] = eta
    eigprob = eigenproblem.eigenproblem(Z_value=Z_level,params_dict=params_dict,Nx=Nx,extra_terms=extra_terms)

    # Solver
    solver = eigprob.solver
    solver.solve_sparse(solver.subproblems[0], N=N_modes, target=eval, rebuild_matrices=True)

    evals = solver.eigenvalues
    evals = evals[np.argsort(evals.real)]
    u_list = np.zeros((N_modes,Nx),dtype=np.complex128)
    v_list = np.zeros((N_modes,Nx),dtype=np.complex128)
    w_list = np.zeros((N_modes,Nx),dtype=np.complex128)
    p_list = np.zeros((N_modes,Nx),dtype=np.complex128)
    bx_list = np.zeros((N_modes,Nx),dtype=np.complex128)
    by_list = np.zeros((N_modes,Nx),dtype=np.complex128)
    vcoeffs_list = np.zeros((N_modes,Nx),dtype=np.complex128)

    for i,ev in enumerate(evals):
        solver.set_state(np.argmin(np.abs(solver.eigenvalues - ev)), solver.subproblems[0].subsystems[0])

        u = [var for var in solver.problem.variables if var.name=='u'][0]
        v = [var for var in solver.problem.variables if var.name=='v'][0]
        w = [var for var in solver.problem.variables if var.name=='w'][0]
        p = [var for var in solver.problem.variables if var.name=='p'][0]
        bx = [var for var in solver.problem.variables if var.name=='bx'][0]
        by = [var for var in solver.problem.variables if var.name=='by'][0]

        u_list[i,:] = u['g']
        v_list[i,:] = v['g']
        w_list[i,:] = w['g']
        p_list[i,:] = p['g']
        bx_list[i,:] = bx['g']
        by_list[i,:] = by['g']
        vcoeffs_list[i,:] = v['c']

    x = eigprob.x

    return solver,evals,u_list,v_list,w_list,p_list,bx_list,by_list,vcoeffs_list,x

# Set target eigenvalue and starting height
target_eval = target_sparse_dict['kz']
start_Zidx = np.argmin(np.abs(target_sparse_dict['Z'] - Z))
target_Zidx = start_Zidx

# Set up loop
eval_list = []
Z_list = []
full_eval_list = []
full_Z_list = []
count = 0
penult_idx = int(0.5*(1+direction)*len(Z) - direction)
field_names = ["u","v","p","w","bx","by"]

# Main loop
with h5py.File(h5_path, "a") as f:
    while target_Zidx != penult_idx+direction*1:
     
        N_modes_to_use = N_modes
        count += 1
        target_Z = Z[target_Zidx]
        print(f"Performing dense solve at height {abs(start_Zidx-target_Zidx)}/{len(Z[start_Zidx::direction]) - 1}, Z = {target_Z}.")
        solver,evals,u_sublist,v_sublist,w_sublist,p_sublist,bx_sublist,by_sublist,vcoeffs_sublist,x = sparse_solve(target_eval, Nx, target_Z, params_dict, eta, N_modes=N_modes_to_use)

        # Make h5 dsets
        if target_Zidx == start_Zidx:
            for field_nm in field_names:
                f.create_dataset(field_nm, (Nx,NZ_total), dtype='complex128', compression="gzip", compression_opts=9)
                f[field_nm][...] = np.nan*np.ones((Nx,NZ_total),dtype=np.complex128)
            f.create_dataset("Z", (NZ_total), dtype='complex128', compression="gzip", compression_opts=9)
            f["Z"][...] = np.nan*np.ones(NZ_total,dtype=np.complex128)
            f.create_dataset("kz", (NZ_total), dtype='complex128', compression="gzip", compression_opts=9)
            f["kz"][...] = np.nan*np.ones(NZ_total,dtype=np.complex128)
            f.create_dataset("start_Zidx", 1, dtype='int')
            f["start_Zidx"][0] = start_Zidx
            f.create_dataset("end_Zidx", 1, dtype='int')

        no_mode_count = 0
        candidate_eval_list = []
        candidate_eval_ids = []
        for i in range(len(evals)):
            # Check if v is of sine parity/u is of cosine parity
            vsin_flag = np.sum(np.abs(vcoeffs_sublist[i][1:] - vcoeffs_sublist[i][:0:-1])**2) > np.sum(np.abs(vcoeffs_sublist[i][1:] + vcoeffs_sublist[i][:0:-1])**2)
            resolved_flag = np.mean(np.abs(vcoeffs_sublist[i][int(Nx*0.45):int(Nx*0.5)]))/np.mean(np.abs(vcoeffs_sublist[i][0:int(Nx*0.05)])) < resolved_threshold
            if match_imag_sign:
                match_imag_sign_flag = np.sign(evals[i].imag) == np.sign(target_eval.imag)
            else:
                match_imag_sign_flag = True
            
            if v_parity == 'sin':
                if vsin_flag&resolved_flag&match_imag_sign_flag:
                    candidate_eval_list.append(evals[i])
                    candidate_eval_ids.append(i)
                    full_eval_list.append(evals[i])
                    full_Z_list.append(target_Z)
                    print("Mode found.")
                else:
                    no_mode_count += 1

            else:
                if (vsin_flag != True)&resolved_flag&match_imag_sign_flag:
                    candidate_eval_list.append(evals[i])
                    candidate_eval_ids.append(i)
                    full_eval_list.append(evals[i])
                    full_Z_list.append(target_Z)
                    print("Mode found.")
                else:
                    no_mode_count += 1
            
        if no_mode_count ==  len(evals):
            print("No resolved modes found of the desired parity.")
            break
        else:
            target_eval_idx = candidate_eval_ids[np.argmin(np.abs(target_eval - np.array(candidate_eval_list)))]
            target_eval = evals[target_eval_idx]

            f["u"][:,target_Zidx] = u_sublist[target_eval_idx]
            f["v"][:,target_Zidx] = v_sublist[target_eval_idx]
            f["p"][:,target_Zidx] = p_sublist[target_eval_idx]
            f["w"][:,target_Zidx] = w_sublist[target_eval_idx]
            f["bx"][:,target_Zidx] = bx_sublist[target_eval_idx]
            f["by"][:,target_Zidx] = by_sublist[target_eval_idx]
            f["Z"][target_Zidx] = target_Z
            f["kz"][target_Zidx] = target_eval
            f["end_Zidx"][0] = target_Zidx

            print(f"kz = {target_eval}")

        target_Zidx += direction*1