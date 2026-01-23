"""
Find eigenvalues at specified value of Z using Dedalus' sparse solver.
Target eigenvalues may be chosen using a GUI, unless a target index is supplied.

Usage:
    sparse_solve_fixed_Z.py [--targetindex=<i>]

Options:
    --targetindex=<i> Index of the eigenvalue to use as the target for the sparse solver (i=0 corresponds to eigenvalue with smallest real part). [default: None]
"""

from docopt import docopt
import numpy as np
from mpi4py import MPI
import pickle
import time
import matplotlib.pyplot as plt

from dedalus import public as d3

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.parallel import Sync
import pathlib
import json
import copy

import eigenproblem

# Determine whether to use interactive mode
args = docopt(__doc__)
interactive = False
try:
    target_index = int(args['--targetindex'])
except:
    interactive = True
    import sys
    sys.path.append(str(pathlib.Path("..").joinpath("packages")))
    from lasso_selector import SelectFromCollection

# Extra terms
extra_terms = None # ["x_diffusion","y_diffusion","z_diffusion","dx_ycomp","dx_xcomp","vAx_without_dx"]

# Fixed parameters
dealias = 3/2
dtype = np.complex128
om = 1
Lx, LZ = (1., 1./4)
ky = 2*np.pi
Fr = 0.025
Gamma = 0.1
dNdZ = 0
kb = 2*np.pi
Z_level = 0
params = [om,Lx,LZ,ky,Fr,Gamma,dNdZ,kb]
params_dict = {'om':om, 'Lx': Lx, 'LZ': LZ, 'ky': ky, 'Gamma': Gamma, 'dNdZ': dNdZ, 'kb': kb, 'Fr': Fr}

# Save directories
data_path = pathlib.Path('data').absolute()
if not data_path.exists():
    data_path.mkdir()

def dense_solve(Nx, Z_level, params_dict, eta):
    params_dict['eta'] = eta
    eigprob = eigenproblem.eigenproblem(Z_value=Z_level,params_dict=params_dict,Nx=Nx,extra_terms=extra_terms)

    # Solver
    solver = eigprob.solver
    solver.solve_dense(solver.subproblems[0], rebuild_matrices=True, left=True)

    # Get finite eigenvalues
    evals = solver.eigenvalues
    evals = evals[np.isfinite(solver.eigenvalues)]
    evals = evals[((evals).real) > 0]
    evals = evals[np.argsort((evals).real)]

    return evals

def sparse_solve(eval, Nx, Z_level, params_dict, eta):
    params_dict['eta'] = eta
    eigprob = eigenproblem.eigenproblem(Z_value=Z_level,params_dict=params_dict,Nx=Nx,extra_terms=extra_terms)

    # Solver
    solver = eigprob.solver
    solver.solve_sparse(solver.subproblems[0], N=1, target=eval, rebuild_matrices=True)

    solver.set_state(0, solver.subproblems[0].subsystems[0])

    u = [var for var in eigprob.solver.problem.variables if var.name=='u'][0]
    v = [var for var in eigprob.solver.problem.variables if var.name=='v'][0]
    w = [var for var in eigprob.solver.problem.variables if var.name=='w'][0]
    p = [var for var in eigprob.solver.problem.variables if var.name=='p'][0]
    bx = [var for var in eigprob.solver.problem.variables if var.name=='bx'][0]
    by = [var for var in eigprob.solver.problem.variables if var.name=='by'][0]
    x = eigprob.x

    return solver,u,v,w,p,bx,by,x

def plot_unresevals(evals,unres_evals,target_eval):
    plt.scatter((evals).real,(evals).imag,color='tab:green',alpha=0.5)
    plt.scatter((unres_evals).real,(unres_evals).imag,color='tab:red',alpha=0.5)
    plt.scatter((target_eval).real,(target_eval).imag,color='tab:blue')
    xlim_r = max((target_eval).real,((evals).real).max())
    ylim_t = max((target_eval).imag,((evals).imag).max())
    ylim_b = min((target_eval).imag,((evals).imag).min())
    plt.xlim(((evals).real).min(), xlim_r)
    plt.ylim(ylim_b,ylim_t)
    plt.xlabel('$\\Re\\{k_z\\}$')
    plt.ylabel('$\\Im\\{k_z\\}$')

def plot_evals(evals,target_eval):
    plt.scatter((evals).real,(evals).imag,color='tab:blue',alpha=0.5)
    plt.scatter((target_eval).real,(target_eval).imag,color='k')
    xlim_r = max((target_eval).real,np.quantile((evals).real,0.99))
    ylim_t = max((target_eval).imag,np.quantile((evals).imag,0.99))
    ylim_b = min((target_eval).imag,np.quantile((evals).imag,0.01))
    plt.xlim(((evals).real).min(), xlim_r)
    plt.ylim(ylim_b,ylim_t)
    plt.xlabel('$\\Re\\{k_z\\}$')
    plt.ylabel('$\\Im\\{k_z\\}$')

# Get resolved eigenvalues
Nx_lres = 96
Nx_hres = 128

eta = 0
evals_lres = dense_solve(Nx_lres, Z_level, params_dict, eta)
evals_hres = dense_solve(Nx_hres, Z_level, params_dict, eta)

evals = []
unres_evals = []
for eval in evals_lres:
    if np.min(np.abs(eval - evals_hres))/np.abs(eval) < 1e-6:
        evals.append(eval)
    else:
        unres_evals.append(eval)
evals = np.array(evals)
unres_evals = np.array(unres_evals)
logger.info(evals)

# Use GUI to select target eigenvalue in interactive mode
if interactive:
    # Plot evals
    fig, ax = plt.subplots()
    all_evals = np.concatenate((evals,unres_evals))
    sc_color = np.zeros(len(all_evals)).astype(str)
    sc_color[0:len(evals)] = 'tab:green'
    sc_color[len(evals):len(evals)+len(unres_evals)] = 'tab:red'
    sc_pts = ax.scatter((all_evals).real,(all_evals).imag,color=sc_color)
    ax.set_xlabel('$\\Re\\{k_z\\}$')
    ax.set_ylabel('$\\Im\\{k_z\\}$')

    # GUI for selecting target eigenvalue
    selector = SelectFromCollection(ax, sc_pts)
    def accept(event):
            if event.key == "enter":
                print("Selected points:")
                print(selector.xys[selector.ind])
                selector.disconnect()
                ax.set_title("")
                fig.canvas.draw()
                plt.close()

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")
    plt.show(block=True)

    selected_val = selector.xys[selector.ind][0][0]+1j*selector.xys[selector.ind][0][1]
    selected_ind = np.argmin(np.abs(evals_lres-selected_val))

    # Select target eigenvalue using GUI input or manual index entry
    print("Use selected eigenvalue as target? (y/n)")
    yn = input()
    if yn == "n":
        print("Use index? (y/n)")
        yn_2 = input()
        if yn == "y":
            print("Enter eigenvalue index to use as target:")
            target_ind = int(input())
            target_eval = evals_lres[target_ind]
        else:
            print("Enter eigenvalue 'x+yj' to use as target:")
            target_eval = complex(input())
            target_ind = 'None'
    else:
        target_ind = selected_ind
        target_eval = evals_lres[target_ind]

# Otherwise, use supplied target index
else:
    target_ind = target_index
    target_eval = evals_lres[target_ind]

print("target k_z: ", target_eval)
print("target k_z index: ", target_ind) # index of output of dense_solve, sorted by real part (ascending)

# Sparse solve
size=128
solver,u,v,w,p,bx,by,x = sparse_solve(target_eval, size, Z_level, params_dict, eta)
logger.info(solver.eigenvalues[0])
if interactive:
    plt.ion()
    fig,axs = plt.subplots(1,2)
    axs[0].plot(x,v['g'].real)
    axs[0].set_title('v')
    axs[1].plot(x,u['g'].real)
    axs[1].set_title('u')
    plt.pause(3)

size=256
solver,u,v,w,p,bx,by,x = sparse_solve(solver.eigenvalues[0], size, Z_level, params_dict, eta)
logger.info(solver.eigenvalues[0])
if interactive:
    plt.ion()
    axs[0].plot(x,v['g'].real)
    axs[0].set_title('v')
    axs[1].plot(x,u['g'].real)
    axs[1].set_title('u')
    plt.pause(2)

size=512
solver,u,v,w,p,bx,by,x = sparse_solve(solver.eigenvalues[0], size, Z_level, params_dict, eta)
logger.info(solver.eigenvalues[0])
if interactive:
    plt.ion()
    axs[0].plot(x,v['g'].real)
    axs[0].set_title('v')
    axs[1].plot(x,u['g'].real)
    axs[1].set_title('u')
    plt.pause(2)

size=1024
solver,u,v,w,p,bx,by,x = sparse_solve(solver.eigenvalues[0], size, Z_level, params_dict, eta)
logger.info(solver.eigenvalues[0])
if interactive:
    plt.ion()
    axs[0].plot(x,v['g'].real)
    axs[0].set_title('v')
    axs[1].plot(x,u['g'].real)
    axs[1].set_title('u')
    plt.pause(2)

size=1152
solver,u,v,w,p,bx,by,x = sparse_solve(solver.eigenvalues[0], size, Z_level, params_dict, eta)
logger.info(solver.eigenvalues[0])
if interactive:
    plt.ion()
    axs[0].plot(x,v['g'].real)
    axs[0].set_title('v')
    axs[1].plot(x,u['g'].real)
    axs[1].set_title('u')
    plt.pause(2)

size=1536
solver,u,v,w,p,bx,by,x = sparse_solve(solver.eigenvalues[0], size, Z_level, params_dict, eta)
logger.info(solver.eigenvalues[0])
if interactive:
    plt.ion()
    axs[0].plot(x,v['g'].real)
    axs[0].set_title('v')
    axs[1].plot(x,u['g'].real)
    axs[1].set_title('u')
    plt.pause(2)

size=1728
solver,u,v,w,p,bx,by,x = sparse_solve(solver.eigenvalues[0], size, Z_level, params_dict, eta)
logger.info(solver.eigenvalues[0])
if interactive:
    plt.ion()
    axs[0].plot(x,v['g'].real)
    axs[0].set_title('v')
    axs[1].plot(x,u['g'].real)
    axs[1].set_title('u')
    plt.pause(2)

size=2048
solver,u,v,w,p,bx,by,x = sparse_solve(solver.eigenvalues[0], size, Z_level, params_dict, eta)
logger.info(solver.eigenvalues[0])
if interactive:
    plt.ion()
    axs[0].plot(x,v['g'].real)
    axs[0].set_title('v')
    axs[1].plot(x,u['g'].real)
    axs[1].set_title('u')
    plt.pause(2)

    plt.show(block=True)

print("Done.")

if interactive:
    axs[0].plot(x,v['g'].real)
    axs[0].set_title('v')
    axs[1].plot(x,u['g'].real)
    axs[1].set_title('u')
    plt.pause(2)
    plt.show(block=True)

data = {'params':params_dict, 'eta': eta, 'Z': Z_level, 'x': x, 'u': u['g'], 'v': v['g'], 'w': w['g'], 'p': p['g'], 'bx': bx['g'], 'by': by['g'], 'kz': solver.eigenvalues[0], 'evals_lres': evals_lres, 'unres_evals_lres':unres_evals, 'Nx_lres': Nx_lres, 'Nx_hres' : Nx_hres, 'target_ind' : target_ind}

print("Saving data...")
save_path = data_path.joinpath(f"eigfxn_exp_fulldiff_ky{str(ky/np.pi).replace('.','p')}pi_Gamma_{str(Gamma).replace('.','p')}_eta{eta}_Z{str(Z_level).replace('.','p')}_ind{target_ind}.pkl")
pickle.dump(data, open(str(save_path), "wb" ))
print("Done.")