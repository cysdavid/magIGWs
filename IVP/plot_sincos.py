"""
Plots sine and cosine parity components.
Must be run in serial.

Usage:
    plot_sincos.py <sim_name> [--stride=<strd>]

Options:
    --stride=<strd>  Interval of snapshots to plot [default: 1]
"""

import h5py
import pickle
from docopt import docopt
import glob
import pathlib
import json
import numpy as np
import matplotlib.pyplot as plt

import plotting_setup
plotting_setup.usetex(False)

from dedalus import public as d3
from dedalus.tools.general import natural_sort

rank = 0
size = 1

args = docopt(__doc__)
sim_name = args['<sim_name>']
strd = int(args['--stride'])
file_dir = pathlib.Path('data').absolute()
file_dir = file_dir.joinpath(sim_name)
files = glob.glob(str(file_dir.joinpath("*.h5")))
files = natural_sort(files)
params_path = pathlib.Path('params').joinpath(sim_name+'.json')

tasks = ['u','v']
fontsize = 10
interpolation = "hanning"
interpolation_stage = "rgba"

print("Getting time series...")

# Get parameters
with open(params_path) as f: 
    params_IVP = json.load(f)
Lx = params_IVP['Lx']
LZ = params_IVP['LZ']
Nx = params_IVP['Nx']
NZ = params_IVP['NZ']
Fr = params_IVP['Fr']
Gamma = params_IVP['Gamma']
kb = params_IVP['kb']
ky = params_IVP['ky']

# Account for restarts, if any
start_it_list = []
for j,filename in enumerate(files):
    with h5py.File(filename, mode='r') as file:
        it = file['scales']['iteration'][0]
        start_it_list.append(it)
        if j == len(files)-1:
            end_it = file['scales']['iteration'][-1]

# Get max values for each field at the last timestep
cmaxlist = []
f = h5py.File(files[-1])
for i,task in enumerate(tasks):
    field = f['tasks'][task][-1,...]
    cmaxlist.append(np.max(np.abs(field.real)))
x = np.array(f['tasks'][tasks[0]].dims[1]['x'])
Z = np.array(f['tasks'][tasks[0]].dims[2]['Z'])
f.close()

# Sin/cos decomposition
coords_real = d3.CartesianCoordinates('x', 'Z')
dist_real = d3.Distributor(coords_real, dtype=np.float64)
xbasis_real = d3.RealFourier(coords_real['x'], size=Nx, bounds=(0, Lx), dealias=3/2)
Zbasis_real = d3.RealFourier(coords_real['Z'], size=NZ, bounds=(0, LZ), dealias=3/2)

creal_cos = dist_real.Field(name='creal_cos', bases=(xbasis_real, Zbasis_real))
creal_sin = dist_real.Field(name='creal_sin', bases=(xbasis_real, Zbasis_real))
cimag_cos = dist_real.Field(name='cimag_cos', bases=(xbasis_real, Zbasis_real))
cimag_sin = dist_real.Field(name='cimag_sin', bases=(xbasis_real, Zbasis_real))

def sin_cos_decomp(c):
    creal_cos['g'] = c.real
    creal_sin['g'] = c.real
    cimag_cos['g'] = c.imag
    cimag_sin['g'] = c.imag

    creal_cos['c'][1::2,:] = 0
    creal_sin['c'][::2,:] = 0
    cimag_cos['c'][1::2,:] = 0
    cimag_sin['c'][::2,:] = 0

    c_cos = creal_cos['g'] + 1j*cimag_cos['g']
    c_sin = creal_sin['g'] + 1j*cimag_sin['g']

    return c_sin, c_cos

# Create output directory if needed
frame_path = pathlib.Path('frames').absolute()
output_path = frame_path.joinpath(f"{sim_name}-sincos")

print("Done.\n")
if not frame_path.exists():
    frame_path.mkdir()
if not output_path.exists():
    output_path.mkdir()

print("Plotting...")

# Import data and plot
for j,filename in enumerate(files):
    with h5py.File(filename, mode='r') as file:
        for i in range(strd*rank,file['scales']['iteration'][:].shape[0],strd*size):
            it = file['scales']['iteration'][i]
            if not ((j < len(files)-1) and (it >= start_it_list[j+1])):
                # Set up figure
                gridspec = dict(hspace=0.0, width_ratios=[1, 1, 1, 0.4, 1, 1, 1])
                fig, axs = plt.subplots(1,7,figsize = (7.52,7.52/8*5),gridspec_kw = gridspec)
                axs[3].set_visible(False)
                title_fontsize = 12

                # Plot fields
                for k,task in enumerate(tasks):

                    cmax = cmaxlist[k]
                    field = file['tasks'][task]
                    c = field[i]
                    if task == "u":
                        c_0, c_1 = sin_cos_decomp(c)
                        labels = [r"$u_{\text{IVP}}$",r"$u_{\text{IVP}}$ (sin)",r"$u_{\text{IVP}}$ (cos)"]
                    elif task == "v":
                        c_1, c_0 = sin_cos_decomp(c)
                        labels = [r"$v_{\text{IVP}}$",r"$v_{\text{IVP}}$ (cos)",r"$v_{\text{IVP}}$ (sin)"]
                    elif task == "p":
                        c_1, c_0 = sin_cos_decomp(c)
                        labels = [r"$p_{\text{IVP}}$",r"$p_{\text{IVP}}$ (cos)",r"$p_{\text{IVP}}$ (sin)"]
                    c_decomp_list = [c, c_0, c_1]

                    t = field.dims[0]['sim_time'][i]
                    write_num = field.dims[0]['write_number'][i]

                    vmin = cmax
                    for l in range(3):
                        axs[4*k+l].imshow(c_decomp_list[l].T.real,cmap='RdBu_r',vmin=(-vmin,vmin),extent=[0,Lx,0,LZ],origin="lower",interpolation=interpolation,interpolation_stage=interpolation_stage)
                        axs[4*k+l].set_title(labels[l])

                panel_label = ['($a$)','($b$)','($c$)','.','($d$)','($e$)','($f$)']
                for m in [0,1,2,4,5,6]:
                    ax = axs[m]
                    if m in [1,2,5,6]:
                        ax.set_yticks([])
                    else:
                        ax.set_ylabel('$z/L$', labelpad=-5)
                    ax.text(0.1, 1.11, panel_label[m], transform=ax.transAxes, fontsize=14, va='top', ha='right')
                    
                    ax.axhline(params_IVP['Z0'],linestyle='dotted',color='k',lw=1.5)
                    xlims = ax.get_xlim()
                    ylims = ax.get_ylim()
                    ax.fill_between([xlims[0],xlims[1]],[params_IVP['s'],params_IVP['s']], facecolor="none", hatch="//////", edgecolor="k", linewidth=0.5)
                    ax.fill_between([xlims[0],xlims[1]],[LZ-params_IVP['s'],LZ-params_IVP['s']],[LZ,LZ], facecolor="none", hatch="//////", edgecolor="k", linewidth=0.5)
                    ax.set_xticks([0,0.5,1],['0','0.5','1'])
                    ax.set_xlabel("$x/L$")
                    ax.set_aspect(4.5 * 1/0.25)

                # plt.suptitle(f"$t\\omega$ = {t:.2f}",y=0.95,fontsize=fontsize)

                savepath = output_path.joinpath('write_%06i.jpg' %(write_num))

                # print(f"Saving image {int(write_num/strd)}/{len(t_arr)}\r", end="")

                plt.savefig(str(savepath), dpi=300, bbox_inches='tight')
                plt.close()
