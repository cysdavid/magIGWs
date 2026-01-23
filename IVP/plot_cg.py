"""
Plots sine and cosine parity components with horizontal lines
advected at the corresponding WKB group velocities.
Must be run in serial.

Usage:
    plot_cg.py <sim_name> [--stride=<strd>]

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
sans_name = plotting_setup.get_font("Avenir","sans")
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
fontsize = 12
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

# Load data from WKB analysis
wkb_path = glob.glob(str(file_dir.joinpath("*-wkb.pickle")))[0]
with open(wkb_path,'rb') as f:
    wkb_dict = pickle.load(f)
mode_ids = wkb_dict['mode_ids']
kz_stack = wkb_dict['kz']
t_arr = wkb_dict['t']
kz_char_stack = wkb_dict['kz_char']
Z_char_stack = wkb_dict['Z_char']
Z_IVP = wkb_dict['Z']
Z_transition = wkb_dict['Z_transition']
t_transition = wkb_dict['t_transition']
mode_char_stack = wkb_dict['mode_char']
color_dict = wkb_dict['color']
linestyle_dict = wkb_dict['linestyle']

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
output_path = frame_path.joinpath(f"{sim_name}-wkb")

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
                gridspec = dict(hspace=0.0, width_ratios=[2.2, 0.8, 1, 1, 0.8, 1, 1])
                fig, axs = plt.subplots(1,7,figsize = (8.6/6.8*7.52,8.6/6.8*7.52/8*4.25),gridspec_kw = gridspec)
                axs[1].set_visible(False)
                axs[4].set_visible(False)

                for ax in axs:
                    ax.tick_params(axis='both', which='both', labelsize=9*fontsize//10)
                    ax.axhline(params_IVP['Z0'],linestyle='dotted',color='k',lw=1.5)

                # Dispersion relation
                ## Plot eigenvalues
                for nm in ['IGW-0', 'evan-0', 'IGW-1', 'evan-1', 'SM-0', 'AW-0', 'SM-1', 'SM-AW-1', 'AW-1']:
                    if 'evan' in nm:
                        label = nm.replace('evan','evan.')
                    else:
                        label=nm
                    if nm == 'SM-AW-1':
                        linestyle = (3, (2.5, 1.5))
                    else:
                        linestyle = linestyle_dict[nm]
                    axs[0].plot(1/Fr*kz_stack[...,mode_ids[nm]].real,Z_IVP,color=color_dict[nm],linestyle=linestyle,linewidth=1.5)

                ## Plot Alfven continuum
                alfven_bdry = 1/Gamma*np.exp(kb*Z_IVP)
                axs[0].fill_betweenx(Z_IVP,1/Fr*alfven_bdry,1/Fr*100*np.ones(len(Z_IVP)),color='lightgray',zorder=0)

                ## Plot pure IGW wave kz
                kx = 2*np.pi
                kz_pure_IGW = np.sqrt(kx**2+ky**2)
                axs[0].axvline(1/Fr*kz_pure_IGW,ymin=0.,ymax=1,color='dimgray',linewidth=1,linestyle='dashdot',zorder=0,alpha=0.5)

                ## Plot SM-1 cutoff height for ky=0
                ky0_cutoff_Z = np.log(6*np.pi*Gamma)/kb
                axs[0].axhline(ky0_cutoff_Z,xmin=0.7,color='dimgray',linewidth=1,linestyle='--')

                ## Limits and labels
                axs[0].text(650,0.025,"ALFVÉN WAVES",rotation=45,fontsize=10,color='dimgray',fontfamily=sans_name)
                xlims = [0.8*1/Fr*np.pi*2,1/Fr*35]
                ylims = [0,LZ]
                axs[0].set_xlim(xlims)
                axs[0].set_ylim(ylims)
                axs[0].set_aspect(1/2.2*(4.5 * 1/0.25) * LZ/1 * (xlims[1]-xlims[0])/(ylims[1]-ylims[0]))
                axs[0].set_xlabel(r"$\Re\{k_z\}/L^{-1}$",fontsize=fontsize)
                axs[0].set_ylabel("$z/L$",fontsize=fontsize)
                axs[0].set_title("wavenumbers",fontsize=fontsize)

                # Plot fields
                for k,task in enumerate(tasks):

                    cmax = cmaxlist[k]
                    field = file['tasks'][task]
                    c = field[i]
                    if task == "u":
                        c_0, c_1 = sin_cos_decomp(c)
                        labels = [r"$u_{\text{IVP}}$ (sin)",r"$u_{\text{IVP}}$ (cos)"]
                    elif task == "v":
                        c_1, c_0 = sin_cos_decomp(c)
                        labels = [r"$v_{\text{IVP}}$ (cos)",r"$v_{\text{IVP}}$ (sin)"]
                    elif task == "p":
                        c_1, c_0 = sin_cos_decomp(c)
                        labels = [r"$p_{\text{IVP}}$ (cos)",r"$p_{\text{IVP}}$ (sin)"]
                    c_decomp_list = [c_0, c_1]

                    t = field.dims[0]['sim_time'][i]
                    write_num = field.dims[0]['write_number'][i]

                    for l,label in enumerate(labels):
                        # ax_idx = 3*(k+1)+l - 1
                        ax_idx = 3*(l+1)+k - 1

                        ax = axs[ax_idx]
                        ax.imshow(np.real(c_decomp_list[l].T),cmap='RdBu_r',vmin=(-cmax,cmax),extent=[0,Lx,0,LZ],origin="lower",interpolation=interpolation,interpolation_stage=interpolation_stage)
                        ax.axhline(params_IVP['Z0'],linestyle='dotted',color='k',lw=1.5)
                        
                        # Track envelope using WKB group velocity
                        wkb_t_idx = np.argmin(np.abs(t_arr - t))
                        trans_mode_names = [['IGW-0','AW-0'],['IGW-1','SM-AW-1','AW-1']]
                        trans_mode_labels = [['SM-0\n$\\uparrow$\nIGW-0','AW-0\n$\\uparrow$\nSM-0'],['SM-1\n$\\uparrow$\nIGW-1','SM-AW-1\n$\\uparrow$\nSM-1','AW-1\n$\\uparrow$\nSM-AW-1']]
                        branch_idx = l

                        ## Horizontal line on field plots
                        Z_char = Z_char_stack[branch_idx,wkb_t_idx]
                        mode_nm = mode_char_stack[branch_idx,wkb_t_idx]
                        ax.axhline(Z_char,color=color_dict[mode_nm],linestyle=linestyle_dict[mode_nm],linewidth=2)
                        ax.text(0.5,Z_char+LZ/100,mode_nm,fontsize=10,color=color_dict[mode_nm],ha='center',fontfamily=sans_name,fontweight='heavy')

                        ## Points on dispersion relation plot
                        kz_char = kz_char_stack[branch_idx,wkb_t_idx]
                        if linestyle_dict[mode_nm] == '-':
                            facecolor = color_dict[mode_nm]
                        else:
                            facecolor = 'lightgray'
                        axs[0].scatter(1/Fr*kz_char.real,Z_char,color=color_dict[mode_nm],facecolor=facecolor,zorder=10)

                        for trans_idx,t_trans in enumerate(t_transition[branch_idx]):
                            trans_mode_nm = trans_mode_names[branch_idx][trans_idx]
                            if t >= t_trans:
                                ax.axhline(Z_transition[branch_idx][trans_idx],alpha=0.5,color=color_dict[trans_mode_nm],linestyle=linestyle_dict[trans_mode_nm])
                                text_color = color_dict[trans_mode_nm]
                            else:
                                text_color = "#00000000"
                            if k==1:
                                if trans_mode_nm == 'IGW-1':
                                    va = 'top'
                                    xtext = 1.35
                                    jitter = -0.005
                                elif (trans_mode_nm == 'SM-AW-1')|(trans_mode_nm == 'AW-1'):
                                    va = 'bottom'
                                    xtext = 1.35
                                    jitter = 0
                                else:
                                    va = 'center'
                                    xtext = 1.25
                                    jitter = 0
                                ax.text(xtext,jitter+Z_transition[branch_idx][trans_idx],trans_mode_labels[branch_idx][trans_idx],color=text_color,fontsize=8,va=va,ha='center',fontfamily=sans_name,fontweight='heavy')

                        xlims = ax.get_xlim()
                        ylims = ax.get_ylim()
                        ax.fill_between([xlims[0],xlims[1]],[params_IVP['s'],params_IVP['s']], facecolor="none", hatch="//////", edgecolor="k", linewidth=0.5)
                        ax.fill_between([xlims[0],xlims[1]],[LZ-params_IVP['s'],LZ-params_IVP['s']],[LZ,LZ], facecolor="none", hatch="//////", edgecolor="k", linewidth=0.5)
                        ax.set_xticks([0,0.5,1],['0','0.5','1'])
                        ax.set_title(label,fontsize=fontsize)
                        ax.set_xlabel('$x/L$',fontsize=fontsize)
                        ax.set_xticks([0,0.5,1],['0','0.5','1'])
                        ax.set_aspect(4.5 * 1/0.25)

                        if ax_idx in [3,6]:
                            ax.set_yticks([])
                        else:
                            ax.set_ylabel('$z/L$',fontsize=fontsize, labelpad=-5)

                # Legend
                leg_ax_idx = 0
                handles_1 = []
                handles_2 = []
                for nm in ['IGW-0', 'SM-1', 'evan-0', 'SM-AW-1', 'IGW-1', 'AW-1', 'evan-1', 'SM-0', 'AW-0']:
                    if 'evan' in nm:
                        label = nm.replace('evan','evan.')
                    else:
                        label=nm
                    if nm == 'SM-AW-1':
                        linestyle = (3, (2.5, 1.5))
                    else:
                        linestyle = linestyle_dict[nm]
                    handle = axs[leg_ax_idx].axvline(np.nan,color=color_dict[nm],linestyle=linestyle,linewidth=1.5,label=label)
                    if nm in ['IGW-0', 'evan-0', 'IGW-1', 'SM-1', 'SM-AW-1', 'AW-1']:
                        handles_1.append(handle)
                    else:
                        handles_2.append(handle)

                handles_3 = []
                handle = axs[leg_ax_idx].axvline(np.nan,ymin=0.,ymax=1,color='dimgray',linewidth=1,linestyle='dashdot',zorder=0,alpha=0.5,label="pure IGW")
                handles_3.append(handle)
                handle = axs[leg_ax_idx].axhline(np.nan,color='dimgray',linewidth=1,linestyle='--',label="SM cutoff height ($k_y = 0$)")
                handles_3.append(handle)

                # Labels, limits
                legend_1 = axs[leg_ax_idx].legend(handles=handles_1,loc="center",bbox_to_anchor=(1.075, -0.25),ncols=3,framealpha=0,columnspacing=2.,labelspacing=0.75)
                axs[leg_ax_idx].add_artist(legend_1)
                legend_2 = axs[leg_ax_idx].legend(handles=handles_2,handletextpad=1,loc="center",bbox_to_anchor=(3.025, -0.2125),ncols=3,framealpha=0,columnspacing=2.5)
                axs[leg_ax_idx].add_artist(legend_2)
                legend_2 = axs[leg_ax_idx].legend(handles=handles_3,handletextpad=1,loc="center",bbox_to_anchor=(0.05+3.035-0.07, -0.29),ncols=3,framealpha=0,columnspacing=1.5)

                plt.suptitle(f"$t\\omega$ = {t:.2f}",y=0.95,fontsize=fontsize)

                savepath = output_path.joinpath('write_%06i.jpg' %(write_num))

                print(f"Saving image {int(write_num/strd)}/{len(t_arr)}\r", end="")

                plt.savefig(str(savepath), dpi=300, bbox_inches='tight')
                plt.close()