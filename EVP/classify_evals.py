"""
Classify eigenvalues using data from eigensolves at two different
horizontal resolutions (but the same number of vertical points)

Usage:
    classify_evals.py <fname_lres> <fname_hres> [--alfvenbdry=<alfbdry>] [--plot=<plot>]

Options:
    --alfvenbdry=<alfbdry>  If 'True', eigenvalues will only be marked as Alfven if kz.real >= 1/Gamma*np.exp(kb*Z) [default: False]
    --plot=<plot> Whether to plot eigenvalues. [default: False]
"""

from docopt import docopt
import matplotlib.pyplot as plt
import h5py
import pickle
import pathlib
import numpy as np
import copy

from dedalus import public as d3

args = docopt(__doc__)
filename_lres = args['<fname_lres>']
filename_hres = args['<fname_hres>']
enforce_alfven_bdry_flag = {'True':True,'False':False}[args['--alfvenbdry']]
plot_flag = {'True':True,'False':False}[args['--plot']]

data_path = pathlib.Path('data').absolute()
file_path_lres = data_path.joinpath(filename_lres)
file_path_hres = data_path.joinpath(filename_hres)

save_dict_lres = pickle.load(open(file_path_lres, 'rb'))
save_dict_hres = pickle.load(open(file_path_hres, 'rb'))
kz_list_lres = np.sqrt(save_dict_lres['kz2'])
kz_list_hres = np.sqrt(save_dict_hres['kz2'])
Z_list = save_dict_lres['Z']

Fr,Nx,Lx,kb,Gamma,LZ,dNdZ,om,ky = [save_dict_lres['params'][key] for key in ["Fr","Nx","Lx","kb","Gamma","LZ","dNdZ","om","ky"]]
if dNdZ==0:
    N = 1

evan_threshold = 6e-3 #1e-1#1e-3 #6e-3 #6e-3#6e-2
alfven_threshold = 1e-8 #5e-13#1e-12#3e-10#1e-8 #3e-10#1e-8

###################################################################################################
# Initial classification of Alfven and evanescent modes
alfven_flag_list = []
evan_flag_list = []
kz_list = []
p_list = []
# u_list = []
v_list = []
for Zind,Z in enumerate(Z_list):
    kz_lres_arr = kz_list_lres[Zind]
    kz_hres_arr = kz_list_hres[Zind]
    alfven_flag_arr = np.ones(len(kz_lres_arr)).astype('bool')
    evan_flag_arr = np.ones(len(kz_lres_arr)).astype('bool')
    p_sublist = []
    # u_sublist = []
    v_sublist = []
    for kzind,kz in enumerate(kz_lres_arr):
        if enforce_alfven_bdry_flag:
            if (np.min(np.abs(kz - kz_hres_arr))/np.abs(kz) >= alfven_threshold) & (kz.real >= 1/Gamma*np.exp(kb*Z)):
                alfven_flag_arr[kzind] = True
            else:
                alfven_flag_arr[kzind] = False
        else:
            if np.min(np.abs(kz - kz_hres_arr))/np.abs(kz) < alfven_threshold:
                alfven_flag_arr[kzind] = False
            else:
                alfven_flag_arr[kzind] = True

        if np.abs(kz.imag/kz.real) < evan_threshold:
            evan_flag_arr[kzind] = False
        else:
            evan_flag_arr[kzind] = True
        
        p_arr = save_dict_lres['p'][Zind][kzind]
        # u_arr = save_dict_lres['u'][Zind][kzind]
        v_arr = save_dict_lres['v'][Zind][kzind]
        p_sublist.append(p_arr)
        # u_sublist.append(u_arr)
        v_sublist.append(v_arr)

    kz_list.append(kz_lres_arr[np.argsort(kz_lres_arr.real)])
    alfven_flag_list.append(alfven_flag_arr[np.argsort(kz_lres_arr.real)])
    evan_flag_list.append(evan_flag_arr[np.argsort(kz_lres_arr.real)])
    p_list.append(np.array(p_sublist)[np.argsort(kz_lres_arr.real)])
    # u_list.append(np.array(u_sublist)[np.argsort(kz_lres_arr.real)])
    v_list.append(np.array(v_sublist)[np.argsort(kz_lres_arr.real)])

# Set up dedalus fields and helper functions for determining parity
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)
xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=1)
x = dist.local_grid(xbasis)
# u = dist.Field(name='u', bases=xbasis)
p = dist.Field(name='u', bases=xbasis)

def get_num_zero_crossings(arr):
    n = (np.diff(np.sign(arr)) != 0).sum() - (arr == 0).sum()
    return n
def get_mode_from_zero_crossings(arr):
    num_zeros = get_num_zero_crossings(arr)
    l = (num_zeros+1)//2
    return l

# Initial classification of parity
parity_list = []
l_list = []

for Zind,Z in enumerate(Z_list):
    print(f"Reading level {Zind+1}/{len(Z_list)}. ",end="\r")
    kz_arr = kz_list[Zind]
    parity_arr = -1*np.ones(len(kz_arr))
    l_arr = -1*np.ones(len(kz_arr))
    for kzind,kz in enumerate(kz_arr):
        if (not alfven_flag_list[Zind][kzind])&(not evan_flag_list[Zind][kzind]):
            
            p_arr = p_list[Zind][kzind]
            p['g'] = (p_arr/np.max(np.abs(p_arr))).real
            peak_ind = np.min(np.argsort(np.abs(p['c']))[-1:])

            l_arr[kzind] = peak_ind//2
            parity_arr[kzind] = peak_ind%2
    
    parity_list.append(parity_arr)
    l_list.append(l_arr)

###################################################################################################
# Correct mode number (l) for non-evanescent modes
for Zind,Z in enumerate(Z_list[:]):
    kz_arr = kz_list[Zind]
    parity_arr = parity_list[Zind]
    l_arr = l_list[Zind]
    alfven_flag_arr = alfven_flag_list[Zind]

    if (Zind>0)&(Zind<len(Z_list)-1):
        for kzind,kz in enumerate(kz_arr[:]):
            p_arr = p_list[Zind][kzind]

            if (not alfven_flag_list[Zind][kzind])&(not evan_flag_list[Zind][kzind]):
                parity = parity_arr[kzind]
                
                parity_arr_below = parity_list[Zind-1]
                mask_below = (parity_arr_below==parity)&(~alfven_flag_list[Zind-1])&(~evan_flag_list[Zind-1])
                masked_kz_arr_below = kz_list[Zind-1][mask_below]

                parity_arr_above = parity_list[Zind+1]
                mask_above = (parity_arr_above==parity)&(~alfven_flag_list[Zind+1])&(~evan_flag_list[Zind+1])
                masked_kz_arr_above = kz_list[Zind+1][mask_above]
                
                # Should be no discontinuities in l
                # Compare to nearest kz at level below first
                if len(masked_kz_arr_below) > 0:
                    dist_to_kz_below = np.min(np.abs(kz - masked_kz_arr_below))
                    kzind_below = np.argwhere(np.abs(kz - kz_list[Zind-1])==dist_to_kz_below)[0,0]
                    l_below = l_list[Zind-1][kzind_below]
                    if (dist_to_kz_below < 3)&(l_arr[kzind] >= l_below)&(l_below>=0):
                    # if (dist_to_kz_below < 3)&(l_below>=0)&(l_arr[kzind] != get_mode_from_zero_crossings(p_arr.real)):
                        l_arr[kzind] = l_below
                    elif (dist_to_kz_below < 3)&(l_below>=0)&(l_arr[kzind] < get_mode_from_zero_crossings(p_arr.real)):
                        l_arr[kzind] = l_below
                    elif len(masked_kz_arr_above) > 0:
                        dist_to_kz_above = np.min(np.abs(kz - masked_kz_arr_above))
                        kzind_above = np.argwhere(np.abs(kz - kz_list[Zind+1])==dist_to_kz_above)[0,0]
                        l_above = l_list[Zind+1][kzind_above]
                        # if (dist_to_kz_above < 3)&(l_arr[kzind] >= l_above)&(l_above>=0):
                        if (dist_to_kz_above < 3)&(l_above>=0)&(l_arr[kzind] != get_mode_from_zero_crossings(p_arr.real)):
                            l_arr[kzind] = l_above

                elif len(masked_kz_arr_above) > 0:
                    dist_to_kz_above = np.min(np.abs(kz - masked_kz_arr_above))
                    kzind_above = np.argwhere(np.abs(kz - kz_list[Zind+1])==dist_to_kz_above)[0,0]
                    l_above = l_list[Zind+1][kzind_above]
                    if (dist_to_kz_above < 3)&(l_arr[kzind] >= l_above)&(l_above>=0):
                    # if (dist_to_kz_above < 3)&(l_above>=0)&(l_arr[kzind] != get_mode_from_zero_crossings(p_arr.real)):
                        l_arr[kzind] = l_above

    l_list[Zind] = l_arr

# Mark stragglers (evanescent modes that are discontinuous) as evanescent
corrected_evan_flag_list = []
for Zind,Z in enumerate(Z_list):
    corrected_evan_flag_arr = np.copy(evan_flag_list[Zind])
    corrected_evan_flag_list.append(corrected_evan_flag_arr)

correct_evan_it = 10
for it in range(correct_evan_it):
    for Zind,Z in enumerate(Z_list[:]):
        kz_arr = kz_list[Zind]
        parity_arr = parity_list[Zind]
        l_arr = l_list[Zind]
        alfven_flag_arr = alfven_flag_list[Zind]

        if (Zind<len(Z_list)-1):
            for kzind,kz in enumerate(kz_arr[:]):
                if (not alfven_flag_list[Zind][kzind])&(not corrected_evan_flag_list[Zind][kzind]):
                    parity = parity_arr[kzind]
                    l = l_arr[kzind]
                    
                    parity_arr_above = parity_list[Zind+1]
                    l_arr_above = l_list[Zind+1]
                    mask_above = (l_arr_above==l)&(parity_arr_above==parity)&(~alfven_flag_list[Zind+1])&(~corrected_evan_flag_list[Zind+1])
                    masked_kz_arr_above = kz_list[Zind+1][mask_above]
                
                    if len(masked_kz_arr_above) == 0:
                        corrected_evan_flag_list[Zind][kzind] = True

###################################################################################################
# Correct parity for evanescent modes
corrected_parity_list = []
corrected_l_list = []
for Zind,Z in enumerate(Z_list):
    print(f"Reading level {Zind+1}/{len(Z_list)}. ",end="\r")
    kz_arr = kz_list[Zind]
    parity_arr = np.copy(parity_list[Zind])
    l_arr = np.copy(l_list[Zind])
    for kzind,kz in enumerate(kz_arr):
        if (not alfven_flag_list[Zind][kzind])&(corrected_evan_flag_list[Zind][kzind]):
            
            p_arr = p_list[Zind][kzind]
            p['g'] = (p_arr/np.max(np.abs(p_arr))).real
            peak_ind = np.min(np.argsort(np.abs(p['c']))[-1:])

            l_arr[kzind] = peak_ind//2
            parity_arr[kzind] = peak_ind%2
    
    corrected_parity_list.append(parity_arr)
    corrected_l_list.append(l_arr)

# Find turning points
turning_pt_list = []
above_turning_pt_list = [np.zeros(len(parity_arr)).astype(bool) for i in range(len(Z_list))]
for Zind,Z in enumerate(Z_list):
    corrected_evan_flag_arr = corrected_evan_flag_list[Zind]
    parity_arr = corrected_parity_list[Zind]
    turning_pt_arr = np.zeros(len(parity_arr)).astype(bool)
    kz_arr = kz_list[Zind]
    corrected_l_arr = corrected_l_list[Zind]
    for kzind,kz in enumerate(kz_arr):
        if (Zind<len(Z_list)-1)&corrected_evan_flag_arr[kzind]:
            parity = parity_arr[kzind]

            parity_arr_above = corrected_parity_list[Zind+1]
            evan_mask_above = (parity_arr_above==parity)&(~alfven_flag_list[Zind+1])&(corrected_evan_flag_list[Zind+1])
            evan_masked_kz_arr_above = kz_list[Zind+1][evan_mask_above]

            # Find turning point. Use l of non-evanescent mode above turning pt to set l of evanescent mode at turning pt
            if len(evan_masked_kz_arr_above) > 0:
                dist_to_evan_kz_above = np.min(np.abs(kz - evan_masked_kz_arr_above))
                if (dist_to_evan_kz_above > 1):
                    turning_pt_arr[kzind] = True

                    # Account for gaps between non-evanescent and evanescent (unresolved eigenvalues)
                    i = 1
                    mask_above = (corrected_parity_list[Zind+i]==parity)&(~alfven_flag_list[Zind+i])&(~corrected_evan_flag_list[Zind+i])
                    masked_kz_arr_above = kz_list[Zind+i][mask_above]

                    while (len(masked_kz_arr_above) == 0)&((Zind+i)<(len(Z_list)-1)):
                        i = i+1
                        mask_above = (corrected_parity_list[Zind+i]==parity)&(~alfven_flag_list[Zind+i])&(~corrected_evan_flag_list[Zind+i])
                        masked_kz_arr_above = kz_list[Zind+i][mask_above]

                    if len(masked_kz_arr_above) > 0:
                        dist_to_kz_above = np.min(np.abs(kz - masked_kz_arr_above))
                        kzind_above = np.argwhere(np.abs(kz - kz_list[Zind+i])==dist_to_kz_above)[0,0]
                        corrected_l_arr[kzind] = corrected_l_list[Zind+i][kzind_above]
                        above_turning_pt_list[Zind+i][kzind_above] = True
                        
    turning_pt_list.append(turning_pt_arr)
    corrected_l_list[Zind] = corrected_l_arr

# Correct l for evanescent modes
for Zind_rev,Z in enumerate(Z_list[::-1]):
    Zind = len(Z_list)-1-Zind_rev
    kz_arr = kz_list[Zind]
    parity_arr = corrected_parity_list[Zind]
    l_arr = corrected_l_list[Zind]
    alfven_flag_arr = alfven_flag_list[Zind]

    if (Zind<len(Z_list)-1):
        for kzind,kz in enumerate(kz_arr[:]):
            p_arr = p_list[Zind][kzind]

            if (not alfven_flag_list[Zind][kzind])&(corrected_evan_flag_list[Zind][kzind]):
                parity = parity_arr[kzind]

                parity_arr_above = corrected_parity_list[Zind+1]
                mask_above = (parity_arr_above==parity)&(~alfven_flag_list[Zind+1])&(corrected_evan_flag_list[Zind+1])
                masked_kz_arr_above = kz_list[Zind+1][mask_above]
                
                # Should be no discontinuities in l
                # Compare to nearest kz at level above
                if len(masked_kz_arr_above) > 0:
                    dist_to_kz_above= np.min(np.abs(kz - masked_kz_arr_above))
                    kzind_above = np.argwhere(np.abs(kz - kz_list[Zind+1])==dist_to_kz_above)[0,0]
                    l_above = corrected_l_list[Zind+1][kzind_above]
                    if (dist_to_kz_above < 3):
                        l_arr[kzind] = l_above
                        
    corrected_l_list[Zind] = l_arr

###################################################################################################
# Categorize waves
wave_type_list = []

for Zind_rev,Z in enumerate(list(reversed(Z_list))):
    Zind = len(Z_list)-1 - Zind_rev
    kz_arr = kz_list[Zind]
    parity_arr = corrected_parity_list[Zind]
    l_arr = corrected_l_list[Zind]
    alfven_flag_arr = alfven_flag_list[Zind]

    wave_type_arr = np.empty(len(kz_arr),dtype='object')
    for kzind,kz in enumerate(kz_arr):
        if alfven_flag_list[Zind][kzind]:
            wave_type_arr[kzind] = 'alfven'
        else:
            parity = parity_arr[kzind]
            l = l_arr[kzind]
            if ~corrected_evan_flag_list[Zind][kzind]:
                mask = (l_arr[:kzind]==l)&(parity_arr[:kzind]==parity)&(~alfven_flag_list[Zind][:kzind])&(~corrected_evan_flag_list[Zind][:kzind])
                
                if np.abs(kz.imag) > evan_threshold/10:
                    if np.imag(kz) < 0:
                        wave_type_arr[kzind] = 'IGW'
                    else:
                        wave_type_arr[kzind] = 'SM'

                elif np.sum(mask) > 0:
                    wave_type_arr[kzind] = 'SM'
                else:
                    wave_type_arr[kzind] = 'IGW'
            else:
                if np.imag(kz) < 0:
                    wave_type_arr[kzind] = 'IGW-evan'
                else:
                    wave_type_arr[kzind] = 'SM-evan'
    wave_type_list.append(wave_type_arr)

wave_type_list = list(reversed(wave_type_list))

# Sort branches into separate arrays
max_l = int(np.max(np.array([np.max(l_arr) for l_arr in corrected_l_list])))
IGW_SM_dict = {f"l={l}":{key:{"Z":[],"kz":[], "p":[]} for key in ["IGW-0","IGW-1","SM-0","SM-1","IGW-evan-0","IGW-evan-1","SM-evan-0","SM-evan-1"]} for l in range(max_l+1)}
IGW_SM_dict["params"] = save_dict_lres['params']

for Zind,Z in enumerate(Z_list):
    kz_arr = kz_list[Zind]
    parity_arr = corrected_parity_list[Zind]
    l_arr = corrected_l_list[Zind]
    wave_type_arr = wave_type_list[Zind]
    p_arr = p_list[Zind]

    for kzind,kz in enumerate(kz_arr):
        if (wave_type_arr[kzind]=='IGW')|(wave_type_arr[kzind]=='SM')|(wave_type_arr[kzind]=='IGW-evan')|(wave_type_arr[kzind]=='SM-evan'):
            l = l_arr[kzind]
            wave_type = wave_type_arr[kzind]
            parity = parity_arr[kzind]
            IGW_SM_dict[f"l={int(l)}"][f"{wave_type}-{int(parity)}"]["Z"].append(Z)
            IGW_SM_dict[f"l={int(l)}"][f"{wave_type}-{int(parity)}"]["kz"].append(kz)
            IGW_SM_dict[f"l={int(l)}"][f"{wave_type}-{int(parity)}"]["p"].append(p_arr[kzind])

# Trim outliers from edges of each branch
for l_key in IGW_SM_dict.keys():
    if l_key != 'params':
        l_dict = IGW_SM_dict[l_key]
        for mode in list(l_dict.keys()):
            mode_dict = l_dict[mode]
            kzarr = np.array(mode_dict["kz"])
            Zarr = np.array(mode_dict["Z"])
            p_mode_list = np.array(mode_dict["p"])

            if len(kzarr) > 2:
                if np.abs(kzarr[0]-kzarr[1]) > 10*np.abs(kzarr[1]-kzarr[2]):
                    # If first kz is very different, get rid of it
                    kzarr = kzarr[1:]
                    Zarr = Zarr[1:]
                    p_mode_list = p_mode_list[1:]
                if np.abs(kzarr[-1]-kzarr[-2]) > 10*np.abs(kzarr[-2]-kzarr[-3]):
                    # If last kz is very different, get rid of it
                    kzarr = kzarr[:-1]
                    Zarr = Zarr[:-1]
                    p_mode_list = p_mode_list[:-1]
                
                IGW_SM_dict[l_key][mode]["kz"] = kzarr
                IGW_SM_dict[l_key][mode]["Z"] = Zarr
                IGW_SM_dict[l_key][mode]["p"] = p_mode_list

# Keep only the keys with eigenvalues/eigenfxns
for l in range(max_l):
    EVP_sub_dict = IGW_SM_dict[f"l={int(l)}"]
    keys = copy.deepcopy(list(EVP_sub_dict.keys()))
    for key in keys:
        if len(EVP_sub_dict[key]['Z']) == 0:
            del EVP_sub_dict[key]

# Save classified eigenvalues and p eigenmodes
def dict2hdf5(filename, dic):
    with h5py.File(filename, 'w') as h5file:
        recursive_dict2hdf5(h5file, '/', dic)


def recursive_dict2hdf5(h5file, path, dic):
    for key, item in dic.items():
        if not isinstance(key, str):
            key = str(key)
        if isinstance(item, (np.ndarray, np.int64, np.float64, int, float, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, list):
            h5file[path + key] = np.array(item)
        elif isinstance(item, dict):
            recursive_dict2hdf5(h5file, path + key + '/',
                                item)
        else:
            raise ValueError('Cannot save %s type' % type(item))

classified_name = filename_lres.split(".")[0]+"_"+filename_hres.split(".")[0]
output_path = data_path.joinpath(classified_name+"_classified.hdf5")
dict2hdf5(output_path,IGW_SM_dict)

###################################################################################################
# Plot
if plot_flag:
    color_dict = {'IGW-1':"#355f8d",'IGW-0':"#36897b",'SM-0':"#aaa100",'SM-1':"#ce7000",
                'IGW-evan-1':"#355f8d",'IGW-evan-0':"#36897b",'SM-evan-0':"#aaa100",'SM-evan-1':"#ce7000"}
    Z_shift=0

    min_l_to_plot = 1
    max_l_to_plot = 1

    fig,axs = plt.subplots(1,2,figsize=(8,4))
    for l in range(min_l_to_plot,max_l_to_plot+1):
        l_dict = IGW_SM_dict[f"l={l}"]
        for mode in list(reversed(list(l_dict.keys()))):
            if 'evan' in mode:
                linestyle='--'
            else:
                linestyle='-'
            mode_dict = l_dict[mode]
            color = color_dict[mode]
            kzarr = np.array(mode_dict["kz"])
            Zarr = np.array(mode_dict["Z"])
            axs[0].plot(kzarr.real,Zarr+Z_shift,linewidth=2,color=color,linestyle=linestyle)
            axs[1].plot(kzarr.imag,Zarr+Z_shift,linewidth=2,color=color,linestyle=linestyle)

        if l <= 3:
            kx = 2*np.pi*l
            kz_pure_IGW = np.sqrt(kx**2+ky**2)*N/om
            axs[0].axvline(kz_pure_IGW,ymin=0.85,ymax=1,color='darkgray',linestyle='--')

    for mode in list(reversed(list(l_dict.keys()))):
        axs[0].plot([],[],color=color_dict[mode],label=mode)

    xlims = axs[0].get_xlim()
    axs[0].plot(1/Gamma*np.exp(kb*np.linspace(0,Z,100)),np.linspace(0,Z,100)+Z_shift,color='gray',linewidth=3,zorder=1)
    axs[0].fill_betweenx(np.linspace(0,Z,100)+Z_shift,1/Gamma*np.exp(kb*np.linspace(0,Z,100)),100*np.ones(100),color='lightgray',zorder=0)

    axs[0].set_xlim(0.8*np.sqrt((2*np.pi)**2+ky**2)*N/om,1.1*1/Gamma*np.exp(kb*LZ))
    axs[0].set_ylim(Z_shift,LZ+Z_shift)
    axs[0].set_xlabel("vertical wavenumber, $\\Re\\{k_z\\}$",fontsize=14)
    axs[0].set_ylabel("Depth, $z/L$",fontsize=14)
    axs[1].set_ylim(Z_shift,LZ+Z_shift)
    axs[1].set_xlabel("vertical wavenumber, $\\Im\\{k_z\\}$",fontsize=14)
    axs[1].set_ylabel("Depth, $z/L$",fontsize=14)
    plt.tight_layout()

    # Save figure
    figure_path = pathlib.Path('figures').absolute()
    if not figure_path.exists():
        figure_path.mkdir()
    classified_fig_path = figure_path.joinpath('classified_evals')
    if not classified_fig_path.exists():
        classified_fig_path.mkdir()

    fig_save_path = classified_fig_path.joinpath(classified_name+".jpg")
    plt.savefig(str(fig_save_path),dpi=300)
    plt.show()