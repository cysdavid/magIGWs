import os

# EIGENVALUES FOR ky = 2 pi (RESISTIVE EQUATIONS)
# Run sparse solve at Z=0 to obtain starting guesses for eigenvalues
target_ids = [2,3,4,5] # target indices correspond to evan.0-IGW0, evan.0-SM0, evan.1-IGW1, evan.1-SM1 branches, respectively
for target_idx in target_ids:
    os.system(f"python sparse_solve_fixed_Z.py --targetindex={target_idx}")

# Run sparse solves, incrementing Z in order to find eigenvalue branches
parities = ['cos','cos','sin','sin'] # restrict by parity
N_modes = [1,1,2,1] # Number of eigenvalues/eigenmodes for sparse solver to return at each Z.
                    # If >1, the eigenpair with matching parity and eigenvalue closest to
                    # that at the previous level in Z is used.
for i,target_idx in enumerate(target_ids):
    target_fname = f"eigfxn_exp_fulldiff_ky2p0pi_Gamma_0p1_eta0_Z0_ind{target_idx}.pkl"
    os.system(f"python sparse_solve_along_branch.py {target_fname} {parities[i]} {N_modes[i]}")

#############################################################################################################################

# EIGENVALUES FOR ky = 0 (IDEAL EQUATIONS)
# Run dense solves at 2 horizontal resolutions
Nx_list = [96, 128]
NZ = 512
for Nx in Nx_list:
    os.system(f"python dense_solve_ideal.py {Nx} {NZ}")

# Classify resolved eigenvalues
fnames = [f"evp_LZ=2p500e-01_ky=0p00e+00pi_Gamma1p00e-01_eta0p0e+00_{Nx}_{NZ}.pickle" for Nx in Nx_list]
os.system(f"python classify_evals.py {fnames[0]} {fnames[1]} --plot=False")