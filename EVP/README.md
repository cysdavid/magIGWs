The WKB wavenumbers and eigenfunctions for both the axisymmetric and non-axisymmetric problems may be computed by running 

    python run_eigen_solves.py

which runs the other scripts in this directory in the correct order.

The WKB solution is constructed and plotted against IVP III ("sim27") in `construct_wkb.ipynb`. For this notebook to run, the simulation snapshots (available on the Zenodo repository) must be located in `EVP/data`.