The WKB wavenumbers and eigenfunctions for both the axisymmetric and non-axisymmetric problems may be computed by running 

    python run_eigen_solves.py

which runs the other scripts in this directory in the correct order.

The WKB solution is constructed and plotted against IVP III ("sim27") in `construct_wkb.ipynb`. For this notebook to run, the simulation snapshots (available on our [Zenodo repository](https://doi.org/10.5281/zenodo.18357092) at [doi:10.5281/zenodo.18357092](https://doi.org/10.5281/zenodo.18357092)) must be located in `IVP/data`. To download the data using the command line, run the following from `IVP/`:<br>

    wget -L -O data.zip "https://zenodo.org/records/18357093/files/data.zip"
    unzip data.zip
    rm data.zip
