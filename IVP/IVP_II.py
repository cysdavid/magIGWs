"""
Simulate linear magneto-Boussinesq equations

Usage:
    waves_IVP.py [--restart_idx=<ridx>]

Options:
    --restart_idx=<ridx> If not 'None', simulation will pick up from the specified index of the last set file [default: None]
"""

from docopt import docopt
import numpy as np
from mpi4py import MPI
import time
import json

from dedalus import public as d3

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.parallel import Sync
from dedalus.tools.general import natural_sort
import pathlib
import glob
import h5py

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

args = docopt(__doc__)
restart_idx = args['--restart_idx']
if restart_idx == None:
    restart = False
else:
    restart = True
    restart_idx = int(restart_idx)

# Simulation name
sim_name = 'sim32'

# Numerical Parameters
Nx, NZ = (4096,3072)
dealias = 3/2
dtype = np.complex128
timestep = 1./3*0.025 * 1/2 * 4 * 2/3 * 1/2 * 1/2
timestepper = d3.RK222

# Physical parameters
Lx, LZ = (1., 1./4)
kw = 2*np.pi
ky = 2*np.pi
eta = 1e-9 # = (1/Lu) * Fr**2
eta_x = 1e-9
Fr = 0.025
Gamma = 0.1
dNdZ = 0
kb = 2*np.pi
Z0 = LZ - 1/40
deltaZ = 0.00416667
T = 1
s = 0.0125
sB = 0.00416667
deltaZB = 0.00104167
hyperdiffusion = False

# Cadences and stop time
output_cadence = 10
stop_iteration = 60000*4 *3/2 * 2 #np.inf # 100
stop_sim_time = 1000 #np.inf # 50000*0.025
snapshot_dt = 2*np.pi/8 #40*0.025

# Create bases and domain
coords = d3.CartesianCoordinates('x', 'Z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
Zbasis = d3.ComplexFourier(coords['Z'], size=NZ, bounds=(0, LZ), dealias=dealias)
x, Z = dist.local_grids(xbasis, Zbasis)

# Fields
u = dist.Field(name='u', bases=(xbasis,Zbasis))
v = dist.Field(name='v', bases=(xbasis,Zbasis))
w = dist.Field(name='w', bases=(xbasis,Zbasis))
p = dist.Field(name='p', bases=(xbasis,Zbasis))
rho = dist.Field(name='rho', bases=(xbasis,Zbasis))
Bx = dist.Field(name='Bx', bases=(xbasis,Zbasis))
By = dist.Field(name='By', bases=(xbasis,Zbasis))
Bz = dist.Field(name='Bz', bases=(xbasis,Zbasis))
tau_p = dist.Field()
t = dist.Field()

# Buoyancy frequency
N2 = dist.Field(name='N2', bases=Zbasis)
N2['g'] = ((2 + 2*dNdZ*Z)/(2 + dNdZ*LZ))**2

# Background magnetic field
vAx = dist.Field(name='vAx', bases=(xbasis,Zbasis))
vAz = dist.Field(name='vAz', bases=(xbasis,Zbasis))
vAx['g'] = -0.5*(np.sin(kb*x)*(np.tanh((sB - Z)/deltaZB) + np.tanh((-LZ + sB + Z)/deltaZB)))/np.exp(kb*Z)
vAz['g'] = -0.5*(np.cos(kb*x)*(np.tanh((sB - Z)/deltaZB) + np.tanh((-LZ + sB + Z)/deltaZB)))/np.exp(kb*Z)
dx_vAx = d3.Differentiate(vAx, coords['x'])
dx_vAx = d3.Grid(dx_vAx)
dZ_vAx = d3.Differentiate(vAx, coords['Z'])
dZ_vAx = d3.Grid(dZ_vAx)
dx_vAz = d3.Differentiate(vAz, coords['x'])
dx_vAz = d3.Grid(dx_vAz)
dZ_vAz = d3.Differentiate(vAz, coords['Z'])
dZ_vAz = d3.Grid(dZ_vAz)
vAx = d3.Grid(vAx)
vAz = d3.Grid(vAz)

# Damping 
D = dist.Field(bases=Zbasis)
D['g'] = (2 + np.tanh((s - Z)/deltaZ) + np.tanh((-LZ + s + Z)/deltaZ))/(2.*T)
D = d3.Grid(D)

# Forcing
F = dist.Field(bases=Zbasis)
F['g'] = np.exp(-(Z - Z0)**2/deltaZ**2)
F = d3.Grid(F)

# Substitutions
eikx =dist.Field(bases=xbasis)
eikx['g'] = np.exp(1j*kw*x)
eikx = d3.Grid(eikx)
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: 1j*ky*A
dZ = lambda A: d3.Differentiate(A, coords['Z'])
avg = lambda A: d3.Average(A, ('x', 'Z'))

# Problem
problem = d3.IVP([u, v, w, p, rho, Bx, By, Bz, tau_p], time=t, namespace=locals())
problem.add_equation("dx(u) + dy(v) + Fr*dZ(w) + tau_p = 0")
problem.add_equation("dt(u) + dx(p) = -(D*u) + Fr*Gamma*(vAz*(-(Fr*dx(Bz)) + dZ(Bx)) + Bz*Fr*(-dx_vAz + dZ_vAx))")
problem.add_equation("dt(v) + dy(p) = -(D*v) + Fr*Gamma*(vAx*(dx(By) - dy(Bx)) + vAz*(-(Fr*dy(Bz)) + dZ(By)))")
problem.add_equation("dt(w) + dZ(p)/Fr + rho/Fr**2 = -(D*w) + Gamma*vAx*(Fr*dx(Bz) - dZ(Bx)) + Bx*Gamma*(dx_vAz - dZ_vAx)")
problem.add_equation("dt(rho) - w*N2 = -(D*rho) + F*eikx*np.exp(-1j*t)")
problem.add_equation("dt(Bx) - eta_x*dx(dx(Bx)) - eta*(dy(dy(Bx)) + dZ(dZ(Bx))) = -(Bx*D) + Fr*Gamma*(vAx*dx(u) - u*dx_vAx + vAz*dZ(u) - Fr*w*dZ_vAx)")
problem.add_equation("dt(By) - eta_x*dx(dx(By)) - eta*(dy(dy(By)) + dZ(dZ(By))) = -(By*D) + Fr*Gamma*(vAx*dx(v) + vAz*dZ(v))")
problem.add_equation("dt(Bz) - eta_x*dx(dx(Bz)) - eta*(dy(dy(Bz)) + dZ(dZ(Bz))) = -(Bz*D) - Gamma*u*dx_vAz + Fr*Gamma*(vAx*dx(w) - w*dZ_vAz + vAz*dZ(w))")
# problem.add_equation("dx(Bx) + dy(By) + Fr*dZ(Bz) = 0")
problem.add_equation("integ(p) = 0")

# Build solver
solver = problem.build_solver(timestepper)
logger.info('Solver built')

# Integration parameters
solver.stop_sim_time = stop_sim_time
solver.stop_iteration = stop_iteration

# Create data and params directories if needed
data_path = pathlib.Path('data').absolute()
params_path = pathlib.Path('params').absolute()
with Sync() as sync:
    if sync.comm.rank == 0:
        if not data_path.exists():
            data_path.mkdir()
        if not params_path.exists():
            params_path.mkdir()
save_data_path = data_path.joinpath(sim_name)
save_params_path = params_path.joinpath(sim_name+".json")

# Save parameters
if rank == 0:
    params_dict = {'Nx':Nx,'NZ':NZ,'timestep':timestep,
            'Lx':Lx,'LZ':LZ,'ky':ky,'Fr':Fr,'Gamma':Gamma,'dNdZ':dNdZ, 'eta':eta, 'eta_x':eta_x,
                'kb':kb,'Z0':Z0,'deltaZ':deltaZ,'T':T,'s':s,'sB':sB,'deltaZB':deltaZB,'hyperdiffusion':hyperdiffusion
                }
    params_json = json.dumps(params_dict, indent = 4)
    with open(save_params_path, "w") as outfile: 
        outfile.write(params_json)

# Restart
if restart:
    files = natural_sort(glob.glob(str(save_data_path.joinpath("*.h5"))))
    filename = files[-1]
    with h5py.File(filename, mode='r') as file:
        # If hdf5 file is corrupted, 0th dimension may be larger than number of iterations by 1
        if restart_idx < 0:
            restart_idx = file['scales']['iteration'].shape[0] + restart_idx
        # Load solver attributes
        write = file['scales']['write_number'][restart_idx]
        dt = file['scales']['timestep'][restart_idx]
        solver.iteration = solver.initial_iteration = file['scales']['iteration'][restart_idx]
        solver.sim_time = solver.initial_sim_time = file['scales']['sim_time'][restart_idx]
        # Log restart info
        logger.info("Loading iteration: {}".format(solver.iteration))
        logger.info("Loading write: {}".format(write))
        logger.info("Loading sim time: {}".format(solver.sim_time))
        logger.info("Loading timestep: {}".format(dt))
        # Load fields
        for field in solver.state:
            if field.name in list(file['tasks']):
                field.load_from_hdf5(file, restart_idx)
    file_handler_mode = 'append'
else:
    file_handler_mode = 'overwrite'

# Analysis
snapshots = solver.evaluator.add_file_handler(str(save_data_path), sim_dt=snapshot_dt, max_writes=1000, mode=file_handler_mode)
# snapshots.add_tasks(solver.state)
snapshots.add_task(u)
snapshots.add_task(v)
snapshots.add_task(w)
snapshots.add_task(p)
snapshots.add_task(rho)
snapshots.add_task(Bx)
snapshots.add_task(By)
snapshots.add_task(Bz)
snapshots.add_task(N2)
snapshots.add_task(vAx)
snapshots.add_task(vAz)
snapshots.add_task(dx(Bx) + dy(By) + Fr*dZ(Bz),name="divB")
snapshots.add_task(avg((np.abs(u)**2 + np.abs(v)**2 + Fr**2*np.abs(w)**2)/4), name="KE")
snapshots.add_task(avg((np.abs(Bx)**2 + np.abs(By)**2 + Fr**2*np.abs(Bz)**2)/4), name="ME")
snapshots.add_task(avg((np.abs(u)**2 + np.abs(v)**2 + Fr**2*np.abs(w)**2)/4 + np.abs(rho)**2/(4*N2) + (np.abs(Bx)**2 + np.abs(By)**2 + Fr**2*np.abs(Bz)**2)/4), name="E")

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=output_cadence)
flow.add_property(np.sqrt(u**2 + v**2 + Fr**2*w**2), name='u')
flow.add_property(np.sqrt(Bx**2 + By**2 + Fr**2*Bz**2), name='B')
flow.add_property(avg((np.abs(u)**2 + np.abs(v)**2 + Fr**2*np.abs(w)**2)/4), name='KE')

# Main loop
try:
    logger.info('Starting loop')
    while solver.proceed:
        solver.step(timestep)
        # hyperdiffusion
        if hyperdiffusion == True:
            for field in solver.state:
                field['c'][Nx//4:-Nx//4,:] = 0.
                field['c'][:,NZ//4:-NZ//4] = 0.
        if (solver.iteration-1) % output_cadence == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, timestep))
            logger.info('Max u = {}, Max B = {}, KE = {}'.format(flow.max('u'), flow.max('B'), flow.max('KE')))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
