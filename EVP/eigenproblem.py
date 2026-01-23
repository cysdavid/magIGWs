import pickle
import json
import numpy as np
from dedalus import public as d3
import pathlib
     
class eigenproblem:
    def __init__(self, Z_value, params_dict, field_dict=None, Nx='auto', build_matrices=False, eta_override=None, extra_terms=None, dz_input=False):
        '''
        Create Dedalus generalized eigenvalue problem
        L q + kz M q = 0

        Parameters
        ----------
        Z_value : float
            Value of Z for which the eigenvalue problem should be built
        params_dict : dictionary
            Dictionary of parameters with keys
            ['om','ky','Gamma','dNdZ','kb','eta','Fr','LZ','Lx']
        field_dict : dictionary, optional
            Dictionary of profiles (1D arrays) in x at Z=Z_value
            with keys ['p','u','v','w','Bx','By'] if eta=0
            and ['p','u','v','w','Bx','By','dBxdz','dBydz'] if eta>0
        Nx : int, optional
            Number of grid points to use in the rescaling of the input data.
            len(field_dict['p'])/Nx must be an integer.
            If 'auto', Nx = len(field_dict['p']).
        build_matrices : bool, optional
            Whether to build L and M matrices
        eta_override: float, optional
            If eta_override != None, will be used as the value of the dimensionless
            magnetic diffusivity: eta = (ohmic diffusivity)/(L**2 * omega).
        dz_input: bool, optional
            Whether to rescale 'dBxdz','dBydz' from field_dict

        '''
        self.Z_value = Z_value
        self.field_dict = field_dict
        self.params_dict = params_dict
        self.Nx = Nx
        self.build_matrices = build_matrices
        self.eta_override = eta_override
        self.extra_terms = extra_terms
        self.dz_input = dz_input

        self.z_diff_flag = False
        self.diff_flag = False

        if self.extra_terms != None:
            if "z_diffusion" in self.extra_terms:
                self.z_diff_flag = True

            if np.sum([("x_diffusion"==term)|("y_diffusion"==term)|("z_diffusion"==term) for term in self.extra_terms]) > 0:
                self.diff_flag = True

        self.extra_term_dict = {
            "dx_ycomp": [
                "",
                "",
                "",
                "- Fr*Gamma*vAx*dx(by) ",
                "",
                "- Fr*Gamma*vAx*dx(v) "
            ],
            "dx_xcomp": [
                "",
                "",
                "",
                "",
                "- Fr*Gamma*vAx*dx(u) ",
                ""
            ],
            "vAx_without_dx": [
                "- Fr*om*Gamma*vAx*kz*bx ",
                "",
                "",
                "+ Fr*Gamma*vAx*dy(bx) ",
                "+ Fr*Gamma*u*dx(vAx) ",
                ""
            ],
            "x_diffusion": [
                "",
                "",
                "",
                "",
                "- eta*dx(dx(bx)) ",
                "- eta*dx(dx(by)) "
            ],
            "y_diffusion": [
                "",
                "",
                "",
                "",
                "- eta*dy(dy(bx)) ",
                "- eta*dy(dy(by)) "
            ],
            "z_diffusion": [
                "",
                "",
                "",
                "",
                "- ((dbxdz*eta*1j*kz)/Fr**2) ",
                "- ((dbydz*eta*1j*kz)/Fr**2) "
            ],
            "x_hyperdiffusion": [
                "",
                "",
                "",
                "",
                "+ hyper_eta*dx(dx(dx(dx(bx)))) ",
                "+ hyper_eta*dx(dx(dx(dx(by)))) "
            ],
            "x_diffusion_indep_eta": [
                "",
                "",
                "",
                "",
                "- eta_x*dx(dx(bx)) ",
                "- eta_x*dx(dx(by)) "
            ]
        }

        self.make_eigenproblem()

    def make_eigenproblem(self):
        Z_level_EVP = self.Z_value
        field_dict = self.field_dict
        params_dict = self.params_dict
        Nx = self.Nx
        build_matrices = self.build_matrices

        # Compute bz
        if self.field_dict != None:
            if ("Bz" in field_dict.keys())&("rho" in field_dict.keys()):
                self.compute_bz_rho = True
            else:
                self.compute_bz_rho = False
        else:
            self.compute_bz_rho = False

        # Rescaling factor
        if self.field_dict != None:
            if Nx == 'auto':
                Nx = len(field_dict['p'])

            if (len(field_dict['p'])/Nx).is_integer():
                rescale = int(len(field_dict['p'])/Nx)
            else:
                raise ValueError(f"x resolution of input data ({len(field_dict['p'])}) is not an integer multiple of the requested x resolution ({Nx}).")

        # Set up Dedalus problem
        dealias = 1 #3/2
        dtype = np.complex128
        self.dealias = dealias
        self.dtype = dtype

        # Physical parameters
        self.params_dict = params_dict
        om,ky,Gamma,dNdZ,kb,eta,Fr,LZ,Lx = [params_dict[key] for key in ['om','ky','Gamma','dNdZ','kb','eta','Fr','LZ','Lx']]
        self.Lx = Lx
        self.Nx = Nx

        if self.eta_override != None:
            eta = self.eta_override
        self.eta = eta

        if self.diff_flag & (self.eta <= 0):
            raise ValueError(f"Diffusive terms requested but eta = {self.eta}. Use `eta_override` kwarg to set eta to positive value.")

        if "hyper_eta" in params_dict.keys():
            hyper_eta = params_dict["hyper_eta"]

        if "eta_x" in params_dict.keys():
            eta_x = params_dict["eta_x"]

        # Create bases and domain
        xcoord = d3.Coordinate('x')
        dist = d3.Distributor(xcoord, dtype=dtype)
        xbasis = d3.ComplexFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)
        x = dist.local_grid(xbasis)
        self.x = x
        
        # Set Z value
        Z = dist.Field()
        Z['g'] = Z_level_EVP

        # Fields
        ## Eigenfunctions
        u = dist.Field(name='u', bases=xbasis)
        v = dist.Field(name='v', bases=xbasis)
        p = dist.Field(name='p', bases=xbasis)
        w = dist.Field(name='w', bases=xbasis)
        bx = dist.Field(name='bx', bases=xbasis)
        by = dist.Field(name='by', bases=xbasis)
        if self.compute_bz_rho:
            bz = dist.Field(name='bz', bases=xbasis)
            rho = dist.Field(name='rho', bases=xbasis)
        dbxdz = dist.Field(name='dbxdz', bases=xbasis)
        dbydz = dist.Field(name='dbydz', bases=xbasis)
        vAx = dist.Field(name='vAx', bases=xbasis)
        vAz = dist.Field(name='vAz', bases=xbasis)

        ## Eigenvalue
        kz = dist.Field()

        # Substitutions
        dx = lambda A: d3.Differentiate(A, xcoord)
        dy = lambda A: 1j*ky*A

        # Background magnetic field
        vAx['g'] = 0.5*np.sin(kb*x)
        vAz['g'] = 0.5*np.cos(kb*x)
        vAx = vAx * 2/np.exp(kb*Z)
        vAz = vAz * 2/np.exp(kb*Z)

        # Buoyancy frequency
        N2 = ((2 + 2*dNdZ*Z)/(2 + dNdZ*LZ))**2

        # Equations:
        eqn0 = "-(kz*om*p) - N2*w"+" = 0"
        eqn1 = "1j*kz*w + dx(u) + dy(v)"+" = 0"
        eqn2 = "-(1j*(om*u + bx*Gamma*kz*vAz)) + dx(p)"+" = 0"
        eqn3 = "-(1j*(om*v + by*Gamma*kz*vAz)) + dy(p)"+" = 0"
        eqn4 = "- bx*1j*om - Gamma*1j*kz*u*vAz"+" = 0"
        eqn5 = "- by*1j*om - Gamma*1j*kz*v*vAz"+" = 0"

        original_equations = [eqn0,eqn1,eqn2,eqn3,eqn4,eqn5]
        all_variables = [u, v, p, w, bx, by, dbxdz, dbydz]
        self.equations,self.variables = self.add_extra_terms(original_equations,all_variables)
        self.variable_names = [var.name for var in self.variables]
        
        # Format equations in latex
        self.format_eqns()

        # Problem
        problem = d3.EVP(self.variables, eigenvalue=kz, namespace=locals())
        for eqn in self.equations:
            problem.add_equation(eqn)

        # Solver
        solver = problem.build_solver(ncc_cutoff=1e-16, entry_cutoff=1e-16)

        # Set fields to IVP data
        if self.field_dict != None:
            u.change_scales(rescale)
            v.change_scales(rescale)
            p.change_scales(rescale)
            w.change_scales(rescale)
            bx.change_scales(rescale)
            by.change_scales(rescale)
            dbxdz.change_scales(rescale)
            dbydz.change_scales(rescale)

            u['g'] = field_dict['u']
            v['g'] = field_dict['v']
            p['g'] = field_dict['p']
            w['g'] = field_dict['w']
            bx['g'] = field_dict['Bx']
            by['g'] = field_dict['By']

            if (self.z_diff_flag)|(self.dz_input):
                if 'dBxdz' in field_dict.keys():
                    dbxdz['g'] = field_dict['dBxdz']
                if 'dBydz' in field_dict.keys():
                    dbydz['g'] = field_dict['dBydz']

            p.change_scales(1)
            u.change_scales(1)
            v.change_scales(1)
            w.change_scales(1)
            bx.change_scales(1)
            by.change_scales(1)
            dbxdz.change_scales(1)
            dbydz.change_scales(1)

            # Get full vector of coefficients (length should be equal to number of eigenvectors)
            if self.z_diff_flag:
                _ = [u['c'],v['c'],p['c'],w['c'],bx['c'],by['c'],dbxdz['c'],dbydz['c']] # Needed to trigger in-place transform
                state_vec_c = solver.subproblems[0].subsystems[0].gather([u, v, p, w, bx, by, dbxdz, dbydz])
            else:
                _ = [u['c'],v['c'],p['c'],w['c'],bx['c'],by['c']] # Needed to trigger in-place transform
                state_vec_c = solver.subproblems[0].subsystems[0].gather([u, v, p, w, bx, by])

            field_rescaled_dict = {}
            field_rescaled_dict['p'] = p['g']
            field_rescaled_dict['u'] = u['g']
            field_rescaled_dict['v'] = v['g']
            field_rescaled_dict['w'] = w['g']
            field_rescaled_dict['Bx'] = bx['g']
            field_rescaled_dict['By'] = by['g']
            field_rescaled_dict['dBxdz'] = dbxdz['g']
            field_rescaled_dict['dBydz'] = dbydz['g']

            self.u = u 
            self.v = v
            self.p = p
            self.w = w
            self.bx = bx
            self.by = by
            self.dbxdz = dbxdz
            self.dbydz = dbydz

            if self.compute_bz_rho:
                bz.change_scales(rescale)
                bz['g'] = field_dict['Bz']
                bz.change_scales(1)
                field_rescaled_dict['Bz'] = bz['g']
                self.bz = bz

                rho.change_scales(rescale)
                rho['g'] = field_dict['rho']
                rho.change_scales(1)
                field_rescaled_dict['rho'] = rho['g']
                self.rho = rho
            
            self.state_vec_c = state_vec_c
            self.field_rescaled_dict = field_rescaled_dict

        if build_matrices:
            solver.build_matrices(solver.subproblems,['M','L'])
            pre_left_pinv = solver.subproblems[0].pre_left_pinv
            pre_right_pinv = solver.subproblems[0].pre_right_pinv
            self.L_min = solver.subproblems[0].L_min
            self.M_min = solver.subproblems[0].M_min
            self.L = pre_left_pinv@self.L_min@pre_right_pinv
            self.M = pre_left_pinv@self.M_min@pre_right_pinv

        self.solver = solver

    def state_vec_coeffs_to_fields(self, state_vec_c, eta='same'):

        if eta == 'same':
            eta = self.params_dict['eta']
        dtype, Lx, dealias = (self.dtype,self.Lx,self.dealias)

        field_dict = state_vec_coeffs_to_fields_gen(state_vec_c, eta, dtype, Lx, dealias)

        return field_dict
    
    def print_eqns(self):
        for eqn in self.equations:
            print(eqn)

    def format_eqns(self):
        latex_replacements = [
            ("dbxdz", r"\partial_z(b_x)"),
            ("dbydz", r"\partial_z(b_y)"),
            ("rho", r"\rho"),
            ("bx", r"b_x"),
            ("by", r"b_y"),
            ("kz",r"k_z"),
            ("om", r"\omega"),
            ("N2", r"N^2"),
            ("Gamma", r"\Gamma"),
            ("vAx", r"v_{A x}"),
            ("vAz", r"v_{A z}"),
            ("dx", r"\partial_x"),
            ("dy", r"i k_y"),
            ("*", r" "),
            ("1j", r"i"),
            ("eta", r"\eta")
        ]

        latex_eqns = []
        for eqn in self.equations:
            new_eqn = eqn
            for src,newstr in latex_replacements:
                new_eqn = new_eqn.replace(src,newstr)
            latex_eqns.append("$"+new_eqn+"$")

        self.latex_eqns = latex_eqns

    def add_extra_terms(self,equations,variables):
        if self.extra_terms != None:
            for term in self.extra_terms:
                for i,eqn in enumerate(equations):
                    equations[i] = eqn.split("=")[0] + self.extra_term_dict[term][i] + "=" + eqn.split("=")[-1]
        
        if self.z_diff_flag:
            equations.append("dbxdz - bx*1j*kz = 0")
            equations.append("dbydz - by*1j*kz = 0")
        else:
            variables = [var for var in variables if ((var.name != 'dbxdz')&(var.name != 'dbydz'))]
        
        return equations,variables

def state_vec_coeffs_to_fields_gen(state_vec_c, eta, dtype, Lx, dealias):

    # Determine number of grid points from length of state_vec_c
    if eta == 0:
        Nx = int(len(state_vec_c)/6)
    else:
        Nx = int(len(state_vec_c)/8)

    # Create bases and domain
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)
    xbasis = d3.ComplexFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)
    x = dist.local_grid(xbasis)

    # Fields
    ## Eigenfunctions
    u = dist.Field(name='u', bases=xbasis)
    v = dist.Field(name='v', bases=xbasis)
    p = dist.Field(name='p', bases=xbasis)
    w = dist.Field(name='w', bases=xbasis)
    bx = dist.Field(name='bx', bases=xbasis)
    by = dist.Field(name='by', bases=xbasis)
    dbxdz = dist.Field(name='dbxdz', bases=xbasis)
    dbydz = dist.Field(name='dbydz', bases=xbasis)

    u['c'] = state_vec_c[:Nx]
    v['c'] = state_vec_c[Nx:2*Nx]
    p['c'] = state_vec_c[2*Nx:3*Nx]
    w['c'] = state_vec_c[3*Nx:4*Nx]
    bx['c'] = state_vec_c[4*Nx:5*Nx]
    by['c'] = state_vec_c[5*Nx:6*Nx]

    field_dict = {}
    field_dict['u'] = u['g']
    field_dict['v'] = v['g']
    field_dict['p'] = p['g']
    field_dict['w'] = w['g']
    field_dict['Bx'] = bx['g']
    field_dict['By'] = by['g']

    if eta != 0:
        dbxdz['c'] = state_vec_c[6*Nx:7*Nx]
        dbydz['c'] = state_vec_c[7*Nx:8*Nx]
        field_dict['dBxdz'] = dbxdz['g']
        field_dict['dBydz'] = dbydz['g']

    return field_dict