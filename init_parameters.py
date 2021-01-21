from dolfin import *
import sys
sys.path.append('mClasses/'), sys.path.append( 'mClasses/ncg_parallel/')
from DomainCs import *
from ProblemCs import *
from SolverCs import *

# Optimization options for the form compiler
parameters["mesh_partitioner"] = "SCOTCH"  #"ParMETIS" # 
parameters["partitioning_approach"] = "PARTITION"
#
# # parameters["form_compiler"]["precision"] = 1000
parameters["allow_extrapolation"]=True
# Make mesh ghosted for evaluation of DG terms
parameters["ghost_mode"] = "shared_facet"
# parameters["ghost_mode"] = "shared_vertex"
# parameters["ghost_mode"] = 'none'
form_compiler_parameters = {"quadrature_degree": 2}

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = \
"-O3 -ffast-math -march=native"
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
