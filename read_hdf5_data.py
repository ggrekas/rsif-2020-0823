
from dolfin import*


'''
Extract the reference mesh and the displacement function of the computed 
solution, from the file reference_funs.h5
----------
filename : string, the filename
el_order : integer, the element order of the computed solution
----------
mesh : dolfin.cpp.mesh object,
        the mesh in the reference (undeformed) configuration
u :  dolfin.functions.function object,
        the function containing the computed solution, 
        representing the displacement vector for each point in the 
        reference configuration. 
'''

def get_info(datafile, el_order):

    f = HDF5File(mpi_comm_world(), datafile, 'r')

    mesh = Mesh(mpi_comm_world())

    f.read(mesh, "mesh", False)

    el_order = 1 
    V_l = VectorFunctionSpace(mesh, 'CG', el_order)
    u = Function(V_l)

    f.read(u, 'u')

    return mesh, u



datafile = 'results/vtkFiles/saved_functions/reference_funs.h5'
el_order =1
mesh, u = get_info(datafile, el_order)