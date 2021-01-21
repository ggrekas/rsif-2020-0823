__author__ = 'Georgios Grekas (grekas.g@gmail.com)'
results_path = 'results/'

from init_parameters import *

el_order = 2 # the elements order 
# --------- Create domain and mesh START ---------------------
x0, y0, x1, y1 = -12, -10, 12, 10 # rectangle corners
c1_x, c1_y, i_rho1 = 4, 0., 2.    # center and radius of a cell
c2_x, c2_y, i_rho2 = -4, 0., 2.   # center and radius of the second cell

u0 = -1.0 # uniform radial displacement, - gives contraction, + gives expansion 
###u0 can also be a vector, i.e. [a, b] for some a and b,  or an expression.
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = x0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = x1)#
# Define Dirichlet boundary (y = 0 or y = 1)
l = Expression(("0.0",
                "0.0"), degree=2)
r = Expression(("0.0",
                "0.0"), degree=2)

domain =RectangularDomain(x0, y0, x1, y1, {l:left, r:right}) # the domain of ECM, it is a rectangular, 
## where the top and the bottom boundaries are free of forces and the left, right boundaries are fixed.
# or it can be Circular
#domain =CircularDomain(rho, c_x, c_y) # here the outer boundary is free

domain.remove_subdomain( CircularDomain(i_rho1, c1_x, c1_y, u0) ) # from the previous defined domain remove a cirlce 
#                                                                   of radius i_rho1 with center (c1_x, c1_y),
#                                                                   u0 is the boundary condition on the cell
domain.remove_subdomain( CircularDomain(i_rho2, c2_x, c2_y, u0) )

resolution = 120 # determine the mesh resolution, higher resolution gives finer mesh
domain.create_mesh(resolution) # create a mesh over the domain for the given resolution
# --------- Create domain and mesh  END ---------------------


# -----------------Define problem type START --------------------------------
# ---------- i.e. mEnergyMinimization or mNonlinearVariational --------------
problem = mEnergyMinimizationProblem('1.0/96.0 * (5*Ic**3 - 9*Ic**2 - \
                                    12*Ic*J**2 + 12*J**2 + 8) + exp( 80*(0.22 - J) )',
                                     domain, el_order=el_order)
# problem.set_k(1, cell_model ='linear_spring') # accounts for cell response  (you can exclude this line)
# problem.set_k(1, cell_model ='PNIPAAm_particles') # accounts for cell response  (you can exclude this line)
problem.add_higher_gradients(epsilon =1e-1/i_rho1)
# -----------------Define problem type END --------------------------------




# ---------------- Define solver type--------------------------------------

# choose if the centers are free to move or not. Cells can move only if one has called the method problem.set_k
solver_t = 'fixed_centers'#'free_centers' # free centers work only if problem_set_k has previously called. 


solver = m_ncgSolver(problem, res_path = results_path, solver=solver_t) # let save_run =False, 

solver.initialization('Polyconvex') # initialize the displacement vector using a polyconvex functional
u = solver.solve()
# ---------------- solving Ends -----------------------------------------




solver.plot_results()
try:
	solver.save_all_function_info()
except:
	print('hdf5 is not supported')


