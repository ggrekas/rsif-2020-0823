from ProblemCs import *
import warnings
from export_run_info import *
from problemInfoCs import *
import numpy as np
import os
from export_run_info import *
# from ncg_parallel.ncg_PETS import minimize_cg as fmin_cg
import sys
sys.path.insert(0, 'ncg_parallel/')

from ncg_PETS import minimize_cg as fmin_cg
import time

#from mpi4py import MPI as MPI4PY


#  TODO: (crusial) exterior boundary conditions in Newton's method haven't been checked
#  TODO: 1) check Newton's initialization for manual and automated solution
#  TODO: 2) step1 and stepk solution differ in variationalProblem and Energy minimization
#  TODO: 3) in initialization: if problem nonlinear variational then init with
#  TODO:      non linear variational, eitherwise with energy minimization.
#  TODO: 4) F11 and F22 between cells: a more robust implementation
#  TODO: 5) in ncg solver f and grad_f, examine line np.array(X), because X must me np.array
#  TODO:    maybe causes code delay.
#  TODO: 6) Solve with fixed and free circles centers should convert to a global solve.
#  TODO:    Movement should be enabled in Problem and not is Solver.

class mSolver:
    def __init__(self, problem, res_path = 'results/', save_run=True):
        self._u = problem.get_u()
        self._du = problem.get_du
        self._problem = problem
        self._bcs = problem.get_bc()

        self._res_path= res_path
        self._Pi = problem.Pi
        self._niterInfo = None
        self._has_initialized= False
        self.init_name = None
        self._comm = mpi_comm_world()
        self._mpiRank = MPI.rank(self._comm)

        self._detF, self._F22, self._dens = None, None, None
        self._F11 = None
        self._norm_u, self._norm_u_def, self._meshy, self._V_norm_u = None, None, None, None
        self._dens_undef, self._V_dens = None, None
        self._eig1, self._eig2, self._p_eval1 = None, None, None

        self._exp_info = '0'

        self._currentInfo = runInfo(self._problem.domain, self._problem, self, res_path,
                                    self._problem.domain.mesh_resolution(),
                                    melements='CG' + str(self._problem.el_order() ))
        self._save_run = save_run
        self._start_time = time.time()


        self._energy_i = None

        return

    def get_name(self):
        return self._solver.__name__

    def solve(self):
        if self._save_run and self._exp_info == '0':
            self._exp_info = str(self._currentInfo.get_previous_run_num() + 1)



        u =self._solver()
        total_time = time.time() - self._start_time

        print('time to solve problem = %d seconds'% (total_time) )
        self._currentInfo.set_execution_time(total_time)

        return u

    # def solve(self):
    #     m = 'Method solve is undefined for class %s'
    #     raise NotImplementedError(m % self.__class__.__name__)

    def add_disturbances(self):
        print('--------------- adding disturbances start --------------')
        u_array = self._u.vector().get_local()
        N = u_array.shape[0]
        u_array = u_array + 0.02*np.random.rand(N)

        self._u.vector().set_local(u_array)

        print('--------------- adding disturbances end --------------')

        return


    def initialization(self, psi_name):
        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)
        if mpiRank == 0:
            print('--------------- initialization start --------------')

        domain, V = self._problem.domain, self._problem.V
        # problem = mNonlinearVariationalProblem(V, psi_name, domain)
        problem = mEnergyMinimizationProblem(psi_name, domain, self._problem.el_order())
        solver = mNewtonSolver(problem)
        u_init = solver.automated_solution()

        # self._u.vector()[:] = u_init.vector()
        self._u.assign(u_init)
        self._has_initialized = True

        if mpiRank == 0:
            print ('--------------- initialization end -------------------')

        return

    def initialization_ncg(self, psi_name):
        print('--------------- initialization start --------------')

        domain, V = self._problem.domain, self._problem.V
        # problem = mNonlinearVariationalProblem(V, psi_name, domain)
        problem = mEnergyMinimizationProblem(V, psi_name, domain, self._problem.get_W, 
        el_order=self._problem.el_order() )
        solver_t = 'fixed_centers'#'free_centers' #
        solver = m_ncgSolver(problem, res_path = '', solver=solver_t,
                     save_run=False)
        u_init = solver.solve()

        # self._u.vector()[:] = u_init.vector()
        self._u.assign(u_init)
        self._has_initialized = True

        print ('--------------- initialization end -------------------')

        return

    def initialization_epsilon(self, psi_name, epsilon):
        print('--------------- initialization start --------------')

        domain, V = self._problem.domain, self._problem.V
        # problem = mNonlinearVariationalProblem(V, psi_name, domain)
        problem = mEnergyMinimizationProblem(V, psi_name, domain, self._problem.get_W)
        problem.add_higher_gradients(epsilon, alpha=8)
        solver = mNewtonSolver(problem)
        u_init = solver.automated_solution()

        # self._u.vector()[:] = u_init.vector()
        self._u.assign(u_init)
        self._has_initialized = True

        print ('--------------- initialization end -------------------')

        return


    def init_from_bc(self):
        print('--------------- initialization start --------------')

        for bc in self._bcs:
            bc.apply(self._u.vector())
        # self._has_initialized = True

        print ('--------------- initialization end -------------------')

        return self._u

    def init_from_function(self, u_init):
        print('--------------- initialization start --------------')

        V = self._problem.getFunctionSpace()

        f_p = project(u_init, V)

        self._u.assign(f_p)
        # for bc in self._bcs:
        #     bc.apply(self._u.vector())
        # self._u.assign(u_init)
        self._has_initialized = True

        print ('--------------- initialization end -------------------')

        return self._u

    def init_from_function2(self, l1, l2):
        print ('--------------- initialization start --------------')
        V =self._problem.getFunctionSpace()
        class myf(Expression):
            def eval(self,value, x):
                # value[0] = (l1 - 1) * x[0] + 0.001 * np.sin(np.pi * x[0])
                # value[1] = (l2 - 1) * x[1] +  0.001*np.sin(np.pi * x[1] )
                # value[0] = (l1 - 1) * x[0] + 0.00 *x[0]*(1-x[0])
                # # value[1] = (l2 - 1) * x[1] + 0.1 *x[1]*(1-x[1])
                # if x[1] > 0.99:
                #     value[1] = (l2 - 1)
                # else:
                #     value[1] = (l2 - 1)*x[1]**0.7
                # k11, k12, k21, k22 = 2, 2, 2, 2
                # value[0] = (l1 - 1) * x[0] + 0.0 * np.sin(
                #     k11 * np.pi * x[0]) * np.sin(k12 * np.pi * x[1])
                # value[1] = (l2 - 1) * x[1] + 0.01 * np.sin(
                #     k11 * np.pi * x[0]) * np.sin(k22 * np.pi * x[1])
                a, k22 =l2,  0.1
                k21 = (l2 - (1.0 - a) * k22) / a
                # k21 = 0.9
                if x[1] < a:
                    value[1] = a * (1 - k21) - (1 - k21) * x[1]
                else:
                    value[1] = a * (1 - k22) - (1 - k22) * x[1]\
                                + 0.0 * np.sin(
                         10 * np.pi * x[1]) * x[0] * (1 - x[0])
                # k11, k12, k21, k22 = 2, 2, 2, 2
                value[0] = (l1 - 1) * x[0]

            def value_shape(self):
                return (2,)

        if dolfin_version() >= '2016.2.0':
            f_p = project(myf(degree=2), V)
        else:
            f_p = project(myf(), V)

        for bc in self._bcs: # only for boundary conditions
            bc.apply(self._u.vector() )

        self._u.assign(f_p)
        self._has_initialized = True
        self._l1_init, self._l2_init = l1, l2
        self.init_name = 'fun2'
        print('--------------- initialization end -------------------')

        return

    def init_from_function3(self, l1, l2):
        print('--------------- initialization start --------------')
        lM, lm = 0.8, 0.01
        V =self._problem.getFunctionSpace()
        class myf(Expression):
            def eval(self,value, x):
                a = (l2-lm)/(lM-lm)
                if x[1] < a:
                    value[1] = (x[1]-a)*lM + a - x[1]
                else:
                    value[1] = (x[1]-a)*lm + a - x[1]

                value[0] = (l1 - 1) * x[0]

            def value_shape(self):
                return (2,)

        if dolfin_version() >= '2016.2.0':
            f_p = project(myf(degree=self._problem.el_order()), V)
        else:
            f_p = project(myf(), V)

        for bc in self._bcs: # only for boundary conditions
            bc.apply(self._u.vector() )

        self._u.assign(f_p)
        self._has_initialized = True
        self._l1_init, self._l2_init = l1, l2
        self.init_name = 'fun3'
        print('--------------- initialization end -------------------')

        return

    def init_from_function2_new(self, l1, l2, lM, lm):
        print('--------------- initialization start --------------')
        V =self._problem.getFunctionSpace()
        class myf(Expression):
            def eval(self,value, x):
                a = (l2-lm)/(lM-lm)
                if x[1] < a:
                    value[1] = (x[1]-a)*lM + a - x[1]
                else:
                    value[1] = (x[1]-a)*lm + a - x[1]

                value[0] = (l1 - 1) * x[0]

            def value_shape(self):
                return (2,)

        if dolfin_version() >= '2016.2.0':
            f_p = project(myf(degree=self._problem.el_order()), V)
        else:
            f_p = project(myf(), V)

        for bc in self._bcs: # only for boundary conditions
            bc.apply(self._u.vector() )

        self._u.assign(f_p)
        self._has_initialized = True
        self._l1_init, self._l2_init = l1, l2
        self.init_name = 'fun2_new'
        print('--------------- initialization end -------------------')

        return

    def init_from_function_u_init(self, l1, l2,u_init):
        print('--------------- initialization start --------------')
        V =self._problem.getFunctionSpace()

        f_p = project(u_init, V)


        self._u.assign(f_p)
        self._has_initialized = True
        self._l1_init, self._l2_init = l1, l2
        self.init_name = 'fun3'
        print('--------------- initialization end -------------------')

        return


    def get_init_l1_l2(self):
        try:
            return self._l1_init, self._l2_init
        except:
            return None, None


    def init_fromHDF5File(self, exp_num, folder_path='/home/ggrekas/Documents/mClasses/results/vtkFiles/'):
        file_path =folder_path + str(exp_num) + '/saved_functions/'

        f = HDF5File(mpi_comm_world(), file_path + 'reference_funs.h5', 'r')
        mesh_i = Mesh(mpi_comm_world())
        # mesh = self._problem.domain.get_mesh()
        f.read(mesh_i, "mesh", False)
        V_i = VectorFunctionSpace(mesh_i, 'Lagrange', 2)
        u_i = Function(V_i)
        f.read(u_i, 'u')
        # plot(u_i, mode='displacement', interactive=True)

        V = self._problem.getV()
        u_init = project(u_i, V)
        self._u.assign(u_init)

        file_u =File('u.pvd')
        file_u << self._u
        print('init_from HDF5File end.....')
        # plot(self._u, interactive=True, mode='displacement')
        return


    @property
    def has_initialized(self):
        return self._has_initialized


    def set_omega(self, omega):
        m = 'Warning, set_omega is valid only in Newton\'s solver, not in %s'
        print(m % (self.__class__.__name__) )

        return

    def get_total_energy_val(self):
        if self._niterInfo == None:
            return -1
        return self._niterInfo.get_energy_val()

    def get_min(w):
        return w.vector().get_local().min()

    def get_max(w):
        return w.vector().get_local().max()

    
    def compute_funcs(self, exp_info=None):
        if exp_info == None:
            exp_info = self._exp_info
        F, J, V0 = self._problem.getF(), self._problem.getJ(), self._problem.getV0()
        V, el_order = self._problem.getFunctionSpace(), self._problem.el_order()
        detF, F22, F11 = calculate_detF_F22_F11(F, J, V0)
        mesh = self._problem.domain.get_mesh()

        self._dens, self._dens_undef, self._V_dens = calculate_dens(self._u, J, el_order, V0, mesh , V)
        self._norm_u, self._norm_u_def, self._meshy, self._V_norm_u = calculate_u_norms(self._u, J,
                           el_order, V0, mesh, V)

        # self._calculate_function(F22, np.array([0,0]))
        self._detF, self._F22 = detF, F22
        self._F11= F11
        
        return

    def plot_results(self, exp_info=None, final_save = True):
        if self._save_run  and final_save:
            self._currentInfo.export_run()
        if exp_info == None:
            exp_info = ''
       
        self.compute_funcs(exp_info)
        self._problem.compute_eval1()
        self._save_functions(exp_info)

        return


    def _calculate_function(self, F, x):
        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)

        vec = PETScVector()
        vec.init(self._comm, (mpiRank, mpiRank + 1))

        vec.set_local(np.array([F(x)]) )
        try:
            val = F(x)
            print ('mpiRank=', mpiRank, ', F22(0,0) = ', val)
        except:
            val = 0

        return

    def _make_functions_plot(self, exp_info, window_width, window_height):
        plotsPath = self._plotsPath

        det_plot = plot(self._detF, title='detF for experiment ' + exp_info,
                        window_width=window_width, window_height=window_height)
        name = plotsPath + exp_info + '_detF'
        det_plot.elevate(50);  # det_plot.update(detF);
        det_plot.write_png(name)

        F22_plot = plot(self._F22, title='F22 for experiment ' + exp_info,
                        window_width=window_width, window_height=window_height)
        name = plotsPath + exp_info + '_F22'
        F22_plot.elevate(50);  # F22_plot.update(F22);
        F22_plot.write_png(name)


        dens_plot = plot(self._dens, title='density on deformed configuration' + exp_info,
                        window_width=window_width, window_height=window_height)
        name = self._plotsPath + exp_info + '_rho'
        dens_plot.elevate(50);
        dens_plot.write_png(name)

        u_plot = plot(self._u, mode='displacement', window_width=window_width,
                      window_height=window_height, title='displacement vector')
        name = plotsPath + exp_info + '_u'
        u_plot.write_png(name)

        return


    def _save_functions(self, exp_info):
        if exp_info =='':
            results_path = self._res_path +"vtkFiles/" 
        else:
            results_path = self._res_path +"vtkFiles/"   + exp_info +"/"
        
        file_u = File(results_path + '/u.pvd')
        file_J = File(results_path + '/detF.pvd')
        file_dens = File(results_path + '/dens.pvd')

        self._detF.rename('detF', 'detF'); file_J << self._detF
        self._dens.rename('dens', 'dens'); file_dens << self._dens
        self._u.rename('u', 'u'); file_u << self._u
         
        return

    def project_to_DG_space(self, fun_v, el_order):

        # fix function space
        V = FunctionSpace(self._problem.domain.get_mesh(), 'DG', el_order)
        p_fun = project(fun_v, V)

        return p_fun

    def project_to_TensorDG_space(self, fun_v, el_order):

        # fix function space
        V = TensorFunctionSpace(self._problem.domain.get_mesh(), 'DG', el_order)
        p_fun = project(fun_v, V)

        return p_fun


    def save_all_function_info(self, exp_info=None):
        if exp_info == None:
            exp_info = ''
        # exp_info = self._exp_info
        mesh = self._problem.domain.get_mesh()
        results_path = self._res_path +"vtkFiles/" + exp_info + "/saved_functions/"
        
        Hdf_r = HDF5File(mesh.mpi_comm(), results_path + "reference_funs.h5", "w")
        Hdf_r.write(mesh, "mesh")
        Hdf_r.write(self._u, "u")
        # Hdf_r.write(self._detF, "detF")
        # Hdf_r.write(self._eig1, "eig1")
        # Hdf_r.write(self._eig2, "eig2")
        # Hdf_r.write(self._p_eval1, "p_eval")
 
        Hdf_r.close()

        # Hdf_d = HDF5File(self._meshy.mpi_comm_world(), results_path +  "deformed_funs.h5", "w")
        # Hdf_d.write(self._meshy, "meshy")
        # Hdf_d.write(self._norm_u_def, "norm_u_def")
        # Hdf_d.write(self._dens, "dens")

        # Hdf_d.close()
        try:

            if self._mpiRank == 0:
                np.save(results_path +'energy_i', self._energy_i[0: np.int(self._niter/100)] )
        except:
            print('Energy value per 100 iterations is available only for the non '
                  'linear conjugate gradient method!!!')
        return


    def norm_extract_vertex_values(self, exp_info=None):
        if exp_info == None:
            exp_info = self._exp_info
        u_norm_dof= self._norm_u.vector().get_local()
        u_norm_def_vert = self._norm_u_def.compute_vertex_values()

        mesh = self._problem.domain.get_mesh()
        # coor = mesh.coordinates()
        coor_def = self._meshy.coordinates()

        # n = self._problem.V.dim()
        n = u_norm_dof.size
        d = mesh.geometry().dim()
        if dolfin_version() <= '1.6.0':
            dof_coordinates = self._V_norm_u.dofmap().tabulate_all_coordinates(mesh)
            dof_coordinates.resize((n, d))
        else:
            dof_coordinates = self._V_norm_u.tabulate_dof_coordinates().reshape(n, d)

        mpiRank = MPI.rank(mpi_comm_world())

        results_path = "results/numpy_data/" + exp_info + '/'
        import os
        if not os.path.exists(results_path):
            try:
                os.makedirs(results_path)
            except:
                print('folder exist')

        np.save(results_path  + str(mpiRank) + 'u_norm_v', u_norm_dof)
        np.save(results_path  + str(mpiRank) + 'u_norm_def_v',
                u_norm_def_vert)

        np.save(results_path  + str(mpiRank) + 'coor_v', dof_coordinates)
        np.save(results_path  + str(mpiRank) + 'coor_def_v',
                coor_def)

        return


    def extract_dens_cell_values(self, exp_info= None):
        if exp_info == None:
            exp_info = self._exp_info
        c_centers_undef =self._get_mesh_centers(self._problem.domain.get_mesh())
        dens_undef_val = self._dens_undef.vector().get_local()

        c_centers = self._get_mesh_centers( self._meshy)
        dens_val = self._dens.vector().get_local()

        mpiRank = MPI.rank(mpi_comm_world())

        results_path = "results/numpy_data/" + exp_info +'/'

        try:
            os.makedirs(results_path)
        except:
            print('dir has been created from previous thread')

        mesh = self._problem.domain.get_mesh()
        n = dens_undef_val.size
        d = mesh.geometry().dim()
        if dolfin_version() <= '1.6.0':
            dof_coordinates = self._V_dens.dofmap().tabulate_all_coordinates(mesh)
            dof_coordinates.resize((n, d))
        else:
            dof_coordinates = self._V_dens.tabulate_dof_coordinates().reshape(n, d)


        np.save(results_path  + str(mpiRank) + 'dens_undef', dens_undef_val)
        np.save(results_path + str(mpiRank) + 'dens',dens_val)

        np.save(results_path  + str(mpiRank) + 'c_centers_undef',
                dof_coordinates)
        np.save(results_path  + str(mpiRank) + 'c_centers', c_centers)

        return

    def _get_mesh_centers(self, mesh):
        num_cells = mesh.num_cells()
        dim = mesh.coordinates().shape[1]
        c_centers = np.zeros((num_cells, dim))
        for i in xrange(num_cells):
            c = Cell(mesh, i)
            c_centers[i, 0] = c.midpoint().x()
            c_centers[i, 1] = c.midpoint().y()

        return c_centers



    def norm_over_radius_data(self,exp_info):
        if exp_info == None:
            exp_info = self._exp_info
        r_c =2
        rho, x, y  = self._problem.domain.get_circle()

        r_pnts, theta_pnts = 50, 20
        r =np.linspace(r_c,rho, r_pnts)
        theta =np.linspace(0, 2*np.pi, theta_pnts)

        norm_u = self._norm_u
        u_on_r_array = np.zeros((r_pnts, theta_pnts))

        j=0
        for th in theta:
            i =0
            for r_i in r:
                x = r_i*np.cos(th)
                y = r_i*np.sin(th)
                u_on_r_array[i,j] = norm_u([x,y])
                i +=1
            j+=1


        results_path = "results/numpy_data/" + exp_info + '/'

        try:
            os.makedirs(results_path)
        except:
            print('dir has been created from previous thread')

        np.save(results_path  + 'u_on_r_array.npy', u_on_r_array)
        np.save(results_path  + 'radius.npy', np.array([r_c, rho]))

        return

    def _iteration_plots(self, F11, F22, exp_info):
        ymin, ymax = self._y_range()

        n = 200
        y = np.linspace(ymin, ymax, n)
        F22_y = np.zeros((n, 1))
        for i in xrange(len(y)):
            F22_y[i] = F22([0, y[i]])

        n = 100
        x, y2 = self._get_between_cells_coor(n)
        F22_x = np.zeros((n, 1))
        for i in xrange(len(y2)):
            F22_x[i] = F11([x[i], y2[i]])

        self._plot(y, F22_y, 'y=0', 'F22', 'F22 (def. gradient element) on axis x = 0',
                   exp_info + '_F22_x=0')

        self._plot(x, F22_x, 'y=0', 'F11', 'F11 (def. gradient element) on axis x = 0',
                   exp_info + '_F11_y=0')

        if self._niterInfo == None:
            return

        self._niterInfo.create_plots(exp_info, self._plotsPath)

        return

    def detMin(self):
        if self._detF == None:
            J, V0 = self._problem.getJ(), self._problem.getV0()
            self._detF = calculate_detF(J, V0)
        detF = self._detF
        ymin, ymax = self._y_range()

        comm = MPI4PY.COMM_WORLD

        n = 400
        ymin, ymax = -5, 5
        y = np.linspace(ymin, ymax, n)
        detF_y = np.full((n, 1), np.inf, dtype=np.float64)
        x0 = 0.0

        mesh = self._problem.domain.get_mesh()
        for i in xrange(n):
            point = Point(x0, y[i])
            if mesh.bounding_box_tree().compute_first_entity_collision(point) < mesh.num_cells():
                detF_y[i] = detF([x0, y[i]])

        newData = comm.gather(detF_y, root=0)
        if comm.rank == 0:
            for i in xrange(newData[0].size):
                for A in newData:
                    b = A[i] != np.inf
                    if b[0]:
                        detF_y[i] = A[i]
                        break

        data = comm.bcast(detF_y, root=0)

        return data.min()


    def _y_range(self):
        domain = self._problem.domain
        if domain.__class__.__name__ == 'CircularDomain':
            rho, x0, y0 = domain.get_circle()
            ymin, ymax = y0 - rho + 0.1, y0 + rho - 0.1
        else:
            m = 'energy and F22 plot is not supported yet for object of type %s'
            raise NotImplementedError(m % domain.__name__)

        return ymin, ymax


    def _get_between_cells_coor(self, n):
        y = np.linspace(0, 0, n)
        x = np.linspace( -2, 2, n)
        return x, y

    def _plot(self, x, y, xlabel, ylabel, title, fig_name):
        fig = plt.figure()
        plt.plot(x, y)
        plt.ylim([np.min(y) - 1, np.max(y) + 1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)  #
        plt.savefig(self._plotsPath + fig_name + '.png')
        plt.close(fig)

        return




class mNewtonSolver(mSolver):
    def __init__(self, problem=None, solver = 'adaptive', res_path = 'results/',
                 error_on_nonconvergence=False, max_iter=2000, omega=1.0,
                 save_run=True):
        #		check_problem()
        mSolver.__init__(self, problem, res_path, save_run=save_run)
        self._ex_bcs = problem.get_ex_bc()

        self._omega = omega
        self._maxIterations = max_iter
        self._error_on_nonconvergence = error_on_nonconvergence
        self._absolute_tol = 1e-9
        self._relative_tol = 1e-9
        self._omegaMin = None
        self._omega_has_divided = None
        self._solvers_dict = self._init_solvers_dict()
        self._solver = self._solvers_dict[solver]

        return



    def get_omega(self):
        return self._omega

    def _init_solvers_dict(self):
        d = {}
        d['automated'] = self.automated_solution
        d['automated_new'] = self.automated_new
        d['constOmega'] = self.constOmega
        d['constOmega_storeInfo'] = self.constOmega_storeInfo
        d['adaptive'] = self.adaptive

        return d

    def automated_new(self):
        L, bcs, J = self._problem.get_L(), self._bcs, self._problem.get_Jacobian()
        if self._problem.get_k() is not None:
            bcs =[]
        nProblem = NonlinearVariationalProblem(L, self._u, bcs,  J=J)
        solver = NonlinearVariationalSolver(nProblem)

        prm = solver.parameters
        prm["newton_solver"]["relaxation_parameter"] = self._omega
        prm["newton_solver"]["maximum_iterations"] = self._maxIterations
        prm["newton_solver"]["error_on_nonconvergence"] = self._error_on_nonconvergence
        #		prm["newton_solver"]["linear_solver"] = "mumps"
        # prm["newton_solver"]["linear_solver"] = "gmres"
        #		set_log_level(PROGRESS)

        solver.solve()
        return self._u


    def automated_solution(self):
        L, bcs, J = self._problem.get_L(), self._bcs, self._problem.get_Jacobian()
        nProblem = NonlinearVariationalProblem(L, self._u, bcs, J=J)
        solver = NonlinearVariationalSolver(nProblem)

        prm = solver.parameters
        prm["newton_solver"]["relaxation_parameter"] = self._omega
        prm["newton_solver"]["maximum_iterations"] = self._maxIterations
        prm["newton_solver"]["error_on_nonconvergence"] = self._error_on_nonconvergence
        # prm["newton_solver"]["linear_solver"] = "mumps"
        #		set_log_level(PROGRESS)

        solver.solve()
        return self._u

    def constOmega(self):
        L, bcs, J = self._problem.get_L(), self._bcs, self._problem.get_Jacobian()
        maxiter, omega = self._maxIterations, self._omega
        self._niterInfo = CIterEnergyInfo(self._Pi, maxiter)

        self._step1_solution()

        eps, tol = 1.0, self._absolute_tol
        du = self._create_u()
        bcs_du =  self._adapt_bc(bcs)
        niter = 0
        while eps > tol and niter < maxiter:
            b = self._stepk_solution(J, L, bcs_du, du)
            self._u.vector().axpy(omega, du.vector())

            eps = self._print_iter_info(niter, b, du, omega)
            self._niterInfo.store_info(niter)
            niter += 1

        self._niterInfo.updateN(niter)
        return self._u


    def constOmega_storeInfo(self):
        L, bcs, J = self._problem.get_L(), self._bcs, self._problem.get_Jacobian()
        maxiter, omega = self._maxIterations, self._omega

        self._niterInfo = CIterAllInfo(self._problem, maxiter)
        self._step1_solution()

        eps, tol = 1.0, self._absolute_tol
        du = self._create_u()
        bcs_du =  self._adapt_bc(bcs)
        niter = 0
        while eps > tol and niter < maxiter:
            b = self._stepk_solution(J, L, bcs_du, du)
            self._u.vector().axpy(omega, du.vector())

            eps = self._print_iter_info(niter, b, du, omega)
            self._niterInfo.store_info(self._u, eps, niter)
            niter += 1

        self._niterInfo.updateN(niter)
        return self._u



    def adaptive(self, omegaMin=0.005):
        self._omegaMin, self._omega_has_divided = omegaMin, False

        L, bcs, J = self._problem.get_L(), self._bcs, self._problem.get_Jacobian()
        maxiter, omega = self._maxIterations, self._omega
        # self._niterInfo = CIterAllInfo(self._problem, maxiter)

        b_old = self._step1_solution()
        du, u_k = self._create_u(), self._create_u()

        eps, tol = 1.0, self._absolute_tol
        eps_old = b_old.norm('l2')
        niter, totalIter = 1, 0

        bcs_du =  self._adapt_bc(bcs)
        while eps > tol and niter < maxiter:
            b = self._stepk_solution(J, L, bcs_du, du)
            eps = self._print_iter_info_a(niter, b, du, omega, totalIter)
            omega, niter = self._check_apply_adaptation(eps, b_old, u_k, du, J, L,
                                                        omega, bcs_du, niter, eps_old)

            u_k.assign(self._u)
            self._u.vector().axpy(omega, du.vector())
            niter += 1
            totalIter += 1
            eps_old=eps

        # self._niterInfo.updateN(niter)
        self._has_converged(niter < maxiter)
        return self._u

    def _adapt_bc(self, bcs):
        bcs_du = self._homogenize(bcs)
        if self._problem.get_k() is not None:
            bcs_du =[]

        return bcs_du

    def _homogenize(self, bcs):
        bcs_hom = bcs[:]
        for bc in bcs_hom:
            bc.homogenize()
        return bcs_hom

    def _check_apply_adaptation(self, eps, b_old, u_k, du, J, L, omega, bcs_du,
                                niter, eps_old):
        # detF = self._niterInfo.store_info(eps, niter)
        if omega == self._omegaMin and self._omega_has_divided:
            self._omega_has_divided = False
            return omega, niter


        #detMin = detF.vector().array().min()
#  ------------------------- fix me...........
        if eps/eps_old > 2:
            if omega == self._omegaMin:
                return omega, niter

            elif omega > 10*self._omegaMin:
                omega = omega/10.0
            else:
                omega = self._omegaMin
            # solve again for previous u
            self._u.assign(u_k)
            self._stepk_solution(J, L, bcs_du, du)
            self._omega_has_divided = True
            niter -= 1

        if eps/b_old.norm('l2') < 1:
            if omega <= self._omega * 0.6:
                omega *= 1.5

        return omega, niter

    def _has_converged(self, b):
        if b:
            return
        print('##########################################')
        print('Newton\'s Method has not converged !!!!!!!')

        return


    def _step1_solution(self):
        L, bcs, J = self._problem.get_L(), self._bcs, self._problem.get_Jacobian()

        if self._has_initialized:
            A, b = assemble_system(J, -L) #, self._homogenize(bcs))
            return b

        A, b = assemble_system(J, -L, bcs)
        solve(A, self._u.vector(), b)
        return b

    def _stepk_solution(self, J, L, bcs, u):
        A, b = assemble_system(J, -L, bcs)

        solve(A, u.vector(), b, 'mumps')
        return b

    def _print_iter_info(self, niter, b, du, omega):
        eps_sup = du.vector().norm('linf')
        eps_l2 = b.norm('l2')

        mpiRank = MPI.rank(self._comm)
        if 0 == mpiRank:
            print(' Newton iteration {0:1d} (L2 norm) = {1:3.3e}, sup norm = \
{2:3.3e}, omega = {3:g}'.format(niter, eps_l2, eps_sup, omega) )

        return eps_l2


    def _print_iter_info_a(self, niter, b, du, omega, totalIter):
        eps_sup = du.vector().norm('linf')
        eps_l2 = b.norm('l2')

        mpiRank = MPI.rank(self._comm)
        if 0 == mpiRank:
            print('total iter =%d'%(totalIter) )
            print(' Newton iteration {0:1d} (L2 norm) = {1:3.3e}, sup norm = \
{2:3.3e}, omega = {3:g}'.format(niter, eps_l2, eps_sup, omega) )

        return eps_l2

    def _create_u(self):
        V = self._problem.V
        return Function(V)

    def _apply_bcs(self):
        for bc in self._bcs:
            bc.apply(self._u.vector())
        return

    def set_omega(self, omega):
        self._omega = omega
        return

    def set_maxIterations(self, maxIterations):
        self._maxIterations = maxIterations
        return

    def set_absolute_tollerance(self, tol):
        self._absolute_tol = tol
        return

    def set_relative_tollerance(self, tol):
        self._relative_tol = tol
        return

    def make_iteration_plots(self, experiment_info):
        if self._niterterInfo == None:
            print('The is no iteration info to plot')
            return
        self._niterterInfo.make_plots(experiment_info)
        return


class m_ncgSolver(mSolver):
    def __init__(self, problem, res_path = 'results/', solver='fixed_centers',
                 save_run = False, full_output = False):
        mSolver.__init__(self, problem, res_path, save_run = save_run)
        # if problem not a minimization problem error......

        self._N = self._u.vector().size()
        self._niter = 0
        # self._niterInfo = CIterEnergyInfo(self._Pi, 20*self._N)
        self._J, self._V0 = problem.getJ(), problem.getV0()
        self._L = problem.get_L()
        self._cells_list = None
        self._cells_num = None
        self._solvers_dict = self._init_solvers_dict()
        self._solver = self._solvers_dict[solver]


        self._free_i=None
        self._x0 = self._u.vector().copy() # initial search point
        self._x0_free_i=None
        self._grad_x0 = None #self._x0.copy()
        self._grad_x0_c = None #self._x0.copy()
        self._aPi = None

        self._full_output = full_output
        self.moving_centers = False
        self._energy_i = np.zeros(self._N)

        return

    def _init_solvers_dict(self):
        d = {}
        d['fixed_centers'] = self.fixed_centers_solver
        d['free_centers'] = self.free_centers_solver

        return d


    def get_total_energy_val(self):
        if self._niter == None or self._aPi == None:
            return -1, -1
        # return self._niterInfo.get_energy_val()
        return self._niter, self._aPi


    def f(self, x, *args):
        self._u.vector().set_local(x.get_local(), self._free_i)
        self._u.vector().apply('')

        aPi = assemble(self._Pi)

        if 0 == self._mpiRank and self._niter%100==0:
            print(' %d %.12f ' % (self._niter, aPi))
            self._energy_i[np.int(self._niter/100)] = aPi

        if self._niter % 200000 == 0:
            if self._niter > 1:
                self.plot_results(self._exp_info + '/temp' , final_save=False)
                self.save_all_function_info(self._exp_info + '/temp' )

        self._niter += 1
        return aPi

    def grad_f(self, x, *args):
        self._u.vector().set_local(x.get_local(), self._free_i)
        self._u.vector().apply('')
        # DP_v = assemble(self._L)

        self._grad_x0.set_local(assemble(self._L).get_local()[self._free_i])
        self._grad_x0.apply('')
        # return self._grad_x0.copy()

        self._grad_x0_c.set_local( self._grad_x0.get_local() )
        self._grad_x0_c.apply('insert')
        return self._grad_x0_c

    def fixed_centers_solver(self):
        self._find_free_indices()
        self._create_search_vector()

        self._x0.set_local(self._u.vector().get_local()[self._free_i])
        self._x0.apply('')


        u_v, warnflag = fmin_cg(self.f, self._x0, fprime=self.grad_f, return_all=True)

        self._u.vector().set_local(u_v.get_local(), self._free_i)
        self._u.vector().apply('')
        self._aPi = assemble(self._Pi)


        if self._full_output == True:
            return self._u, warnflag
        return self._u


    def solve_old(self):
        if self._problem.get_k() == 0:
            print('k is undefined')
            return


        x = self._u.vector().array()
        print('in solve', x.shape)
        from scipy.optimize import fmin_cg
        # from ncg_parallel.ncg_PETS import minimize_cg as fmin_cg
        u_array = fmin_cg(self.f_old, x, fprime=self.grad_f_old)
#         u_array = optimize.minimize(self._fun, x, method='Nelder-Mead',  jac=self.grad_f)
#         self._niterInfo.updateN(self._niter)
        self._u.vector()[:] = u_array

        return self._u

    def f_m(self, x, *args):
        self._u.vector().set_local(x.get_local(), self._free_i)
        self._u.vector().apply('')

        x_centers_ar = self._gather_center_movement(x)
        self._move_centers(x_centers_ar)

        aPi = assemble(self._Pi)

        if 0 == self._mpiRank and self._niter%100==0:
            print(' %d %.12f' % (self._niter, aPi))
            self._energy_i[np.int(self._niter / 100)] = aPi
        self._niter += 1

        if self._niter % 200000 == 0:
            if self._niter > 1:
                self.plot_results(self._exp_info + '/temp' , final_save=False)
                self.save_all_function_info(self._exp_info + '/temp' )

        return aPi

    def grad_f_m(self, x, *args):
        self._u.vector().set_local(x.get_local(), self._free_i)
        self._u.vector().apply('')

        x_centers_ar = self._gather_center_movement(x)
        self._move_centers(x_centers_ar)
        self._grad_x0.set_local(assemble(self._L).get_local()[self._free_i], self._x0_free_i)
        self._grad_x0.apply('')

        self._centre_partial_derivatives(x)
        # self._grad_x0.apply('')
        # return self._grad_x0.copy()

        self._grad_x0_c.set_local(self._grad_x0.get_local())
        self._grad_x0_c.apply('insert')
        # if 0 == self._mpiRank:
        #     print('%.12f' % (self._grad_x0_c.sum()))
        return self._grad_x0_c


    def _gather_center_movement(self, x):
        i_start = x.size() - 2*self._cells_num
        i2copy = np.array( np.linspace(i_start, x.size()-1, 2*self._cells_num),
                           dtype='intc')

        if dolfin_version() <= '1.6.0':
            x_move_c = PETScVector()
        else:
            x_move_c = PETScVector(mpi_comm_self())
        x.gather(x_move_c, i2copy)  # modify me !!!!!
        x_move_c.apply('')

        return x_move_c.get_local()

    def _gather_center_movement_mpi4py(self, x):
        i_start = x.size() - 2*self._cells_num
        i2copy = np.array( np.linspace(i_start, x.size()-1, 2*self._cells_num),
                           dtype='intc')

        x_move_c = PETScVector()
        x.gather(x_move_c, i2copy)  # modify me !!!!!
        x_move_c.apply('')
        if dolfin_version() <= '1.6.0':
            return x_move_c.array()

        mpiRank, mpiSize = MPI.rank(self._comm), MPI.size(self._comm)

        from mpi4py.MPI import DOUBLE
        comm2 = self._comm.tompi4py()
        if mpiRank == 0:
            x_move_c_ar = x_move_c.get_local()
            for i in range(1, mpiSize):
                comm2.Send([x_move_c_ar, DOUBLE], dest=i, tag=i)
        else:
            x_move_c_ar = np.empty(i2copy.size, dtype='double')
            comm2.Recv([x_move_c_ar, DOUBLE], source=0, tag=mpiRank)

        return x_move_c_ar

    def free_centers_solver(self):
        if self._problem.get_k() == None:
            raise ValueError('k is undefined, cell movement is supported only'
                             'for k>0')


        self.moving_centers = True

        self._cells_list, self._cells_num = self._get_interior_circles_num()

        self._find_free_indices()
        self._create_search_vector()

        # ????????
        self._x0.set_local(self._u.vector().get_local()[self._free_i], self._x0_free_i)
        self._init_cells_centers()
        self._x0.apply('')

        u_v, warnflag = fmin_cg(self.f_m, self._x0, fprime=self.grad_f_m, return_all=True)
        # self._niterInfo.updateN(self._niter)
        self._u.vector().set_local(u_v.get_local(), self._free_i)
        self._u.vector().apply('')
        self._aPi = assemble(self._Pi)
        # print 'centres: ', u_array[-2*self._cells_num:]


        if self._full_output == True:
            return self._u, warnflag
        return self._u

    def _cell_center_displacement_array(self, domain):
        d = domain.get_center_displacement()
        vals = np.zeros(2)

        d.eval(vals, np.zeros(2))
        return vals



    def _init_cells_centers(self):
        g = np.zeros(2*self._cells_num, dtype='double')
        i =0
        for mdomain in self._cells_list:
            g[i:i+2] = self._cell_center_displacement_array(mdomain)
            i+=2 # +=3 in case of 3D problem.


        mpiRank, mpiSize = MPI.rank(self._comm), MPI.size(self._comm)
        if mpiRank == mpiSize - 1:
            i_start = self._x0.local_size() - 2 * self._cells_num
            i2copy = np.array(np.linspace(i_start, self._x0.local_size() - 1, 2 * self._cells_num)
                , dtype='intc')
            self._x0.set_local(g, i2copy)

        self._x0.apply('')
        return

    def _get_interior_circles_num(self):
        mdomain = self._problem.domain
        sList_t = mdomain.get_subDomainsList()
        sList = sList_t[:]

        for sdomain in sList:
            if sdomain.__class__.__name__ is not 'CircularDomain':
                sList.remove(sdomain)
                warnings.warn("\n cell movement is has not been implemented for non CirclularDomain"
                              " object")
        # if len(sList) == 1:
        #     warnings.warn("\n cell movement is not possible, matrix contains only"
        #                   "one cell.")
        return sList, len(sList)
        # return sList[1:], len(sList) -1
        # return [sList[0]], 1

    # 3D case must be considered.
    def _move_centers(self, x):
        i = 0
        for mdomain in self._cells_list:
            mdomain.set_center_displacement(Constant( (x[i], x[i+1])) )
            i+=2 # +=3 in case of 3D problem.

        return

    def _centre_partial_derivatives(self, x):
        g = np.zeros(2*self._cells_num, dtype='double')

        i= 0; j=0
        # surf_int_val = self._surface_integrals(x)
        # while i < 2*self._cells_num:
        #     g[i:i+2] = self._i_centre_derivative_old2(i, x, surf_int_val)
        #     i +=2
        while i < 2*self._cells_num:
            g[i:i+2] = self._i_centre_derivative_new(j)
            i +=2
            j+=1

        mpiRank, mpiSize = MPI.rank(self._comm), MPI.size(self._comm)
        if mpiRank == mpiSize - 1:
            i_start = x.local_size() - 2 * self._cells_num
            i2copy = np.array(np.linspace(i_start, x.local_size() - 1, 2 * self._cells_num)
                , dtype='intc')
            self._grad_x0.set_local(g, i2copy)

        self._grad_x0.apply('')
        return

    def _i_centre_derivative_new(self, i):
        '''
        Given the total potential energy approximate the derivative
         of the center of cirlce with number i.
        :param i: is the #circle
        :param x: PETSC vector, contains the values of the displacement vector
         and in the last indices the centers of the circles
        :param aPi: given the vector x the total potential energy

        :return:
        '''
        mdomain = self._cells_list[i]
        d0, d1 = self._problem.derivative_over_center[mdomain]
        ad0=  assemble(d0)
        ad1 =assemble(d1)


        return ad0, ad1


    # !!!!!! _i_centre_derivative_old2 is newer than _i_centre_derivative_old
    def _i_centre_derivative_old2(self, i, x, surface_int_val_old):
        '''
        Given the total potential energy approximate the derivative
         of the center of cirlce with number i.
        :param i: is the #circle
        :param x: PETSC vector, contains the values of the displacement vector
         and in the last indices the centers of the circles
        :param aPi: given the vector x the total potential energy

        :return:
        '''

        h= 1e-10

        # store infinitensimal movement value to local vector (center values
        #  are stored to the last process)
        mpiRank, mpiSize = MPI.rank(self._comm), MPI.size(self._comm)
        if mpiRank == mpiSize-1:
            i_start = x.local_size() - 2 * self._cells_num
            self._update_vector_i_val(x, x.get_local()[ i_start + i] + h, i_start+i)

        x.apply('')
        grad_f1 = (self._surface_integrals(x) -surface_int_val_old)/h
        if mpiRank == mpiSize - 1:
            self._update_vector_i_val(x, x.get_local()[i_start + i] - h,
                                      i_start + i)
            self._update_vector_i_val(x, x.get_local()[i_start + i+1] + h,
                                      i_start + i+1)

        x.apply('')
        grad_f2 = (self._surface_integrals(x) -surface_int_val_old)/h
        if mpiRank == mpiSize - 1:
            self._update_vector_i_val(x, x.get_local()[i_start + i+1] - h,
                                      i_start + i+1)

        x.apply('')
        return grad_f1, grad_f2

    def _centre_partial_derivatives_old(self, x):
        g = np.zeros(2*self._cells_num, dtype='double')
        aPi = self._f_m2(x)

        i= 0
        while i < 2*self._cells_num:
            g[i:i+2] = self._i_centre_derivative_old(i, x, aPi)
            i +=2

        mpiRank, mpiSize = MPI.rank(self._comm), MPI.size(self._comm)
        if mpiRank == mpiSize - 1:
            i_start = x.local_size() - 2 * self._cells_num
            i2copy = np.array(np.linspace(i_start, x.local_size() - 1, 2 * self._cells_num)
                , dtype='intc')
            self._grad_x0.set_local(g, i2copy)

        self._grad_x0.apply('')
        return

    def _i_centre_derivative_old(self, i, x, aPi):
        '''
        Given the total potential energy approximate the derivative
         of the center of cirlce with number i.
        :param i: is the #circle
        :param x: PETSC vector, contains the values of the displacement vector
         and in the last indices the centers of the circles
        :param aPi: given the vector x the total potential energy

        :return:
        '''

        h= 1e-7

        # store infinitensimal movement value to local vector (center values
        #  are stored to the last process)
        mpiRank, mpiSize = MPI.rank(self._comm), MPI.size(self._comm)
        if mpiRank == mpiSize-1:
            i_start = x.local_size() - 2 * self._cells_num
            self._update_vector_i_val(x, x.get_local()[ i_start + i] + h, i_start+i)

        x.apply('')
        grad_f1 = (self._f_m2(x) - aPi)/h
        if mpiRank == mpiSize - 1:
            self._update_vector_i_val(x, x.get_local()[i_start + i] - h,
                                      i_start + i)
            self._update_vector_i_val(x, x.get_local()[i_start + i+1] + h,
                                      i_start + i+1)

        x.apply('')
        grad_f2 = (self._f_m2(x) - aPi)/h
        if mpiRank == mpiSize - 1:
            self._update_vector_i_val(x, x.get_local()[i_start + i+1] - h,
                                      i_start + i+1)

        x.apply('')
        return grad_f1, grad_f2

    def _update_vector_i_val(self, v, val, i):
        val_ar = np.array([val], dtype='double')
        v.set_local(val_ar, np.array([i], dtype='intc'))

        return

    def _f_m2(self, x, *args):
        self._u.vector().set_local(x.get_local(), self._free_i)
        self._u.vector().apply('')

        x_centers_ar = self._gather_center_movement(x)
        self._move_centers(x_centers_ar)


        aPi = assemble(self._Pi)
        return aPi

    def _surface_integrals(self, x, *args):
        self._u.vector().set_local(x.get_local(), self._free_i)
        self._u.vector().apply('')

        x_centers_ar = self._gather_center_movement(x)
        self._move_centers(x_centers_ar)
        surf_int_val = assemble(self._problem.boundPi)


        return surf_int_val


    def _find_free_indices(self):

        u0 = Function(self._problem.V)
        if self._problem.get_k() == None:
            self._apply_bc(u0, self._extract_bcs())

            if self._has_initialized == False:
                self._apply_bc(self._u, self._problem.get_bc())
            else:
                Warning('check if initialization boundary conditions are '
                        'compatible with the given boundary conditions')
        else:
            self._apply_exterior_bcs(self._u)

        self._mark_exterior_bcs(u0)
        free_ind = np.nonzero(u0.vector().get_local() == 0.0)
        free_i = free_ind[0]

        self._free_i = free_i.astype('intc')
        self._x0_free_i = np.array( np.linspace(0, len(self._free_i)-1,
                                                len(self._free_i)), dtype='intc')
        return

    def _apply_bc(self, u, bcs):
        for bc in bcs: #self._problem.get_bc():
                bc.apply(u.vector())
        return

    def _extract_bcs(self):

        domain = self._problem.domain
        sub_domain_list = domain.get_subDomainsList()
        bcs = []

        if dolfin_version() >= '2016.2.0':
            # print(self._problem.el_order() )
            u0 = Expression(('1.0', '1.0'), degree =self._problem.el_order() )
        else:
            u0 = Expression(('1.0', '1.0'))

        for sub_domain in sub_domain_list:
            if sub_domain.has_bc():
                if sub_domain.__class__.__name__ == 'RectangularDomain':
                    Gamma_list = sub_domain.get_bnd_function()
                    for Gamma in Gamma_list:
                        bc = DirichletBC(self._problem.V, u0, Gamma)
                        bcs.append(bc)
                else:
                    Gamma = sub_domain.get_bnd_function()
                    bc = DirichletBC(self._problem.V, u0, Gamma)
                    bcs.append(bc)

        return bcs

    def _mark_exterior_bcs(self, u):
        if self.init_name == 'fun3':
            self._mark_Rectangular_bnds_partially(u)
            return

        domain = self._problem.domain
        if dolfin_version() >= '2016.2.0':
            u0 = Expression(('1.0', '1.0'), degree=self._problem.el_order())
        else:
            u0 = Expression(('1.0', '1.0'))
        if domain.has_bc():
            if domain.__class__.__name__ == 'RectangularDomain':
                Gamma_list = domain.get_bnd_function()
                for Gamma in Gamma_list:
                    bc = DirichletBC(self._problem.V, u0, Gamma)
                    bc.apply(u.vector())
                return 
            
            Gamma = domain.get_bnd_function()
            bc = DirichletBC(self._problem.V, u0, Gamma)
            bc.apply(u.vector())

        return

    def _mark_Rectangular_bnds_partially(self, u):
        leftNright_ex = CompiledSubDomain("(near(x[0], 0) || near(x[0], 1) )&& on_boundary")
        upNdown_ex = CompiledSubDomain("( near(x[1], 0) || near(x[1], 1) ) && on_boundary")

        if dolfin_version() == '2016.2.0':
            leftNright = Expression(("1.0", "0.0"),degree=self._problem.el_order())
            upNdown = Expression(("0.0", "1.0"),degree=self._problem.el_order())
        else:
            leftNright = Expression(("1.0", "0.0"))
            upNdown = Expression(("0.0", "1.0"))
        V = self._problem.V
        bcl = DirichletBC(V, leftNright, leftNright_ex)
        bcr = DirichletBC(V, upNdown, upNdown_ex)

        bcl.apply(u.vector())
        bcr.apply(u.vector())

        return





    def create_bc3(self, V):
        self._u0 = ' deform rectangle'




    def _apply_exterior_bcs(self, u):
        domain = self._problem.domain

        if domain.has_bc():
            bcs = self._problem.get_bc()
            bc = bcs[-1]
            bc.apply(u.vector())

        return

    def _create_search_vector(self):
        free_i_num_ar = self._free_indices_num()
        # free_i_num_ar = free_i_num.array()

        mpiRank = MPI.rank(self._comm)
        mpiSize = MPI.size(self._comm)

        self._x0 = PETScVector()
        f_i = free_i_num_ar.astype('intc')

        prev_sum = sum(f_i[:mpiRank])
        if mpiRank < mpiSize -1:
            if dolfin_version() >= '2017.2.0':
                self._x0.init( (prev_sum, int(prev_sum + f_i[mpiRank])))
            else:
                self._x0.init(self._comm, (prev_sum, int(prev_sum + f_i[mpiRank])))
        else:
            if self.moving_centers:
                i4_centers_mov = 2 * self._cells_num  # in 3D: 2 must be replaced from 3
            else:
                i4_centers_mov = 0

            if dolfin_version() >= '2017.2.0':
                self._x0.init((prev_sum, int(prev_sum + f_i[mpiRank] +
                                                     i4_centers_mov)))
            else:
                self._x0.init(self._comm, (prev_sum, int(prev_sum + f_i[mpiRank] +
                                                         i4_centers_mov)))
        self._grad_x0= self._x0.copy()
        self._grad_x0_c = self._grad_x0.copy()
        self._grad_x0.apply('insert'); self._grad_x0_c.apply('insert')
        return

    def _free_indices_num(self):
        """
        Creates and returns a vector which in the index mpiRank contains
        the number len(free_i)
        """
        mpiRank = MPI.rank(self._comm)

        free_i_num_local = PETScVector()
        if dolfin_version() >= '2017.2.0':
            free_i_num_local.init( (mpiRank, mpiRank + 1))
        else:
            free_i_num_local.init(self._comm, (mpiRank, mpiRank + 1))
        free_i_num_local.set_local(np.array([len(self._free_i)], dtype='double'))

        mpiSize = MPI.size(self._comm)

        if dolfin_version() <= '1.6.0':
            numOf_free_i = PETScVector()
        else:
            numOf_free_i = PETScVector(mpi_comm_self())
        free_i_num_local.gather(numOf_free_i,
                np.linspace(0, mpiSize - 1, mpiSize).astype(
                                    'intc'))
        return numOf_free_i.get_local()

    def _free_indices_num_mpi4py(self):
        """
        Creates and returns a vector which in the index mpiRank contains
        the number len(free_i)
        """
        mpiRank = MPI.rank(self._comm)

        free_i_num_local = PETScVector()
        free_i_num_local.init(self._comm, (mpiRank, mpiRank + 1))
        free_i_num_local.set_local(np.array([len(self._free_i)], dtype='double'))

        mpiSize = MPI.size(self._comm)

        if dolfin_version() <= '1.6.0':
            numOf_free_i = PETScVector()
            free_i_num_local.gather(numOf_free_i,
                                    np.linspace(0, mpiSize - 1, mpiSize).astype(
                                        'intc'))
            return numOf_free_i.get_local()

        numOf_free_i = np.zeros(mpiSize, dtype='intc')
        comm2 = self._comm.tompi4py()
        from mpi4py.MPI import INT32_T
        comm2.Allgather([free_i_num_local.get_local().astype('intc'), INT32_T], [numOf_free_i, INT32_T])

        return numOf_free_i

    # def solve2(self):
    #     solve(self._Pi == 0, u, solver_parameters={'snes_type': 'ncg'})
class m_ncgSolver2(mSolver):
    def __init__(self, problem,  res_path = 'results/'):
        mSolver.__init__(self, problem, res_path, save_run=False)
        # if problem not a minimization problem error......

        # self._niter = 0
        #self._niterInfo = CIterEnergyInfo(self._Pi, maxiter)
        self._Jacobian = problem.get_Jacobian()
        self._L = problem.get_L()


        return




    def solve(self):
        # Create the PETScTAOSolver
        solver = PETScTAOSolver()
        solver = PETScSNESSolver()
        # Set some parameters
        # solver.parameters['linear_solver']='superlu_dist'
        solver.parameters["method"] = "SNESQN" #"ncg"
        # solver.parameters["method"] = "aspin"
        #solver.parameters["monitor_convergence"] = True
        solver.parameters["report"] = True
        max_iter = 8*self._u.vector().get_local().shape[0]
        # solver.parameters["maximum_iterations"] = max_iter
        # solver.parameters["function_absolute_tol"] = 1e-7
        solver.parameters["relative_tolerance"] = 1e-17
        # solver.parameters["gradient_absolute_tol"] = 1e-5

        #solver.parameters["gradient_relative_tol"] = 1e-5
        # PETScOptions.set('tao_cg_type', 'prp')
        # PETScOptions.set('tao_cg_eta', '0.1')
        #PETScOptions.set('tao_ls_type', 'armijo')

        # PETScOptions.set('tao_ls_type', 'more-thuente')
        #PETScOptions.set('tao_ls_stepmin', '1e-6')
        #PETScOptions.set('tao_ls_stepmax', '1')
        # PETScOptions.set('tao_max_funcs', max_iter)
        # PETScOptions.set('tao_delta_min', '1e-7')
        # PETScOptions.set('tao_delta_max', '50')
        # solver.parameters["line_search"] = 'gpcg'
        # solver.parameters["line_search"] = 'unit'

        # parameters.parse()

        # Solve the problem
        snes_solver_parameters = {"method": "qn",
                                  "relative_tolerance":1e-12,
                                  'line_search':"l2",
                                  "maximum_iterations":max_iter,
                                  "solution_tolerance":1.0e-6,
                                  "maximum_residual_evaluations":10000}

        optProblem = MyOptimizationProblem(self._u, self._Pi, self._L, self._Jacobian)
        solver.parameters.update(snes_solver_parameters)
        solver.solve(optProblem, self._u.vector())
        print('hello thereeeee')
        return self._u

class MyOptimizationProblem(OptimisationProblem):

    def __init__(self, u, Pi, DPi, DDPi):
        OptimisationProblem.__init__(self)
        self._Pi = Pi
        self._DPi = DPi
        self._DDPi = DDPi
        self._u = u
        self._niter = 0

    # Objective function
    def f(self, x):
        self._u.vector()[:] = x
        #as_backend_type(self._u.vector()).update_ghost_values()

        self._niter += 1
        aPi = assemble(self._Pi)
        print (self._niter, aPi)
        return aPi

    # Gradient of the objective function
    def F(self, b, x):
        self._u.vector()[:] = x
        #as_backend_type(self._u.vector()).update_ghost_values()
        assemble(self._DPi, tensor=b)
       # print 'DPi sum = ', b.array().sum()

    # Hessian of the objective function
    def J(self, A, x):
        self._u.vector()[:] = x
        assemble(self._DDPi, tensor=A)



class m_ncgSolver3(mSolver):
    def __init__(self, problem, maxiter = 80000, res_path = 'results/'):
        mSolver.__init__(self, problem, res_path)
        # if problem not a minimization problem error......

        #self._niter = 0
        #self._niterInfo = CIterEnergyInfo(self._Pi, maxiter)
        self._Jacobian = problem.get_Jacobian()
        self._L = problem.get_L()


        return




    def solve(self):
        # Create the PETScTAOSolver
        solver = PETScSNESSolver()

        # Set some parameters
        solver.parameters["method"] = "ncg"
        PETScOptions.set('snes_ncg_monitor', True)
        # PETScOptions.set('snes_ncg_type', 'fr')
        PETScOptions.set('snes_ncg_type', 'prp')
        # PETScOptions.set('snes_linesearch_type', 'cp')

        #solver.parameters["monitor_convergence"] = True
        solver.parameters["report"] = True
        solver.parameters["maximum_iterations"] = self._u.vector().array().shape[0]

        parameters.parse()

        # Solve the problem

        optProblem = MyOptimizationProblem(self._u, self._Pi, self._L, self._Jacobian)
        solver.solve(optProblem, self._u.vector())

        return self._u
