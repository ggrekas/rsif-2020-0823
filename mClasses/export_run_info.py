from dolfin import*
import os
import numpy as np

# TODO: test consequences of collapsed run
# TODO: test if path is given appropriately with the character '\'

class runInfo:
    def __init__(self, domain, problem, solver, res_path, resolution,
                 melements = 'C Lagrange 1'):
        self._domain = domain
        self._problem = problem
        self._solver = solver
        self._file_name = res_path + 'simulations'
        self._resolution = resolution
        self._elements = melements

        comm = mpi_comm_world()
        mpiSize = MPI.size(comm)
        self._parallel = mpiSize > 1
        self._threads_num = mpiSize

        self._total_time = 0
        self._infile= None
        self._domain_info = None
        self._problem_info = None
        self._solver_info = None
        self._run_num = None
        return

    def set_execution_time(self, total_time):
        self._total_time = total_time
        return

    def get_previous_run_num(self):
        file_name = self._file_name + '.txt'

        comm = mpi_comm_world()
        mpiRank =MPI.rank(comm)
        if mpiRank == 0:
            self._execute_command('cp ' + file_name +' ' +self._file_name + '_back_up.txt' )
            self._infile = open(file_name, 'r+')
            self._lock_file()

            lines = self._infile.readlines()
            current_run = int(lines[0]) + 1
            self._run_num = current_run

        self._run_num = self._gather_run_num()
        self._push_current_run_num()
        return self._run_num -1

    def _gather_run_num(self):
        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)

        vec = PETScVector()
        if dolfin_version() >= '2017.2.0':
            vec.init( (mpiRank, mpiRank + 1) )
        else:
            vec.init(comm, (mpiRank, mpiRank + 1))
        if mpiRank ==0:
            vec.set_local(np.array([self._run_num +0.0]) )

        if dolfin_version() <= '1.6.0':
            v_run_num = PETScVector()
        else:
            v_run_num = PETScVector(mpi_comm_self())

        vec.gather(v_run_num, np.array([0], dtype='intc'))

        print('extract info...', int(v_run_num.get_local()) )
        return int(v_run_num.get_local())

    def _lock_file(self):
        try:
            from fcntl import flock, LOCK_EX, LOCK_NB
        except ImportError:
            exit("ERROR: Locking is not supported on this platform.")
        import time

        while True:
            try:
                flock(self._infile.fileno(), LOCK_EX | LOCK_NB)
                break
            except:
                print("Waiting: Another amp process is already running.")
                time.sleep(5)
        return


    def _push_current_run_num(self):
        # file_name = self._file_name + '.txt'

        line = str(self._run_num) + '\n'

        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)

        if mpiRank ==0:
            outfile = self._infile #open(file_name, 'w')
            outfile.seek(0)
            outfile.truncate()
            outfile.write(line)

            outfile.close()

        return


    def _get_run_info(self):
        self._domain_info = self._get_domain_info()
        self._problem_info = self._get_problem_info()
        self._solver_info = self._get_solver_info()
        return


    def _get_domain_info(self):
        domain = self._domain
        s1, s2 = self._domain_type(domain)
        domains_t, domains_coor ='    <td>' + s1.upper(), '    <td>' + s2

        bc_type = '    <td>' +domain.bc_type()

        subDomains_List = domain.get_subDomainsList()
        for msubDomain in subDomains_List:
            s1, s2 = self._domain_type(msubDomain)
            domains_t, domains_coor = domains_t + s1, domains_coor + ' <br> '+ s2
            bc_type += ' <br> ' + msubDomain.bc_type()

        bc_type += ' </td>\n'
        # mesh = domain.get_mesh()
        u = self._problem.get_u()
        domains_t += ' <br> vector size=' +str(u.vector().size()) +\
                     '<br>' + 'resolution='+str(self._resolution) +'</td>\n'
        domains_coor += ' </td>\n'

        return (domains_t, domains_coor, bc_type)




    def _domain_type(self, domain):
        if domain.__class__.__name__ == 'CircularDomain':
            m = self._get_circle_info(domain)
            return 'c', m
        elif domain.__class__.__name__ == 'RectangularDomain':
            x1, y1, x2, y2 = domain.get_corners()
            m = '(x1,y1)=(%g,%g), (x2,y2)=(%g,%g)'%(x1, y1, x2, y2)
            if domain.has_uniform_mesh():
                m += '<br> Uniform mesh'
            return 'r', m
        elif domain.__class__.__name__ == 'EllipticalDomain':
            c, a, b = domain.get_ellipsis()
            m = 'center=(%g,%g), a=%g, b= %g' % (c[0], c[1], a, b)
            return 'e', m
        else:
            m = 'cannot get info for domain of type %s'
            raise NotImplementedError(m % domain.__class__.__name__)

    def _get_circle_info(self, domain):
        rho, x, y = domain.get_circle()
        m = 'r=%g, (x0,y0) = (%g,%g)'%(rho, x, y)
        d = domain.get_center_displacement()

        vals = np.zeros(2) # 2D case
        d.eval(vals, np.zeros(2))

        m += '->' + '(%.3f,%.3f)'%(vals[0] + x, vals[1] +y)

        return m


    def _get_problem_info(self):
        problem = self._problem
        solver = self._solver
        m2 = '    <td>' + problem.get_psi_name() + '</td>\n'
        if problem.__class__.__name__ == 'mNonlinearVariationalProblem':
            m1 = '    <td> Weak-Form </td>\n'
            return m1 + m2
        elif problem.__class__.__name__ == 'mEnergyMinimizationProblem':
            m1 = '    <td>Energy Min, k=%g, '%(problem.get_k()) + problem.cell_model()

            if solver.__class__.__name__ == 'mNewtonSolver':
                m1 += '<br>total energy=%g' % (solver.get_total_energy_val())
            elif solver.__class__.__name__ == 'm_ncgSolver':
                it_ener = solver.get_total_energy_val()
                m1 += '<br>i=%g, energy=%g'%(it_ener[0],it_ener[1] )

            epsilon = problem.get_epsilon()
            if epsilon != None:
                m1 += '<br>' + 'epsilon=%g'%(epsilon)
                m1 += '<br>' + 'alpha=%g'%(problem.get_alpha())

            par_info = problem.parameters_info()
            m1 += '<br>' + par_info
            m1 +=    '</td>\n'
            return m1 + m2

        return ['<td>undefined</td>']

    def _get_solver_info(self):
        solver = self._solver
        s_init = ''
        if solver.has_initialized:
            s_init = '<br>initialized'
            l1, l2 = self._solver.get_init_l1_l2()
            if l1 != None:
                s_init = s_init +'<br>' + 'l1=' +str(l1) + ', l2 = ' + str(l2)
                s_init = s_init + '<br>' + solver.init_name

        if solver.__class__.__name__ == 'mNewtonSolver':
            m = '    <td>'
            m+= 'Newton, omega =%g'%(solver.get_omega())+' <br> ' + solver.get_name()
            return m + s_init + '</td>\n'
        elif solver.__class__.__name__ == 'm_ncgSolver':
            if solver.moving_centers:
                return '    <td> ncg  moving centers' + s_init + '</td>\n'
            else:
                return '    <td> ncg' + s_init + '</td>\n'
        elif solver.__class__.__name__ == 'm_ncgSolver2':
            return '    <td> ncg_tao' + s_init + '</td>\n'
        m = 'cannot get info for solver of type %s'
        raise NotImplementedError(m % solver.__class__.__name__)



    def export_run(self, test_mode= False):
        if test_mode:
            return

        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)
        if mpiRank != 0:
            return
        self._get_run_info()

        file_name = self._file_name + '.html'
        self._execute_command('cp '+file_name +' '+self._file_name+'_back_up.html')
        infile = open(file_name, 'r')
        lines = infile.readlines()
        infile.close()
        line_num = -3
        lines.insert(line_num, '  <tr>\n')
        lines = self._insert_info(lines, line_num)
        lines.insert(line_num, '  </tr>\n')
        
        outfile = open(file_name, 'w')
        outfile.writelines(lines)
        outfile.close()

        return


    def _insert_info(self, lines, line_num):
        lines.insert(line_num,'    <td>' + str(self._run_num) + '</td>\n')
        lines.insert(line_num, self._domain_info[2])  #  boundary conditions
        lines.insert(line_num, self._domain_info[0])  #  Domain type
        lines.insert(line_num, self._domain_info[1])  #  Domain coordinates

        lines.insert(line_num, self._solver_info)
        lines.insert(line_num, self._problem_info)
        lines.insert(line_num, '    <td>' + self._elements +  '</td>\n')
        lines.insert(line_num, '    <td>' + str(self._parallel) + '<br> threads='+ str(self._threads_num) + '</td>\n')
        total_time = '    <td> %g </td>\n'%(self._total_time)
        lines.insert(line_num, total_time)
        lines.insert(line_num, '    <td>' + self._get_version() +  '</td>\n')

        return lines


    def _get_version(self):
        cmd = 'git  tag > tempVersion.txt'
        self._execute_command(cmd)

        infile = open('tempVersion.txt', 'r')
        lines = infile.readlines()
        infile.close()

        cmd = 'rm tempVersion.txt'
        self._execute_command(cmd)
        s =lines[-1]
        print('version = ', s[:-1])

        return s[:-1]

    def _execute_command(self, cmd):
        failure = os.system(cmd)
        if failure:
            print('Execution os "%s" failed!\n' % cmd)
            os.sys.exit(1)
        return
