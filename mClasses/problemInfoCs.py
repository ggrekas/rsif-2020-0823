from dolfin import *
import numpy as np
#import matplotlib.pyplot as plt
from ProblemCs import *


# TODO calculate_detF_F22, calculate_detF etc... delete code replication

class CIterEnergyInfo:
    def __init__(self, Pi, maxIter):  # fun = 'all_info' or 'info'
        self._N = 0
        self._Pi = Pi
        self._energy = np.zeros(maxIter)

        return

    def store_info(self, niter):
        self._energy[niter] = assemble(self._Pi)
        return

    def store_computed_info(self, niter, aPi):
        self._energy[niter] = aPi
        return

    def updateN(self, niter):
        self._N = niter
        return

    def create_plots(self, run_info, folder_name):
        N = self._N

        fig = plt.figure()
        en = self._energy[:N]
        en[en > en[0]] = en[0]
        plt.plot(en)
        plt.xlabel('iteration')
        plt.title('energy per iteration')
        plt.savefig(folder_name + run_info + 'energy_pIteration' +'.png')
        plt.close(fig)

        return

    def get_energy_val(self):
        return self._energy[self._N-1]

class CIterAllInfo(CIterEnergyInfo):
    def __init__(self, problem, maxIter):  # fun = 'all_info' or 'info'
        CIterEnergyInfo.__init__(self, problem.Pi, maxIter)
        self._J = problem.getJ()
        self._F = problem.getF()
        self._V0 =problem.getV0()
        self._min_det = np.zeros(maxIter)
        self._l2error_mat = np.zeros(maxIter)
        self._mesh = problem.domain.get_mesh()

        return

    def store_info(self, eps, niter):
        detF, F22 = calculate_detF_F22(self._F, self._J, self._V0)
        self._min_det[niter] = detF.vector().get_local().min()
        self._l2error_mat[niter] = eps
        self._energy[niter] = assemble(self._Pi)

        return detF


    def create_plots(self, run_info, folder_name):
        N = self._N

        A, B = self._min_det[:N], self._l2error_mat[:N]
        self._plot(A, B, ['detF min', 'l2 error'], 'det min and l2 error per iteration',
            folder_name + run_info + 'det_error_pIteration' +'.png')

        y1 = np.log(self._min_det[:N] +abs(A.min()) + 0.5)
        y2 = np.log(self._l2error_mat[:N] + abs(B.min()) + 0.5)
        self._plot(y1, y2, ['log_detF min', 'log_l2 error'],
         '(log) det min and l2 error per iteration',
         folder_name + run_info + 'log_det_error_pIteration' +'.png')

        fig = plt.figure()
        plt.plot(self._energy[:N])
        plt.xlabel('iteration')
        plt.title('energy per iteration')
        plt.savefig(folder_name + run_info + 'energy_pIteration' +'.png')
        plt.close(fig)

        return

    def _plot(self, y1, y2, legend, title, fig_name):
        N = self._N
        x= np.linspace(1, N, N)

        fig = plt.figure()
        plt.plot(x, y1, x, y2)
        plt.ylim([np.min([y1, y2])-1, np.max([y1, y2])+1] )
        plt.xlabel('iteration')
        plt.title(title) #
        plt.legend(legend)
        plt.savefig(fig_name)
        plt.close(fig)

def calculate_detF_F22(F, J, V0):
    detF = project(J, V0)
    F22 =  project(F[1,1], V0)

    return detF, F22



def calculate_detF_F22_F11(F, J, V0):
    detF = project(J, V0)
    F22 =  project(F[1,1], V0)
    F11 =  project(F[0,0], V0)

    return detF, F22, F11

def calculate_detF(J, V0):
    detF = project(J, V0)

    return detF


def calculate_dens(u, J, el_order, V0, mesh, V2):
        meshy = Mesh(mesh)

        # fix function space
        W = FunctionSpace(meshy, 'DG', 0)
        J1 = project(J, V0)
        V_dens = FunctionSpace(mesh, 'DG', el_order-1)
        dens_undef = project(1.0 / J1, V_dens)
        dens = project(1.0 / J1, W)
        V = VectorFunctionSpace(meshy, 'CG', 1)

        u2 = Function(V2)
        u2.vector()[:] = u.vector().get_local()
        u0 =  project(u2, V)
        try:
            ALE.move(meshy,u0)
        except: #dolfin_version() <= '1.6.0'
            meshy.move(u0)

        return dens, dens_undef, V_dens


def calculate_u_norms(u, J, el_order, V0, mesh, V2):
    meshy = Mesh(mesh)


    # fix function space
    V_norm_u = FunctionSpace(mesh, 'CG', el_order)
    u_norm = project(sqrt(inner(u, u)), V_norm_u)
    u_norm_def = project(sqrt(inner(u, u)), FunctionSpace(meshy, 'CG', 1))

    V = VectorFunctionSpace(meshy, 'CG', 1)
    # V2 = VectorFunctionSpace(meshy, 'CG', el_order)

    u2 = Function(V2)
    u2.vector()[:] = u.vector().get_local()
    u0 = project(u2, V)

    try:
        ALE.move(meshy,u0)
    except: #dolfin_version() <= '1.6.0'
            meshy.move(u0)
    
    
    return u_norm, u_norm_def, meshy, V_norm_u


def get_eig2D(hes):
    mesh = hes.function_space().mesh()
    S00_ = hes.sub(0)
    S01_ = hes.sub(1)
    S11_ = hes.sub(3)

    if dolfin_version() >= '2016.2.0':
        eig = project(Expression(
            ("sqrt(0.5*(S00+S11-sqrt((S00-S11)*(S00-S11)+4.*S01*S01)))", \
             "sqrt(0.5*(S00+S11+sqrt((S00-S11)*(S00-S11)+4.*S01*S01)))"), \
            S00=S00_, S01=S01_, S11=S11_, degree=2), VectorFunctionSpace(mesh, 'DG', 0))
    else:
        eig = project(Expression(
            ("sqrt(0.5*(S00+S11-sqrt((S00-S11)*(S00-S11)+4.*S01*S01)))", \
                              "sqrt(0.5*(S00+S11+sqrt((S00-S11)*(S00-S11)+4.*S01*S01)))"), \
                             S00=S00_, S01=S01_, S11=S11_), VectorFunctionSpace(mesh, 'DG', 0))
    eig2, eig1 = split(eig)
    eig1 = project(eig1, FunctionSpace(mesh, 'DG', 0));
    eig2 = project(eig2, FunctionSpace(mesh, 'DG', 0));
    eig1.rename('lambda1', 'lambda1'); eig2.rename('lambda2', 'lambda2')

    return eig1, eig2

# def calculate_detF_F22(mesh, u):
#     V0 = FunctionSpace(mesh, "DG", 0)
#     detF = Function(V0)
#     F22 =  Function(V0)
#
#     for cell_no in range(mesh.num_cells() ):
#         cell = Cell(mesh, cell_no)
#         detF.vector()[cell_no], F22.vector()[cell_no]= calculate_cell_detF_F22(cell, u)
#
#     return detF, F22


# def calculate_cell_detF_F22(cell, u):
#     x = cell.get_vertex_coordinates()
#     x0 = np.array([ x[0], x[1] ])
#     x1 = np.array([ x[2], x[3] ])
#     x2 = np.array([ x[4], x[5] ])
#
#     a1 = x1 - x0
#     a2 = x2 - x0
#     A = np.array([a1, a2])
#
#     b1 = u(x1) - u(x0) +a1
#     b2 = u(x2) - u(x0) +a2
#     B = np.array([b1, b2])
#
#     F0 = np.dot(np.linalg.inv(A), B ).transpose()
#
#     return  np.linalg.det(F0), F0[1,1] #np.trace(F0)
#

