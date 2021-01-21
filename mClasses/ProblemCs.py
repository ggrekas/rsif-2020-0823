import ufl
from dolfin import *
import numpy as np
# TODO: split energy minimization problem code to solvers
# TODO: fix adaptive Newton deadlock

class Problem2D:
    def __init__(self, psi_name=None, domain=None, el_order=None,
                 a =100, b=0.01):
        #		self._check_FunctionSpace(V)
        # Define functions
        self._el_order = el_order
        self.V = VectorFunctionSpace(domain.get_mesh(), "Lagrange", el_order)
        domain.create_bc(self.V)
        
        self._V0 = FunctionSpace(domain.get_mesh(), 'DG', 0)
        self._du = TrialFunction(self.V)  # Incremental displacement
        self._v = TestFunction(self.V)  # Test function
        
        u = Function(self.V)  # Displacement from previous iteration
        self._B = Constant((0.0, 0.0))  # Body force per unit volume
        self._T = Constant((0.0, 0.0))  # Traction force on the boundary

        # Kinematics
        d = u.geometric_dimension()

        I = Identity(d)  # Identity tensor
        F = variable(I + grad(u))  # Deformation gradient
        C = variable(F.T * F)  # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        Ic = tr(C)
        J = variable(det(F))
        J2 = (J ** 2 + 0.0001) ** 0.5

        self._exp_a, self._exp_b = None, None
        self._en_alpha = None
        
        self._l0, self._l1 = None, None
        self._relax_parameters = None

        psi_dict2 = self._dictionary_of_psi2()

        self._psi_name = psi_name
        try:  # This should contain an elastic energy defined with name psi_name
            self._psi = psi_dict2[psi_name](Ic, J)
        except: # This should be a user defined elastic energy with respect Ic and J
            self._psi = self.user_w(Ic, J)



        self._u = u
        self._F = F
        self._C = C
        self._J = J
        self._bc = domain.get_bc()  # change name from bc to bcs
        self.domain = domain

        self.Pi = self._psi * dx#(metadata={'quadrature_degree': 2})
        self._tractionPi = 0.0
        self._set_traction_forces()

        self._Jacobian = None
        self._L = None
        self._S = diff(self._psi,F)
        self._eval1 =None

        

        return

    def _exp_coef_init(self, psi_name):
        if psi_name is 'Rosakis_Grekas3' or psi_name is 'Rosakis_Grekas5'\
                or psi_name is 'Rosakis_Grekas6':
            self._exp_a, self._exp_b = 60, 0.21
            return
        elif psi_name is 'Rosakis_Grekas1':
            self._exp_a, self._exp_b = 40, 0.25
            return
        elif psi_name is 'Rosakis_Grekas2':
            self._exp_a, self._exp_b = 80, 0.2
            return
        elif psi_name is 'Rosakis_Grekas4':
            self._exp_a, self._exp_b = 40, 0.33
            return

        self._exp_a, self._exp_b = 100, 0.01
        return

    def el_order(self):
        return self._el_order

    def pam_l0(self):
        return self._l0, self._l1

    def parameters_info(self):
        s= ''
        if self._exp_a is not None:
            s+= 'exp(a,b)=(%g, %g)' % (self._exp_a, self._exp_b) + ',\n'
        if self._en_alpha is not None:
            s += 'en_a=%g' % (self._en_alpha)

        if self._l0 is not None:
            s += '<br> (l0, l1) =(%g, %g)' % (self._l0, self._l1)
        if self._relax_parameters is not None:
            s += '<br> (b, c) =(%g, %g)' % (self._relax_parameters[0],self._relax_parameters[1])

        return s

    def Rosakis_1st(self, Ic, J):
        lmbda = 0.0002
        return (1.0 /2.0)*(Ic - 2) + 2.0/J + J + lmbda*(J-1)**2 - 3.0

    def Rosakis_exp(self, Ic, J):
        self._exp_a  =100
        self._exp_b  = 0.01
        return 1.0 / 96 * (5*Ic**3 - 9*Ic**2 -12  * Ic *J**2+ 12 * J**2 +8)\
               + exp(self._exp_a * (self._exp_b - J))



    def Rosakis_Grekas3(self, Ic, J):
        self._exp_a, self._exp_b = 60, 0.21
        self._en_alpha = 0.5
        a = self._en_alpha
        return 1.0 / 48.0 * ( 5 * Ic ** 3 - 9 * Ic ** 2 - 12 * Ic * J ** 2\
                + 12 * J ** 2 + 8) + exp(self._exp_a * (self._exp_b - J)) - \
                a / 16.0 * (3 * Ic ** 2 - 4 * J ** 2) +\
                a/ 2.0 * Ic -a / 2.0


    def Grekas0 (self, Ic, J):
        self._en_alpha = 0.476
        self._exp_a, self._exp_b = 100, 0.2

        a = self._en_alpha
        return  1.0 / 48.0 * ( 5 * Ic ** 3 - 9 * Ic ** 2 - 12 * Ic * J ** 2\
                + 12 * J ** 2 + 8) +exp(self._exp_a * (self._exp_b - J)) -  a / 16.0 * (\
                3 * Ic ** 2 - 4 * J ** 2) + a / 2.0 * Ic - a / 2.0 + \
                exp(10 * (0.5 - Ic))

    def Grekas04 (self, Ic, J):
        self._en_alpha = 0.532
        self._exp_a, self._exp_b = 30, 0.25
        # self._en_alpha = 2.45
        # self._exp_a, self._exp_b = 50, 0.18
        self._en_alpha = 3.78
        self._exp_a, self._exp_b = 30, 0.1
        k_t = 8

        self._en_alpha = 0.58
        a = self._en_alpha
        self._exp_a, self._exp_b = 30, 0.1
        k_t = 1
        return 1.0 / 48.0 * (k_t * (5 * Ic ** 3 - 12 * Ic * J ** 2  \
                                   - 9 * Ic ** 2 + 12 * J ** 2 + 8) )\
                +exp(self._exp_a * (self._exp_b - J)) -  a / 16.0 * (\
                3 * Ic ** 2 - 4 * J ** 2) + a  / 2.0 * Ic\
                +a / 2 -  a



        # return  1.0 / 48.0 * ( k_t*(5 * Ic ** 3 - 12 * Ic * J ** 2 -16) \
        #                        - 9 * Ic ** 2 + 12 * J ** 2 + 24)\
        #         +exp(self._exp_a * (self._exp_b - J)) -  a / 16.0 * (\
        #         3 * Ic ** 2 - 4 * J ** 2) + (a+1-k_t) / 2.0 * Ic\
        #         +a / 2 -  (a+1-k_t)\
        #         + exp(10 * (0.5 - Ic))

    def R_G04 (self, Ic, J):

        self._en_alpha = 0.51
        a = self._en_alpha
        self._exp_a, self._exp_b = 10, 0.6
        k_t = 1
        l1, l2 = sqrt(0.5*(Ic -sqrt(Ic**2 - 4*J**2)) ), sqrt(0.5*(Ic +sqrt(Ic**2 - 4*J**2)) )
        return 1.0 / 48.0 * (k_t * (5 * Ic ** 3 - 12 * Ic * J ** 2  \
                                   - 9 * Ic ** 2 + 12 * J ** 2 + 8) )\
                 -  a / 16.0 * (\
                3 * Ic ** 2 - 4 * J ** 2) + a  / 2.0 * Ic\
                +a / 2 -  a +\
               exp(50* (0.01 - J)) + 20*exp(self._exp_a * (self._exp_b - (l1+l2) ))

    def R_G (self, Ic, J):

        self._en_alpha = 0.65
        a = self._en_alpha
        self._exp_a, self._exp_b = 80, 0.23

        lMin = sqrt(0.5*(Ic -sqrt(Ic**2 - 4*J**2)) )**0.4
        lMax = sqrt(0.5*(Ic +sqrt(Ic**2 - 4*J**2)) )**0.4
        g5 =  5*lMin**6 + 3*lMin**4 *lMax**2 +3 *lMin**2*lMax**4 + 5*lMax**6
        g3 =  9*lMin**4 + 6 *lMin**2 * lMax**2 + 9*lMax**4
        g1 = 24* lMin**2 + 24 * lMax**2
        return 1.0 / 48.0 * (8 - 24 *a + g5 - (a + 1)* g3 + a* g1)\
               + exp(self._exp_a* (self._exp_b - J))

    def my_local_min_old(self, Ic, J):
        self._l0, self._l1 = 1.05, 0.3
        l0, l1 = self._l0, self._l1
        self._exp_a, self._exp_b = 100, 0.2
        At = Ic + l0**2 + l1**2
        Bt = sqrt(Ic + 2*J)

        lMin = sqrt(0.5*(Ic -sqrt(Ic**2 - 4*J**2)) )**0.4
        lMax = sqrt(0.5*(Ic +sqrt(Ic**2 - 4*J**2)) )**0.4
        g0 =  (lMax - 1)**2 + (lMin-1)**2
        g1 = (lMax - l0)**2 + (lMin-l1)**2

        return g0 *g1  + exp(self._exp_a* (self._exp_b - J))

    def my_local_min(self, Ic, J):
        self._l0, self._l1 = 0.3, 1.3
        l0, l1 = self._l0, self._l1
        self._exp_a, self._exp_b = 100, 0.01
        At = Ic + l0**2 + l1**2
        Bt = sqrt(Ic + 2*J)


        g0 =  Ic - 2 *Bt + 2
        g1 = At**2 -2*At*(l0 + l1)* Bt
        g2 = 4*(l0*l1*Ic + (l0**2 + l1**2)*J)
        return g0 *(g1+g2) + exp(self._exp_a* (self._exp_b - J))

    def Grekas(self, Ic, J):
        self._en_alpha = 0.85
        a = self._en_alpha
        self._exp_a, self._exp_b = 60, 0.4

        return  np.pi*( 1.0/ 48.0 * ( 5 * Ic ** 3 - 9 * Ic ** 2 - 12 * Ic * J ** 2\
                                    + 12 * J ** 2 + 8)
                       -a / 16.0 * (3 * Ic ** 2 - 4 * J ** 2) + a / 2.0 * Ic
                                    - a / 2.0 )\
                + 0.4*( J - 1)**4   + exp(self._exp_a* (self._exp_b     - J))

    def Grekas2(self, Ic, J):
        self._en_alpha = 0.5274
        self._exp_a, self._exp_b = 40, 0.28

        a = self._en_alpha
        return  np.pi* (1.0/ 48.0 * ( 5 * Ic ** 3 - 9 * Ic ** 2 - 12 * Ic * J ** 2\
                                        + 12 * J ** 2 + 8)
                        - a / 16.0 * (3 * Ic ** 2 - 4 * J ** 2)
                        + a / 2.0 * Ic - a / 2.0) \
                + exp(self._exp_a * (self._exp_b - J))

    def Grekas3 (self, Ic, J):
        self._en_alpha =  0.4696
        self._exp_a, self._exp_b = 80, 0.11

        a = self._en_alpha
        return  0.5 *( 1/ 48.0 * ( 5 * Ic ** 3 - 9 * Ic ** 2 - 12 * Ic * J ** 2\
                                        + 12 * J ** 2 + 8)
                         -  a / 16.0 * (3 * Ic ** 2 - 4 * J ** 2) +
                                        a / 2.0 * Ic - a / 2.0) \
                +exp(self._exp_a * (self._exp_b - J))

    def GrekasThesis(self, Ic, J):
        self._en_alpha = 0.5
        self._exp_a, self._exp_b = 80, 0.22

        a = self._en_alpha
        # multiply by 0.5 because we multiply by 1/(2Pi)
        return   0.5*(1/ 48.0 * ( 5 * Ic ** 3 - 9 * Ic ** 2 - 12 * Ic * J ** 2\
                         + 12 * J ** 2 + 8)\
                         -  a / 16.0 * (3 * Ic ** 2 - 4 * J ** 2) +
                                        a / 2.0 * Ic - a / 2.0) \
                +exp( self._exp_a * (self._exp_b - J) )



    

    def RG_no_hysteresis(self, Ic, J):
        self._exp_a, self._exp_b = 80, 0.22

        return 1.0 / 96 * (5*Ic**3 - 9*Ic**2 -12  * Ic *J**2+ 12 * J**2 +8)\
               + exp(self._exp_a * (self._exp_b - J))
    
    def memory_relax_tension(self, Ic, J):
        self._relax_parameters = [1, 0.1, 0]
        self._en_alpha, beta, c = 0.365, self._relax_parameters[0]**2, self._relax_parameters[1]
        a, b = self._en_alpha, beta

        self._exp_a, self._exp_b = 60, 0.28
        # l9 = (6.3*Ic**5.0 - 28.0*Ic**3.0 *J**2 + 24.0*Ic*J**4 -25.6)/256.0
        # l7 = (3.5*Ic**4.0 - 12.0* (Ic**2.0 - 2*J**2.0)* J**2 - 19.2*J**4.0 - 12.8)/102.4
        l9 = (63*Ic**5.0 - 280*Ic**3.0 *J**2 + 240*Ic*J**4 -256)/2560.0
        l7 = (35*Ic**4.0 - 120* (Ic**2.0 - 2*J**2.0)* J**2 - 192*J**4.0 - 128)/1024.0
        l5 = (5*Ic**3.0 -12*Ic*J**2.0 -16)/96.0
        l3 = (3*Ic**2 - 4*J**2 -8)/32.0
        l = (Ic  - 2.0)/4.0 


        return   l9 -(1 + a + 2*b)*l7 +(a + 2*b + 2*a*b + b**2 +c)*l5 \
                -(2*a*b + b**2 +a*b**2+ c*(a+1))*l3 +(a*b**2+a*c) *l \
                +exp( self._exp_a * (self._exp_b - J) )

    
    def l1_minus1(self, Ic, J):
        self._exp_a, self._exp_b = 80, 0.22
        J2 =  (J ** 2 + 0.0001) ** 0.5
        return Ic/4.0 + 1.0/2.0 - \
            1.0/2.0*(sqrt(Ic + 2.0*J2) + 3.0* (Ic - 2.0*J2)/(10.0*sqrt(Ic + 2*J2) + sqrt(Ic + 14.0*J2)))  +\
             exp(self._exp_a * (self._exp_b - J))
    
    def l3_minus1(self, Ic, J):
        self._exp_a, self._exp_b = 80, 0.22
        J2 =  (J ** 2 + 0.0001) ** 0.5
        return 1.0/64.0*(3.0*Ic**2 - 4.0*J**2) - \
            1.0/4.0*(sqrt(Ic + 2.0*J2) + 3.0* (Ic - 2.0*J2)/(10.0*sqrt(Ic + 2*J2) + sqrt(Ic + 14.0*J2)))  +\
             exp(self._exp_a * (self._exp_b - J))

    
    def l5_minus1(self, Ic, J):
        self._exp_a, self._exp_b = 80, 0.22
        J2 =  (J ** 2 + 0.0001) ** 0.5
        return (5*Ic**3.0 -12*Ic*J**2.0 -16)/96.0 +1 - \
            1.0/2.0*(sqrt(Ic + 2.0*J2) + 3.0* (Ic - 2.0*J2)/(10.0*sqrt(Ic + 2*J2) + sqrt(Ic + 14.0*J2)))  +\
             exp(self._exp_a * (self._exp_b - J))
    
    
    def l7_minus1(self, Ic, J):
        self._exp_a, self._exp_b = 80, 0.22
        J2 =  (J ** 2 + 0.0001) ** 0.5
        return (35*Ic**4.0 - 120* (Ic**2.0 - 2*J**2.0)* J**2 - 192*J**4.0 - 128)/1024.0 +1 - \
            1.0/2.0*(sqrt(Ic + 2.0*J2) + 3.0* (Ic - 2.0*J2)/(10.0*sqrt(Ic + 2*J2) + sqrt(Ic + 14.0*J2)))  +\
             exp(self._exp_a * (self._exp_b - J))
     


    def l9_minus1(self, Ic, J):
        self._exp_a, self._exp_b = 80, 0.22
        J2 =  (J ** 2 + 0.0001) ** 0.5
        return (63*Ic**5.0 - 280*Ic**3.0 *J**2 + 240*Ic*J**4 -256)/2560.0 +1 - \
            1.0/2.0*(sqrt(Ic + 2.0*J2) + 3.0* (Ic - 2.0*J2)/(10.0*sqrt(Ic + 2*J2) + sqrt(Ic + 14.0*J2)))  +\
             exp(self._exp_a * (self._exp_b - J))


    def Rosakis_2019(self, Ic, J):
        self._exp_a, self._exp_b = 80, 0.05

        return  - 0.1232  +  0.1252 *J +  J*( 2 + J*(  Ic - 4 ) ) + exp(self._exp_a * (self._exp_b - J))

    def Rosakis_2019_2(self, Ic, J):
        self._exp_a, self._exp_b = -80, 0.0
        J2 = (J ** 2 + 0.0001) ** 0.5
        return  2.1252 + Ic/J2**3 - 4/J2 +  J2*( -0.1232 + 1.49182*exp(self._exp_a/J2)  ) 

    def Rosakis_2019_3(self, Ic, J):
        self._exp_a, self._exp_b = -80, 0.0
        J2 = (J ** 2 + 0.0001) ** 0.5
        return  2. + Ic/J2**3 - 4/J2

    def user_w(self, Ic, J):
        self._exp_a, self._exp_b = 80, 0.22

        try:
            return eval( self._psi_name )
        except:
            raise TypeError('The elastic energy %s is not a function of Ic and J or'
                            'it is not contained in the functions list\
                        ' %  self._psi_name)
        return

                
    def _dictionary_of_psi2(self):
        functions_dict ={"Polyconvex":self.Rosakis_1st}

        return functions_dict


    def _set_traction_forces(self):

        domain = self.domain
        msubdomains_list = domain.get_subDomainsList()
        for msubdomain in msubdomains_list:
            if msubdomain.has_traction_bc():
                self._add_traction(msubdomain.get_Tn(), msubdomain,
                                   domain.get_boundary_parts() )

        # if domain.has_bc():
        #     self._add_surface_integral(u0, domain, domain.get_boundary_parts())

        self.Pi = self.Pi -  self._tractionPi
        return

    def _add_traction(self, Tn, msubdomain, boundary_parts):

        T = self._traction_on_boundary(Tn, msubdomain)
        try:
           self._tractionPi += dot(T,self._u)* ds(msubdomain.get_boundary_partNum(),
                                     subdomain_data=boundary_parts)
        except:
            raise NotImplementedError('supported only for circular domains')

        return

    # exclude non boundary point to ensure correct computations
    def _traction_on_boundary(self, Tn, msubdomain):
        # d = msubdomain.get_center_displacement()
        # rho = msubdomain.get_radius()
        rho, x0, y0 = msubdomain.get_circle()


        if dolfin_version() >='2016.2.0':
            T = Expression((
                "scale*(x[0]-x0)/sqrt( (x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0) )",
                "scale*(x[1]-y0)/sqrt( (x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0) )"),
                scale=Tn, x0=x0, y0=y0, degree=2)
        else:
            T = Expression((
            "scale*(x[0]-x0)/sqrt( (x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0) )",
            "scale*(x[1]-y0)/sqrt( (x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0) )"),
            scale=Tn, x0=x0, y0=y0)

        return T


    def _check_FunctionSpace(self, V):
        if V.__class__.__name__ != 'VectorFunctionSpace' or V == None:
            raise TypeError('In %s there is not a VectorFunctionSpace\
            ' % self.__class__)

        return

    def get_u(self):
        return self._u

    def get_du(self):
        return self._du

    def get_Jacobian(self):
        return self._Jacobian

    def getF(self):
        return self._F

    def rightCauchyGreen(self):
        return self._C

    def getJ(self):
        return self._J

    def getFunctionSpace(self):
        return self.V

    def getV(self):
        return self.V

    def getV0(self):
        return self._V0

    def get_L(self):
        return self._L

    def get_bc(self):
        return self._bc

    #  get domain's exterior boundary conditions
    def get_ex_bc(self):
        return self._bc

    def get_psi_name(self):
        return self._psi_name

    def get_total_energy_val(self):
        return assemble(self.Pi)

    def set_k(self, k):
        m = 'Warning, k is set only in minimization problems, this problem is a %s'
        print(m%self.__class__.__name__)

    def compute_eval1(self):
        S = self._S
        F = self._F
        i, j = ufl.indices(2)

        self._eval1 = self._psi - S[i,j]*F[i,j]
        return

    def get_eval1(self):
        return self._eval1


class mNonlinearVariationalProblem(Problem2D):
    def __init__(self, psi_name, domain, el_order=None, a=100, b=0.01):
        Problem2D.__init__(self, psi_name, domain, el_order=el_order,
                           a=a, b=b)

        P = diff(self._psi, self._F)
        self._L = inner(P, grad(self._v)) * dx  # - dot(B,v)*dx - dot(T, v)*ds
        self._Jacobian = derivative(self._L, self._u, self._du)

        return

    def add_higher_gradients(self, epsilon=1, alpha = 8):
        if self._el_order == 1:
            raise RuntimeError(
                "higher gradients don't work for Galerkin elements of order 1")


        self._epsilon, self._alpha = epsilon, alpha

        mesh = self.domain.get_mesh()
        try:
            h = CellDiameter(mesh)
        except:
            h = CellSize(mesh)
        h_avg = (h('+') + h('-'))/2.0
        n = FacetNormal(mesh)

        self._update_hg_variatinal_form(h_avg, n)
        self._Jacobian = derivative(self._L, self._u, self._du)

        return

    def _update_hg_variatinal_form(self, h_avg, n):
        epsilon2 = Constant(self._epsilon**2.)
        alpha = Constant(self._alpha)
        # F = self._F
        Gv = grad(self._v)
        GGv = grad(Gv)
        av_GGv = Constant(0.5)*(GGv('+') + GGv('-'))
        u = self._u
        Gu = grad(u)
        F2 = Gu # Identity(u.geometric_dimension()) + Gu
        GGu = grad(Gu)
        av_GGu = avg(GGu) #Constant(0.5)*(GGu('+') + GGu('-')) #
        i, j, k, m = indices(4)

        self._L += epsilon2*(
            GGu[i,j,k]*GGv[i,j,k]*dx
                 - av_GGu[i,k,j]* Gv[i,j]('+')*n[k]('+')*dS
                 - av_GGu[i,k,j]* Gv[i,j]('-')*n[k]('-')*dS
                 - av_GGv[i,k,j]* F2[i,j]('+')*n[k]('+')*dS
                 - av_GGv[i,k,j]* F2[i,j]('-')*n[k]('-')*dS
                 + alpha/h_avg*(F2[i,j]('+') - F2[i,j]('-'))*
                 (Gv[i,j]('+') - Gv[i,j]('-') )*dS
                )

        return


class mEnergyMinimizationProblem(Problem2D):
    def __init__(self, psi_name, domain, el_order=None, a=100, b=0.01):
        Problem2D.__init__(self, psi_name, domain, el_order=el_order, a=a, b=b)

        self.boundPi = 0.0 # Pi on bound
        self.derivative_over_center = {}
        self._u1, self._u2, self._u0= None, None, None
        print( self.domain.get_mesh() )
        # self._W = FunctionSpace(self.domain.get_mesh(), "Lagrange", el_order)  #  scalar FunctionSpace
        

        self._L = derivative(self.Pi, self._u, self._v)
        self._Jacobian = derivative(self._L, self._u, self._du)
        self._k = None
        self._k_inv = None

        #higher gradients variables
        self._epsilon = None
        self._alpha = None

        self._temp = None
        self._cell_model = 'linear_spring'
        return

    def add_higher_gradients(self, epsilon=1, alpha = 24):
        if self._el_order == 1:
            raise RuntimeError(
                "higher gradients don't work for Galerkin elements of order 1")


        self._epsilon, self._alpha = epsilon, alpha

        mesh = self.domain.get_mesh()
        try:
            h = CellDiameter(mesh)
        except:
            h = CellSize(mesh)
        h_avg = (h('+') + h('-'))/2.0
        n = FacetNormal(mesh)

        self._update_hg_energy(h_avg, n)
        self._update_hg_energy_derivative(h_avg, n)

        return


    def _update_hg_energy(self, h_avg, n):
        epsilon2 = Constant(self._epsilon**2)
        alpha2 = Constant(self._alpha/2.0)
        F = self._F
        u = self._u
        Gu = grad(u)
        GGu = grad(Gu)
        F2 = Gu #Identity(u.geometric_dimension()) + grad(u)
        av_GGu = avg(GGu)
        i, j, k, m = indices(4)

        self.Pi += epsilon2*( Constant(0.5)*GGu[i,j,k]*GGu[i,j,k]*dx
                 - av_GGu[i,k,j]* F2[i,j]('+')* n[k]('+')*dS
                 - av_GGu[i,k,j]* F2[i,j]('-')* n[k]('-')*dS
                 + alpha2/h_avg*
                             (F2 [i,j]('+') - F2[i,j]('-'))*
                             (F2[i,j]('+') - F2[i,j]('-'))*dS
                             )

        return


    def _update_hg_energy_derivative(self, h_avg, n):
        epsilon2 = Constant(self._epsilon**2.)
        alpha = Constant(self._alpha)
        # F = self._F
        Gv = grad(self._v)
        GGv = grad(Gv)
        av_GGv = Constant(0.5)*(GGv('+') + GGv('-'))
        u = self._u
        Gu = grad(u)
        F2 = Gu # Identity(u.geometric_dimension()) + Gu
        GGu = grad(Gu)
        av_GGu = avg(GGu) #Constant(0.5)*(GGu('+') + GGu('-')) #
        i, j, k, m = indices(4)

        self._L += epsilon2*(
            GGu[i,j,k]*GGv[i,j,k]*dx
                 - av_GGu[i,k,j]* Gv[i,j]('+')*n[k]('+')*dS
                 - av_GGu[i,k,j]* Gv[i,j]('-')*n[k]('-')*dS
                 - av_GGv[i,k,j]* F2[i,j]('+')*n[k]('+')*dS
                 - av_GGv[i,k,j]* F2[i,j]('-')*n[k]('-')*dS
                 + alpha/h_avg*(F2[i,j]('+') - F2[i,j]('-'))*
                 (Gv[i,j]('+') - Gv[i,j]('-') )*dS
                )



        # self._L = derivative(self.Pi, self._u, self._v)
        self._Jacobian = derivative(self._L, self._u, self._du)

        # du =self._du
        # Gdu = grad(du)
        # F2 = Gdu  # Identity(u.geometric_dimension()) + Gu
        # GGdu = grad(Gdu)
        # av_GGdu = avg(GGdu)
        # self._Jacobian += epsilon*(
        #     GGdu[i,j,k]*GGv[i,j,k]*dx
        #          - av_GGdu[i,j,k]* Gv[i,j]('+')*n[k]('+')*dS
        #          - av_GGdu[i,j,k]* Gv[i,j]('-')*n[k]('-')*dS
        #          - av_GGv[i,j,k]* F2[i,j]('+')*n[k]('+')*dS
        #          - av_GGv[i,j,k]* F2[i,j]('-')*n[k]('-')*dS
        #          + alpha/h_avg*(F2[i,j]('+')*n[j]('+') + F2[i,j]('-')*n[j]('-') )*
        #          (Gv[i,m]('+')*n[m]('+') + Gv[i,m]('-')*n[m]('-') )*dS
        #         )

        return

    # to be removed
    # def _derivative_calc(self):
    #     w = TestFunction(self._W)
    #     self._u1, self._u2 = Function(self._W), Function(self._W)
    #
    #     Dpsi = diff(self._psi, self._F)
    #     DPi = dot(Dpsi, grad(w))
    #     return DPi[0]*dx, DPi[1]*dx


    def get_k(self):
        return self._k

    def cell_model(self):
        return str(self._cell_model)

    def set_k(self, k, cell_model = 'linear_spring'):
        # if abs(k) < 1e-10:
        #     return

        self._k = k
        self._cell_model = cell_model
        # self.Pi = k_inv*self.Pi
        self._u0 = Function(self.V)
        for bc in self._bc:
            bc.apply(self._u0.vector() )

        self._add_surface_integrals()

        # self._update_derivatives()

        self._L = derivative(self.Pi, self._u, self._v)
        self._Jacobian = derivative(self._L, self._u, self._du)

        return


    def _add_surface_integrals(self):
        u0 = self._u0

        try:
            x_e = Expression(("x[0]", "x[1]"), degree=2)
        except:
            x_e = Expression(("x[0]", "x[1]"))

        x_val = project(x_e, self.V)

        domain = self.domain
        msubdomains_list = domain.get_subDomainsList()
        for msubdomain in msubdomains_list:
            if msubdomain.has_bc():
                self._add_surface_integral(u0, msubdomain, domain.get_boundary_parts(),
                                           x_val)
                b0, b1 = self._surface_integral_derivative(u0, msubdomain,
                                        domain.get_boundary_parts(), x_val)
                der_x, der_y = Constant(self._k * 0.5) * b0, Constant(self._k * 0.5) * b1
                self.derivative_over_center[msubdomain] = (der_x, der_y)


        # if domain.has_bc():
        #     self._add_surface_integral(u0, domain, domain.get_boundary_parts())

        self.Pi +=  Constant(self._k*0.5) *self.boundPi
        return

    def _add_surface_integral(self, u0, msubdomain, boundary_parts, x):
        d = msubdomain.get_center_displacement()
        center_i = msubdomain.cirlce_center()
        rho = msubdomain.get_radius()
        u0_val = -msubdomain.u0_val()
        try:
            if self._cell_model == 'linear_spring':
                self.boundPi +=  ( sqrt ( inner(self._u + x- (center_i+d), self._u +x
                                                - (center_i+d)) )
                                   -(rho-u0_val) )**2\
                                 *ds( msubdomain.get_boundary_partNum(),
                        subdomain_data = boundary_parts)
            else:
                self.boundPi += inner(self._u - u0 - d, self._u - u0 - d) * ds(
                    msubdomain.get_boundary_partNum(),
                    subdomain_data=boundary_parts)
        except:
            raise NotImplementedError('supported only for circular domains')

        return


    def _surface_integral_derivative(self, u0, msubdomain, boundary_parts, x):
        d = msubdomain.get_center_displacement()
        center_i = msubdomain.cirlce_center()
        rho = msubdomain.get_radius()
        u0_val = -msubdomain.u0_val()
        try:
            if self._cell_model == 'linear_spring':
                # raise NotImplementedError('does not suppoerted for linear_spring')
                const_part0 = sqrt( inner(self._u + x - (center_i + d), self._u + \
                                                       x - (center_i + d)) )
                const_part1  = Constant(2) *(const_part0 - (rho-u0_val) )
                b0 =  -const_part1/const_part0 *(self._u[0] + x[0] -\
                                                 (center_i[0] + d[0]) )* ds(
                    msubdomain.get_boundary_partNum(),
                    subdomain_data=boundary_parts)

                b1 = -const_part1/const_part0 *(self._u[1] + x[1] -\
                                                 (center_i[1] + d[1]) )* ds(
                    msubdomain.get_boundary_partNum(),
                    subdomain_data=boundary_parts)
            else:
                # d0, d1 = split(d)
                b0 =Constant(-2) * (self._u[0] - u0[0] - d[0])*  ds(
                    msubdomain.get_boundary_partNum(),
                    subdomain_data=boundary_parts)
                b1 = Constant(-2) * (self._u[1] - u0[1] - d[1]) * ds(
                    msubdomain.get_boundary_partNum(),
                    subdomain_data=boundary_parts)

            return b0, b1
        except:
            raise NotImplementedError('supported only for circular domains')

        return

    def _update_derivatives(self):
        k_inv = Constant(str(1.0 / self._k))
        self._k_inv = k_inv

        self._DPi0, self._DPi1 = self._k_inv*self._DPi0, self._k_inv*self._DPi1
        u0 = self._u0

        W = self._W
        w = TestFunction(W)
        self._u1, self._u2 = Function(W), Function(W)
        u01, u02 = Function(W), Function(W)
        u01.vector()[:] = np.array(u0.vector().get_local()[::2])
        u02.vector()[:] = np.array(u0.vector().get_local()[1::2])

        boundary_parts = self.domain.get_boundary_parts()
        msubdomains_list = self.domain.get_subDomainsList()
        for msubdomain in msubdomains_list:
            if msubdomain.has_bc():
                self._DPi0 += self._add_der_surface_integral(self._u1, u01, w,
                                        msubdomain, boundary_parts)
                self._DPi1 += self._add_der_surface_integral(self._u2, u02, w,
                                        msubdomain, boundary_parts)
        if self.domain.has_bc():
            self._DPi0 += self._add_der_surface_integral(self._u1, u01, w,
                                        self.domain, boundary_parts)
            self._DPi1 += self._add_der_surface_integral(self._u2, u02, w,
                                        self.domain, boundary_parts)

        return



    def _add_der_surface_integral(self, u, u_b, w, msubDomain, boundary_parts):

        return 2*( dot(u, w) - dot(u_b, w) )*ds(msubDomain.get_boundary_partNum(),
                                               subdomain_data=boundary_parts)

    def get_u1_u2(self):
        return self._u1, self._u2

    # def get_DPi(self):
    #     return self._DPi0, self._DPi1

    @property
    def get_W(self):
        return self._W

    def get_epsilon(self):
        return self._epsilon

    def get_alpha(self):
        return self._alpha

    def get_surface_integrals_val(self):
        return assemble(self.boundPi)

