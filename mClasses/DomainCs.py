#from dolfin.cpp.mesh import MeshFunction
import mshr
from dolfin import *
import numpy as np

# TODO: move create bc to Problem ..... !!!!!
# TODO: rename domain to domain_geometry
# TODO: generalize boundary conditions. Create new classes that impose bc
# TODO: enable get_bnd_function for Rectangular Domain

class Domain:
    def __init__(self, u0=None, Tn = None):
        self._domain = None
        self._mesh = None
        self._subDomainsList = []
        self._u0 = u0  # boundary displacement vector
        self._Tn = Tn
        if u0 is not None and Tn is not None:
            raise ValueError('Both Dirichlet and Force boundary conditions '
                             'have been given')

        self._bcs = []  # list of boundary condition functions
        self._boundary_parts = None
        self._b_part_num = 0
        self._tol = 0.1

        self.isOnBoundary = None
        self._Gamma = None
        self._mesh_resolution = None

        return

    def set_Dirichlet_bc(self, u0):
        self._u0 = u0

    def remove_subdomain(self, m_subDomain):
        # subDomain must be contained in self._domain
        self.check_domain()
        self._domain = self._domain - m_subDomain.get_domain()
        self._subDomainsList.append(m_subDomain)

        return

# !!!!!!!! user is responsible for the appropiate mesh input. Should be in  agreement
#     with domain!!!!!!!
    def set_mesh(self, mesh, resolution=None):
        self.check_domain()

        self._mesh =mesh
        self._mesh_resolution = resolution
        return

    def create_mesh(self, resolution = 100):
        self.check_domain()
        # generator = mshr.CSGCGALMeshGenerator2D()
        # mesh = Mesh()
        # generator.generate(self._domain, mesh)
        self._mesh_resolution = resolution
        self._mesh = mshr.generate_mesh(self._domain, resolution)
        return

    def get_mesh(self):
        self._check_mesh_val()
        return self._mesh

    def mesh_resolution(self):
        return self._mesh_resolution

    def check_domain(self):
        if self._domain == None:
            raise NotImplementedError('no domain definition for %s' % self.__class__.
                                      __name__)
        return

    def _check_mesh_val(self):
        if self._mesh == None:
            raise NotImplementedError('undefined mesh for %s' % self.__class__.
                                      __name__)
        return

    def get_domain(self):
        return self._domain

    def has_bc(self):
        if self._u0 == None:
            return False
        return True

    def has_traction_bc(self):
        if self._Tn == None:
            return False
        return True


    def bc_type(self):
        if self._u0 is not None:
            if type(self._u0).__name__ is 'CompiledExpression':
                # a = np.zeros(2)
                # self._u0.eval(a, np.zeros(2))
                # s = str(a)
                s = str(self._u0.cppcode)
            else:
                s = str(self._u0)
            return 'Dirichlet=' + s
        elif self._Tn is not None:
            return 'Traction Force=' + str(self._Tn)

        return 'Free'


    
    def create_bc(self, V):
        self._bcs = [] # ----------------------- ?????
        self._check_mesh_val()
        mesh = self._mesh
        self._boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

        for m_subDomain in self._subDomainsList:
            bc = m_subDomain._impose_bc(V)
            if bc != None:
                
                if m_subDomain.__class__.__name__ == 'RectangularDomain':
                    print(m_subDomain.__class__.__name__)
                    for bci in bc:
                        self._bcs.append(bci) # make bcs a dictionary
                else:
                     self._bcs.append(bc) # make bcs a dictionary
                
            
            m_subDomain._mark_boundary(self._boundary_parts, self._b_part_num)
            self._b_part_num += 1
            
        bc = self._impose_bc(V)
        self._mark_boundary(self._boundary_parts, self._b_part_num)
        if bc != None:
            if self.__class__.__name__ == 'RectangularDomain':
                for bci in bc:
                        self._bcs.append(bci) # make bcs a dictionary
            else:
                 self._bcs.append(bc) # make bcs a dictionary
            
            # self._bcs.append(bc)

        return self._bcs

    def get_bc(self):
        return self._bcs

    def get_Tn(self):
        return self._Tn

    def get_subDomainsList(self):
        return self._subDomainsList

    def get_boundary_parts(self):
        return self._boundary_parts

    def get_boundary_partNum(self):
        return self._b_part_num

    def _impose_bc(self, V=None):
        raise NotImplementedError('_impose_bc missing in class \
         %s' % self.__class__.__name__)
        return

        # mark boundary and store subDomain num
    def _mark_boundary(self, boundary_parts, b_part_num):
        self._Gamma.mark(boundary_parts, b_part_num)
        self._b_part_num = b_part_num
        self._boundary_parts = boundary_parts
        return

    # def _mark_boundary(self, b=None):
    #     raise NotImplementedError('_mark_boundary missing in class \
    #      %s' % self.__class__.__name__)
    #     return

    def refine_mesh(self):
        mesh = self._mesh
        marker = CellFunction('bool', mesh, True )
        self._mesh = refine(mesh, marker, True)
        #processes2 = CellFunction('size_t', mesh, MPI.rank(mesh.mpi_comm()))
        return

    def band_refinement(self):
        sList = self._subDomainsList
        cell_markers = CellFunction("bool", self._mesh, True)
        self._init_cell_markers(cell_markers)

        for i in range(len(sList)-1):
            for j in range(i+1, len(sList)):
                self._apply_band_refinement(sList[i], sList[j], cell_markers)

        self._mesh = refine(self._mesh, cell_markers, True)
        return

    def circle_refinement(self, rho, c_x, c_y):
        cell_markers = CellFunction("bool", self._mesh, True)
        self._init_cell_markers(cell_markers)

        for c in cells(self._mesh):
            if self.is_cell_in_circle(c, c_x, c_y, rho):
                cell_markers[c] = True

        self._mesh = refine(self._mesh, cell_markers, True)
        return

    def is_cell_in_circle(self, c, c_x, c_y, rho):
        x, y = c.midpoint().x(), c.midpoint().y()
        d = sqrt( (x-c_x)**2 + (y-c_y)**2 )
        if d < rho:
            return True
        return False


    def _apply_band_refinement(self, myDomain1, myDomain2, cell_markers):

        if myDomain1.__class__.__name__ == 'RectangularDomain' or \
           myDomain2.__class__.__name__ == 'RectangularDomain':
            print('band refinement not supported for non CircularDomain')
            exit()

        r1, x1, y1 = myDomain1.get_circle()
        r2, x2, y2 = myDomain2.get_circle()
        v1, v2 = np.array([x1, y1]), np.array([x2, y2])
        min_r, min_x, min_y = min(r1, r2), min(x1, x2), min(y1, y2)
        max_x, max_y = max(x1, x2), max(y1, y2)
        for c in cells(self._mesh):
            x, y = c.midpoint().x(), c.midpoint().y()
            v0 = np.array([x, y])
            if  self._point2line_dist(v0, v1, v2) < 3*min_r/2.0 and \
                (between(x, (min_x, max_x)) or between(y, (min_y, max_y))):
                cell_markers[c] = True

        return


    def rect_band_refinement(self, min_x, min_y, max_x, max_y):
        cell_markers = CellFunction("bool", self._mesh, True)
        self._init_cell_markers(cell_markers)

        for c in cells(self._mesh):
            x, y = c.midpoint().x(), c.midpoint().y()
            if  between(x, (min_x, max_x)) and between(y, (min_y, max_y)):
                cell_markers[c] = True

        self._mesh = refine(self._mesh, cell_markers, True)
        return

    def _point2line_dist(self, x0, x1, x2):
        tmp1, tmp2 = x1-x0, x2-x1
        t= -tmp1.dot(tmp2)/tmp2.dot(tmp2)
        return np.linalg.norm(x1 -x0 + (x2-x1)*t)

    def _init_cell_markers(self, cell_markers):
        for c in cells(self._mesh):
            cell_markers[c] = False

    def get_bnd_function(self):
        return self._Gamma

class CircularDomain(Domain):
    def __init__(self, rho=10, x=0, y=0, u0=None, Tn=None, bound_res=50):
        Domain.__init__(self, u0=u0, Tn=Tn)
        self._rho, self._x, self._y = rho, x, y

        try:
            self._center = dolfin.Point(x, y)
            self._domain = mshr.Circle(dolfin.Point(x, y), rho, bound_res)
        except:
            self._center = Point(x, y)
            self._domain = mshr.Circle(Point(x, y), rho, bound_res) #version 2018.1.0 or later
        self._Gamma = CircleBoundary(x, y, rho, self._tol)
        self._center_displ = Constant((0.0, 0.0))

    def get_circle(self):
        rho, x, y = self._rho, self._x, self._y
        return rho, x, y

    def  cirlce_center(self):
        return Constant((self._x, self._y))

    def get_center_displacement(self):
        return self._center_displ

    def set_center_displacement(self, d):
        self._center_displ.assign(d)
        return

    def _displacement_on_bnd(self):
        if type(self._u0).__name__ is 'CompiledExpression':
            return self._u0

        rho, x0, y0 = self.get_circle()
        try:
            u0 = Expression((
                "scale*(x[0]-x0)/sqrt( (x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0) )",
                "scale*(x[1]-y0)/sqrt( (x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0) )"),
                scale=self._u0, x0=x0, y0=y0, degree=2)
        except:
            u0 = Expression((
            "scale*(x[0]-x0)/sqrt( (x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0) )",
            "scale*(x[1]-y0)/sqrt( (x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0) )"),
            scale=self._u0, x0=x0, y0=y0)

        return u0

    def _impose_bc(self, V):
        tol = self._tol

        Gamma = self._Gamma
        if self._u0 == None:
            return None

        u0  =self._displacement_on_bnd()
        # print( 'warning in impose bc, tol = ', tol)

        bc = DirichletBC(V, u0, Gamma)
        return bc

    def get_radius(self):
        return self._rho

    def u0_val(self):
            return self._u0

class EllipticalDomain(Domain):
    def __init__(self, x=0, y=0, a=1, b=1, u0=None, Tn=None, bound_res=100):
        Domain.__init__(self, u0=u0, Tn=Tn)
        self._x, self._y = x, y
        self._a, self._b = a, b



        self._center = dolfin.Point(x, y)
        self._domain = mshr.Ellipse(dolfin.Point(x, y), self._a, self._b, bound_res)
        self._Gamma = EllipsisBoundary(x, y, a, b, self._tol)
        self._center_displ = Constant((0.0, 0.0))

    def get_ellipsis(self):
        x, y = self._x, self._y

        return (x, y), self._a, self._b

    def _impose_bc(self, V):
        tol = self._tol


        if self._u0 == None:
            return None
        raise NotImplementedError('_impose_bc missing in class \
                 %s' % self.__class__.__name__)
        Gamma = self._Gamma

        return

class RectangularDomain(Domain):
    def __init__(self, x1=0, y1=0, x2=1, y2=1, sides_bc =None, 
                 isMeshUniform=False ):
       
        Domain.__init__(self, u0=sides_bc)
        self._x1, self._y1, self._x2, self._y2 = x1, y1, x2, y2
        try:
            self._domain = mshr.Rectangle(dolfin.Point(x1, y1), dolfin.Point(x2, y2))
        except:
            self._domain = mshr.Rectangle(Point(x1, y1), Point(x2, y2)) # version 2018.1.0 or latter
        self._tol = 1e-7
        self._Gamma = RectangularBoundary(x1, y1, x2, y2, self._tol)
        
        self._isMeshUniform = isMeshUniform

        self._sides_bc = sides_bc

        return

    def create_mesh(self, resolution = 100):
        self.check_domain()
        self._mesh_resolution = resolution

        if self._isMeshUniform:
            self._mesh = UnitSquareMesh(resolution, resolution, 'crossed')
            return

        self._mesh = mshr.generate_mesh(self._domain, resolution)
        return

    def has_uniform_mesh(self):
        return self._isMeshUniform

    def get_corners(self):
        return self._x1, self._y1, self._x2, self._y2

    def create_bc2(self, V, lambda1=1, lambda2=1):
        def u0_boundary(x, on_boundary):
            return on_boundary

        if dolfin_version() >= '2016.2.0':
            u0 = Expression(('lambda1*x[0]', 'lambda2*x[1]'), lambda1=lambda1,
                            lambda2=lambda2, degree=2)
        else:
            u0 = Expression(("lambda1*x[0]", "lambda2*x[1]"),
                            lambda1=lambda1, lambda2=lambda2)


        bc = DirichletBC(V, u0, u0_boundary)
        self.isOnBoundary = u0_boundary()
        self._bcs.append(bc)

    def _impose_bc(self, V):
        if self._sides_bc == None:
            return None
        bcs = []
        for u0 in self._sides_bc:
            bc = DirichletBC(V, u0, self._sides_bc[u0])
            bcs.append(bc)

        return bcs

    def get_bnd_function(self):
        if self._sides_bc == None:
            return None
        bound_fun_list = []
        for u0 in self._sides_bc:
            bound_fun_list.append(self._sides_bc[u0])

    
        return bound_fun_list
        


    def _impose_bc_old(self, V):
        lambda1, lambda2 = self._lambda1, self._lambda2

        Gamma = self._Gamma
        if lambda1==None and lambda2 == None:
            return None
        if lambda1==None or lambda2 == None:
            raise NotImplementedError('boundary conditions for one of the boundaries free'
                                      'and the others fixed is missing '
                                      '(in class \
         %s)' % self.__class__.__name__)
            return

        if dolfin_version() >= '2016.2.0':
            u0= Expression(("(lambda1-1)*x[0]", "(lambda2-1)*x[1]"),
                           lambda1=lambda1, lambda2=lambda2, degree=2)
        else:
            u0 = Expression(("(lambda1-1)*x[0]", "(lambda2-1)*x[1]"),
                            lambda1=lambda1, lambda2=lambda2)
        bc = DirichletBC(V, u0, Gamma)

        return bc 



    def create_bc3(self, V, lambda1=1, lambda2=1):
        # useless function???
        # Mark boundary subdomians

        # self.isOnBoundary = True
        return

class CircleBoundary(SubDomain):
    def __init__(self, x0, y0, rho, tol):
        SubDomain.__init__(self)
        self.mx = x0
        self.my = y0
        self.mrho = rho
        self.mtol = tol

    def inside(self, x, on_boundary):
        x0, y0, rho, tol = self.mx, self.my, self.mrho, self.mtol
        r = sqrt((x[0] - x0) * (x[0] - x0) + (x[1] - y0) * (x[1] - y0))
        return on_boundary and between(r, (rho - tol, rho + tol))

class EllipsisBoundary(SubDomain):
    def __init__(self, x0, y0, a, b, tol):
        SubDomain.__init__(self)
        self._a,self._b = a, b
        self._x0 , self._y0 = x0, y0
        self.mtol = tol

    def inside(self, x, on_boundary):
        a, b,  tol = self._a, self._b, self.mtol
        x0, y0 = self._x0, self._y0

        r = ( (x[0]-x0)/a)**2 + ( (x[1]-y0)/b)**2
        return on_boundary and between(r, (1 - tol, 1 + tol))



class RectangularBoundary(SubDomain):
    def __init__(self, x1, y1, x2, y2, tol=1e-5):
        SubDomain.__init__(self)
        self._mx1 = x1
        self._my1 = y1
        self._mx2 = x2
        self._my2 = y2
        self.mtol = tol

    def inside(self, x, on_boundary):
        tol = self.mtol
        onb_rect =  between(x[0], (self._mx1, self._mx1 + tol)) \
                or  between(x[0], (self._mx2, self._mx2 + tol)) \
                or  between(x[1], (self._my1, self._my1 + tol)) \
                or  between(x[1], (self._my2, self._my2 + tol))
        return on_boundary  and onb_rect

