"""
Codim-2 CutFEM on a 1D curve embedded in 3D .

The curve is cut out of a 3D background mesh by TWO level sets at once.

    Omega is the heart geometry. Gamma epi is the outer surface (boundary).  

To obtain the level sets:
    
    -lap_Omega psi  = 1       , psi = 0 on Gamma_epi
    
     delta = eps * max_omega(psi)

Run:  python codim2_scaffold.py
"""
from ngsolve import *
from netgen.occ import *
from xfem import *
from xfem.mlset import *
import numpy as np
import matplotlib.pyplot as plt


# ================================== config ===================================
  
order = 1                # P1 background space
gamma_n = 0.1           # normal-gradient stabilisation weight (see below)


# ================================ level-set ===================================
eps = 0.01    #how far inside the heart the first levelset sits

mesh = Mesh("01_strocchi_LV.vol.gz")

fes = H1(mesh, order=order, dirichlet="epicardium")
u, v = fes.TnT()
f = LinearForm(fes)
f += 1*v*dx

a = BilinearForm(fes)
a += grad(u) * grad(v) * dx

a.Assemble()
f.Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), "sparsecholesky") * f.vec   #solver for symmetric matrices

psi = gfu.vec.FV().NumPy()
delta = eps * psi.max()

phi_1 = delta - gfu

level_sets_old = (phi_1, y-90)


# ============================== cut geometry =================================

# The two level sets are INTERPOLATED to P1.  The discrete curve Gamma_h is the
# intersection of the two P1 zero-sets -- that is what dCut integrates over.
lsets_p1_old = tuple(GridFunction(H1(mesh, order=1)) for _ in level_sets_old)
for lset_p1_old, lset_old in zip(lsets_p1_old, level_sets_old):
    InterpolateToP1(lset_old, lset_p1_old)

# (IF, IF) = "on the interface of phi_1 AND on the interface of phi_2" = the curve.
line_old = DomainTypeArray((IF, IF))
mlci_old = MultiLevelsetCutInfo(mesh, lsets_p1_old)
els_old = mlci_old.GetElementsWithContribution(line_old)      # elements Gamma passes through



##--To find a point on Gamma
VTKOutput(mesh, coefs=[x, y, z, lsets_p1[0], lsets_p1[1]],
         names=["x", "y", "z", "phi1", "phi2"],
         filename="06_check", subdivision=0).Do(drawelems=els)


# ============================== discrete space ===============================
# dgjumps=True is NOT set: it is only needed for a ghost penalty (u.Other()), and
# this method uses the normal-gradient stabilisation alone.

alpha = 0.1      #diffusion coef

periodic = False
if periodic:
    V = Periodic(H1(mesh, order=order))
else:
    V = H1(mesh, order=order, dgjumps = True)
 


u, v = V.TnT()
h = specialcf.mesh_size
dGamma_old = dCut(lsets_p1_old, line_old, definedonelements=els_old)      # integrate along Gamma


#Centre of the ring example
x_Gamma_center = Integrate( x *dGamma_old, mesh) / Integrate( CF(1.) *dGamma_old, mesh)
z_Gamma_center = Integrate( z *dGamma_old, mesh) / Integrate( CF(1.) *dGamma_old, mesh)

theta=atan2(z - z_Gamma_center, x - x_Gamma_center)    #polar angle with respect to this centre
A=5     #amplitude of wave
k=7      #no of oscillations
phi_2 = y-90-A*sin(k*theta)    #second level set, oscillating plane

level_sets = (phi_1,phi_2)

lsets_p1 = tuple(GridFunction(H1(mesh, order=1)) for _ in level_sets)
for lset_p1, lset in zip(lsets_p1, level_sets):
    InterpolateToP1(lset, lset_p1)
    
line = DomainTypeArray((IF, IF))
mlci = MultiLevelsetCutInfo(mesh, lsets_p1)
els = mlci.GetElementsWithContribution(line)  

dGamma = dCut(lsets_p1, line, definedonelements=els)
freedofs = GetDofsOfElements(V, els) & V.FreeDofs()   
 
 # ============================ normals / projections ==========================
def Normalized(v):
    return 1.0 / Norm(v) * v

g1, g2 = grad(lsets_p1[0]), grad(lsets_p1[1])
e1 = Normalized(g1)
e2 = Normalized(g2 - (g2 * e1) * e1)


def P(v):        # tangential projection: along the curve
    return v - (v * e1) * e1 - (v * e2) * e2


def Q(v):        # normal projection: onto the 2D normal space
    return (v * e1) * e1 + (v * e2) * e2


# ============================== bilinear form ================================
# The restricted BilinearForm essentially is a bilinear form only evaluating all
# (bi) linearform integrators over the given elements (or facets -- but not salient in this case)

#point on Gamma, start point of Gaussian pulse
xc=28.3029
yc=89.3293
zc=394.675
width = 0.1
u0 = exp( -width*((x-xc)**2 + (y-yc)**2 + (z-zc)**2)) #Gaussian pulse initial cond

#--Convection field
speed=20
w_raw =  Cross(e1,e2)
w =  speed * Normalized(w_raw)
name = "codim2_CD_heart_pulse"

m = RestrictedBilinearForm(V, element_restriction=els, check_unused=False)
m += u * v * dGamma
m.Assemble()

a = RestrictedBilinearForm(V, element_restriction=els, check_unused=False)
a += alpha * InnerProduct(P(Grad(u)), P(Grad(v))) * dGamma
a += InnerProduct(w, P(Grad(u))) * v * dGamma
a += gamma_n * h * InnerProduct(Q(Grad(u)), Q(Grad(v))) * dx(definedonelements=els)
#a += 0.05 / h**2 * (u - u.Other()) * (v - v.Other()) * dw
a.Assemble()

# -- geometric stabilisation: normal-gradient.
# NOTE `definedonelements=els`: this is a VOLUME (dx) integral, so without the
# restriction it would be assembled over the entire background cylinder.




# ================================== solve ====================================
zeta = 0.5  #1 for Implicit Euler, 0.5 for Crank-Nicolson
T = 5
gfu = GridFunction(V)
rhs = gfu.vec.CreateVector()
dt0 = 0.5
gamma_c=0.01 

#VTK = VTKOutput(mesh, coefs=[gfu, lsets_p1[0], lsets_p1[1], u0],
#          names=["gfu", "lset1", "lset2", "u_exact"],
#          filename=name, subdivision=0)

#VTK.Do(drawelems=els)   

errortype = 'L2'     #'Linf' or 'L2'
errors_by_CIP={}


#CIP stabilisation toggle on and off
CIP=False

if CIP:
    a += gamma_c * Norm(w) * ju * jv * dx(skeleton=True, definedonelements=facets)
a.Assemble()

vtk_name = "CIP" if CIP else "noCIP"

#VTK = VTKOutput(mesh,coefs=[gfu],
#    names=["gfu"],
#    filename=f"codim2_VanishingDiffusion__heart_{vtk_name}", subdivision=0)

     
for i_t in range(5,6):
    dt = 0.5**i_t*dt0
    mstar = a.mat.CreateMatrix()
    mstar.AsVector().data = m.mat.AsVector() + zeta * dt * a.mat.AsVector()
    inv = mstar.Inverse(freedofs, inverse="umfpack")

    t = 0
    gfu.Set(u0)

    nsteps = int(round(T / dt))
    for n in range(nsteps):
        rhs.data = m.mat * gfu.vec - (1 - zeta) * dt * (a.mat * gfu.vec)
        gfu.vec.data = inv * rhs 
        t += dt
        VTK.Do(drawelems=els)  
       
        
   