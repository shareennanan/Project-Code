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
gamma_n = 0.01           # normal-gradient stabilisation weight (see below)


# ================================ level-set ===================================
# Each geometry block defines:
#   level_sets      = (phi_1, phi_2)
#   periodic        bool         True if Gamma closes into a loop

#   u_exact, rhs                 manufactured solution and its right-hand side
#   w                            background flow (unused by Poisson; for the TODO)


eps = 0.01

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
gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), "sparsecholesky") * f.vec

psi = gfu.vec.FV().NumPy()
delta = eps * psi.max()

phi_1 = delta - gfu

level_sets = (phi_1, y-90)


# ============================== cut geometry =================================

# The two level sets are INTERPOLATED to P1.  The discrete curve Gamma_h is the
# intersection of the two P1 zero-sets -- that is what dCut integrates over.
lsets_p1 = tuple(GridFunction(H1(mesh, order=1)) for _ in level_sets)
for lset_p1, lset in zip(lsets_p1, level_sets):
    InterpolateToP1(lset, lset_p1)

# (IF, IF) = "on the interface of phi_1 AND on the interface of phi_2" = the curve.
line = DomainTypeArray((IF, IF))
mlci = MultiLevelsetCutInfo(mesh, lsets_p1)
els = mlci.GetElementsWithContribution(line)      # elements Gamma passes through



##To find a point on Gamma
#VTKOutput(mesh, coefs=[x, y, z, lsets_p1[0], lsets_p1[1]],
#        names=["x", "y", "z", "phi1", "phi2"],
#          filename="06_check", subdivision=0).Do(drawelems=els)


# ============================ normals / projections ==========================
def Normalized(v):
    return 1.0 / Norm(v) * v


# The normal space of a codim-2 curve is 2-DIMENSIONAL, spanned by the two
# level-set gradients.  Two things matter here:
#
#  * use grad of the P1-INTERPOLATED level sets, not the analytic gradients:
#    these are the normals of the discrete curve Gamma_h we actually integrate
#    over.  Analytic normals are inconsistent with Gamma_h 
#
#  * GRAM-SCHMIDT them.  grad(phi_1) and grad(phi_2) are linearly independent on
#    Gamma but in general NOT orthogonal.  They happen to be orthogonal for the
#    line (phi = (x, y)) -- there the second line below is a no-op -- but they are
#    NOT for the corkscrew, and P/Q are only projections if (e1, e2) is orthonormal.
g1, g2 = grad(lsets_p1[0]), grad(lsets_p1[1])
e1 = Normalized(g1)
e2 = Normalized(g2 - (g2 * e1) * e1)


def P(v):        # tangential projection: along the curve
    return v - (v * e1) * e1 - (v * e2) * e2


def Q(v):        # normal projection: onto the 2D normal space
    return (v * e1) * e1 + (v * e2) * e2


# ============================== discrete space ===============================
# dgjumps=True is NOT set: it is only needed for a ghost penalty (u.Other()), and
# this method uses the normal-gradient stabilisation alone.

alpha = 0.1

periodic = False
if periodic:
    V = Periodic(H1(mesh, order=order))
else:
    V = H1(mesh, order=order, dgjumps = True)

# THE ACTIVE-DOF RESTRICTION.  Only dofs belonging to elements that Gamma passes
# through are "live"; every other dof is untouched by all integrators, so the
# matrix is singular there.  Restricting the solve to `freedofs` is what makes the
# method well posed. 
freedofs = GetDofsOfElements(V, els) & V.FreeDofs()
u, v = V.TnT()
h = specialcf.mesh_size
dGamma = dCut(lsets_p1, line, definedonelements=els)      # integrate along Gamma

#Point on Gamma
xc=29.3225
yc=89.9492
zc=398.566
#Initial cond: Gaussian pulse centred on above point
width = 0.01
u0 = exp( -width*((x-xc)**2 + (y-yc)**2 + (z-zc)**2))

#Convection Field: 'False' for non divergence-free, 'True' for divergence-free
w_divfree= False

if w_divfree:
    w=10*Normalized(Cross(e1,e2))
else:
    x_Gamma_center = Integrate( x *dGamma, mesh) / Integrate( CF(1.) *dGamma, mesh)
    z_Gamma_center = Integrate( z *dGamma, mesh) / Integrate( CF(1.) *dGamma, mesh)
    w_raw =  CF((z - z_Gamma_center, 0, -x + x_Gamma_center))
    w =  10 * Normalized(w_raw)
    name = "codim2_CD_heart_pulse"
    
 
# ============================== bilinear form ================================
# The restricted BilinearForm essentially is a bilinear form only evaluating all
# (bi) linearform integrators over the given elements (or facets -- but not salient in this case)


m = RestrictedBilinearForm(V, element_restriction=els, check_unused=False)
m += u * v * dGamma
m.Assemble()

a = RestrictedBilinearForm(V, element_restriction=els, check_unused=False)
a += alpha * InnerProduct(P(Grad(u)), P(Grad(v))) * dGamma
a += InnerProduct(w, P(Grad(u))) * v * dGamma
a += gamma_n * h * InnerProduct(Q(Grad(u)), Q(Grad(v))) * dx(definedonelements=els)

a.Assemble()

# -- geometric stabilisation: normal-gradient.
# NOTE `definedonelements=els`: this is a VOLUME (dx) integral, so without the
# restriction it would be assembled over the entire background cylinder.



# ================================== solve ====================================
zeta = 0.5  #1 for Implicit Euler, 0.5 for Crank-Nicolson



gfu = GridFunction(V)
rhs = gfu.vec.CreateVector()
dt0 = 0.5



VTK = VTKOutput(mesh, coefs=[gfu, lsets_p1[0], lsets_p1[1]],
          names=["gfu", "lset1", "lset2"],
           filename='heartactive', subdivision=0)

  
i_t=5
dt = 0.5**i_t*dt0
mstar = a.mat.CreateMatrix()
mstar.AsVector().data = m.mat.AsVector() + zeta * dt * a.mat.AsVector()
inv = mstar.Inverse(freedofs, inverse="umfpack")

mass=[]
Ts=np.arange(0,11.5,1)
for T in Ts:

    t = 0
    gfu.Set(u0)
    mass_0 = Integrate(gfu * dGamma, mesh=mesh)
   
    nsteps = int(round(T / dt))
    for n in range(nsteps):
        rhs.data = m.mat * gfu.vec - (1 - zeta) * dt * (a.mat * gfu.vec)
        gfu.vec.data = inv * rhs 
        t += dt
        #VTK.Do(drawelems=els)
#mass calculation for conservative formulation investigation           
    mass_T= Integrate(gfu * dGamma, mesh=mesh)
    mass_loss = abs(mass_0-mass_T)*100/mass_0
    mass.append(mass_loss)
    
    
mass=np.array(mass)   

data = np.column_stack((Ts, mass))
np.savetxt(
    "mass_change.dat",
    data,
    header="T mass_change",
    comments=""
)

plt.figure()
plt.plot(Ts,mass, marker='o')
plt.xlabel(r'$T$')
plt.ylabel('`% change in total mass')
plt.title('Change in Total Mass vs Final Time')
plt.show()

