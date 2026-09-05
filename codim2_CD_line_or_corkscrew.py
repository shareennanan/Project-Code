"""
Codim-2 CutFEM on a 1D curve embedded in 3D  --  Convection-Diffusion

The curve is cut out of a 3D background mesh by TWO level sets at once:

    Gamma = { x in Omega : phi_1(x) = 0  and  phi_2(x) = 0 },


    geometry = "line"       Gamma = the z-axis   (phi_1 = x, phi_2 = y)
                            
    geometry = "corkscrew"  Gamma = a helix on the unit cylinder.
                           
"""
from ngsolve import *
from netgen.occ import *
from xfem import *
from xfem.mlset import *
import numpy as np
import matplotlib.pyplot as plt

# ================================== config ===================================
geometry = "line"   # "line" | "corkscrew"
order = 1                # P1 background space
gamma_n = 0.01          # normal-gradient stabilisation weight 


# ================================ geometry ===================================
# Each geometry block defines:
#   maxh                         background mesh size
#   make_mesh(maxh) -> Mesh      the 3D background mesh
#   level_sets      = (phi_1, phi_2)
#   periodic        bool         True if Gamma closes into a loop
#   u_exact, rhs                 manufactured solution and its right-hand side
#   w                            background flow (unused by Poisson; for the TODO)



alpha = 0.01                   #diffusion rate
allerrors={}
meshsize = [0.05,0.1]

for maxh in meshsize:
    if geometry == "line":
        #maxh = 0.05
        R_dom, z_max = 1.0, 1.0
    
        def make_mesh(maxh):
            cyl = Cylinder(Pnt(0, 0, 0), Z, r=R_dom, h=z_max)
            cyl.faces.Min(Z).Identify(cyl.faces.Max(Z), "periodic",
                                      IdentificationType.PERIODIC)
            return Mesh(OCCGeometry(cyl).GenerateMesh(maxh=maxh))
    
        level_sets = (x, y)               # Gamma = {x=0} n {y=0} = the z-axis
        periodic = True                 # open curve: Dirichlet at z = 0 and z = 1
        u_exact = lambda t: exp(-alpha * (2 * pi)**2 * t) * sin(2 * pi * (z - t))         # vanishes at z = 0, 1 -> matches Dirichlet
        w = CF((0, 0, 1))                 # flow along the line (for the TODO)
                            
    
    elif geometry == "corkscrew":
        #maxh = 0.05
        c_pitch = 0.4        # helix pitch: z = z0 + c_pitch*theta; z-rise per turn = 2*pi*c_pitch
        n_turns = 3          # number of turns inside the domain
        z0 = 0.0
        R_dom = 1.4          # ambient cylinder radius (> 1, so the helix r = 1 is interior)
        kmode = 2            # Fourier mode of the manufactured solution (INTEGER, see below)
    
        lam = sqrt(1 + c_pitch**2)       # arc-length stretch |d r / d theta|
        z_max = n_turns * 2 * pi * c_pitch
    
        def make_mesh(maxh):
            # Cylinder of height z_max with bottom/top faces IDENTIFIED, so the helix
            # (which climbs by z_max over n_turns turns) closes into a single loop.
            cyl = Cylinder(Pnt(0, 0, 0), Z, r=R_dom, h=z_max)
            cyl.faces.Min(Z).Identify(cyl.faces.Max(Z), "periodic",
                                      IdentificationType.PERIODIC)
            return Mesh(OCCGeometry(cyl).GenerateMesh(maxh=maxh))
    
        # The helix { (cos th, sin th, z0 + c*th) } on r = 1.
        phi_arg = (z - z0) / c_pitch
        level_sets = (x - cos(phi_arg), y - sin(phi_arg))
        periodic = True                  # closed curve -> Periodic(H1(...)) below
        w = CF((y,-x,0))
    	
        # On the curve the angle is theta = (z - z0)/c_pitch -- exactly, and without a
        # branch cut -- and the arc length is s = lam*theta.  Hence
        #     lap_Gamma = d^2/ds^2 = (1/lam^2) d^2/dtheta^2,
        # so for u = sin(k*theta):   -lap_Gamma u = (k^2/lam^2) u.
        # kmode must be an INTEGER: theta runs over [0, n_turns*2*pi], so sin(k*theta)
        # then matches across the periodic bottom/top identification.
        omega = 1.0 / (1 + c_pitch**2)
        theta = (z - z0) / c_pitch
        u_exact = lambda t: exp(-alpha * kmode**2 / lam**2 * t) * sin(kmode * (theta + omega * t))
       
    
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    
    
    # ============================== cut geometry =================================
    mesh = make_mesh(maxh)
    
    # The two level sets are INTERPOLATED to P1.  The discrete curve Gamma_h is the
    # intersection of the two P1 zero-sets -- that is what dCut integrates over.
    lsets_p1 = tuple(GridFunction(H1(mesh, order=1)) for _ in level_sets)
    for lset_p1, lset in zip(lsets_p1, level_sets):
        InterpolateToP1(lset, lset_p1)
    
    # (IF, IF) = "on the interface of phi_1 AND on the interface of phi_2" = the curve.
    line = DomainTypeArray((IF, IF))
    mlci = MultiLevelsetCutInfo(mesh, lsets_p1)
    els = mlci.GetElementsWithContribution(line)      # elements Gamma passes through
    
    # Sanity check: the LENGTH of Gamma.
    #   line:      1.0                       (the z-axis in a unit-height cylinder)
    #   corkscrew: n_turns * 2*pi * lam      (~20.30 for c_pitch=0.4, n_turns=3)
    length = Integrate(CoefficientFunction(1) * dCut(lsets_p1, line, order=3), mesh=mesh)
    print(f"geometry = {geometry!r}:  length(Gamma) = {length:.4f},  "
          f"cut elements = {sum(1 for e in els if e)}")
    
    
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
    if periodic:
        V = Periodic(H1(mesh, order=order))
    else:
        V = H1(mesh, order=order, dirichlet=".*")
    
    # THE ACTIVE-DOF RESTRICTION.  Only dofs belonging to elements that Gamma passes
    # through are "live"; every other dof is untouched by all integrators, so the
    # matrix is singular there.  Restricting the solve to `freedofs` is what makes the
    # method well posed. 
    freedofs = GetDofsOfElements(V, els) & V.FreeDofs()
    
    u, v = V.TnT()
    h = specialcf.mesh_size
    dGamma = dCut(lsets_p1, line, definedonelements=els)      # integrate along Gamma
    
    
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
    
    
    
    # =============================== solve + error calculation  ============================
    
    
    
    zeta = 0.5  #1 for Implicit Euler, 0.5 for Crank-Nicolson
    T = 4
    gfu = GridFunction(V)
    rhs = gfu.vec.CreateVector()
    dt0 = 0.5
    dtvals = []
    errors = []
    
    
    VTK = VTKOutput(mesh, coefs=[gfu, lsets_p1[0], lsets_p1[1], u_exact(0)],
              names=["gfu", "lset1", "lset2", "u_exact"],
             filename=f"codim2_CD_{geometry}", subdivision=0)
           
             
       
    for i_t in range(0,8):
        dt = 0.5**i_t*dt0
        mstar = a.mat.CreateMatrix()
        mstar.AsVector().data = m.mat.AsVector() + zeta * dt * a.mat.AsVector()
        inv = mstar.Inverse(freedofs, inverse="umfpack")
        
        t = 0
        gfu.Set(u_exact(0))
        
        nsteps = int(round(T / dt))
        for n in range(nsteps):
            rhs.data = m.mat * gfu.vec - (1 - zeta) * dt * (a.mat * gfu.vec)
            gfu.vec.data = inv * rhs 
            t += dt
            #VTK.Do(drawelems=els)
        err = sqrt(Integrate((gfu - u_exact(t))**2 * dGamma, mesh=mesh))
        print(f"dt={dt:.1e}, L2 error={err:.3e}")
        dtvals.append(dt)
        errors.append(err)
    allerrors[maxh]=np.array(errors)
        
    dtvals = np.array(dtvals)
    errors = np.array(errors)
C = allerrors[0.05][0] / dtvals[0]

print(allerrors[0.05][-1]/allerrors[0.1][-1])

plt.figure()
for maxh in [0.05,0.1]:
    if zeta == 1:
        plt.loglog(dtvals, C * dtvals, '--', label='O(dt)')
        plt.loglog(dtvals, errors, 'o-', label='Implicit Euler')
    elif zeta == 0.5:
        #plt.loglog(dtvals, C * dtvals**2, '--', label='$O(dt^2)$')
        plt.loglog(dtvals, allerrors[maxh], 'o-', label=fr'$h={maxh}$')
plt.loglog(dtvals, C * dtvals**2, '--', label='$O(dt^2)$')
plt.xlabel(r'$\Delta t$')
plt.ylabel('$||u(T)-u_h(T)||_{L^2}$')
plt.legend()
plt.title(r'$L^2$ error vs $\Delta t$, (Crank Nicolson scheme)')
plt.show()

