"""CODIM 1 TraceFEM: Convection-Diffusion on the unit circle"""


from ngsolve import *
from netgen.geom2d import SplineGeometry
from xfem import *
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

order = 1                       #P1 FE space
kmode = 2                     # Fourier mode of the manufactured soln
alpha = 0.1                    #Diffusion coefficient
maxhs=[0.025,0.05]      
s={}


#Looping solver over 2 mesh sizes
for maxh in maxhs:
    # -- background mesh: a plain box [-1.5, 1.5]^2 containing the circle
    geo = SplineGeometry()
    geo.AddRectangle((-1.5, -1.5), (1.5, 1.5))
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    
    # -- level set phi = sqrt(x^2+y^2)-1 (curved -> P1 circle is O(h^2) accurate)
    lset_exact = sqrt(x * x + y * y) - 1.0
    lset_approximated = GridFunction(H1(mesh, order=1))
    InterpolateToP1(lset_exact, lset_approximated )
    
    ci = CutInfo(mesh, lset_approximated)
    els = ci.GetElementsOfType(IF)              # elements the circle passes through
    n = 1.0 / Norm(grad(lset_approximated)) * grad(lset_approximated)     # surface normal (outward radial)
    Draw(n, mesh, "n")
    
    # -- active-dof space: plain P1 restricted to the cut elements
    V = H1(mesh, order=order)
    freedofs = GetDofsOfElements(V, els) & V.FreeDofs()
    u, vt = V.TnT(); h = specialcf.mesh_size
    Pt = lambda g: g - (g * n) * n              # tangential part of a gradient
    Qn = lambda g: (g * n) * n                  # normal part (for the stabilisation)
    dG = dCut(lset_approximated, IF, definedonelements=els)  # integrate along the circle
    w = CF((y, -x))    #convection field
    
    # -- bilinear form:        
    m = RestrictedBilinearForm(V, element_restriction=els, check_unused=False)
    m +=u * vt *dG
    m += 0.5 * h * Qn(grad(u)) * Qn(grad(vt)) * dx(definedonelements=els)  # normal-grad
    m.Assemble()
    
    
    a = RestrictedBilinearForm(V, element_restriction=els, check_unused=False)
    a += alpha * InnerProduct(Pt(Grad(u)), Pt(Grad(vt))) * dG
    a += InnerProduct(w, Pt(Grad(u))) * vt * dG
    a += 0.5 * h * Qn(grad(u)) * Qn(grad(vt)) * dx(definedonelements=els)    # normal-grad 
    a.Assemble()
    
    # -- right-hand side f = 0
    u_exact = lambda t: exp(-alpha * kmode**2 * t) * sin(kmode * (atan2(y, x) + t))
    f = LinearForm(V)
    f.Assemble()
    
    theta = 0.5  #1 for Implicit Euler, 0.5 for Crank-Nicolson
    T = 2
    gfu = GridFunction(V)
    rhs = gfu.vec.CreateVector()
    dt0 = 0.5
    dtvals = []
    errors = []
    
    VTK = VTKOutput(mesh, coefs=[gfu, lset_approximated], names=["gfu", "lset_approximated"],
              filename="vtkout_circle_CD", subdivision=0)
    
#-- Solve over different time steps for error plot
    for i_t in range(12):
        dt = 0.5**i_t*dt0
        mstar = a.mat.CreateMatrix()
        mstar.AsVector().data = m.mat.AsVector() + theta * dt * a.mat.AsVector()
        inv = mstar.Inverse(freedofs, inverse="umfpack")
        
        t = 0
        gfu.Set(u_exact(0))
        nsteps = int(round(T / dt))
        for n in range(nsteps):
            rhs.data = m.mat * gfu.vec - (1 - theta) * dt * (a.mat * gfu.vec)
            gfu.vec.data = inv * rhs 
            t += dt
            #VTK.Do()
        err = sqrt(Integrate((gfu - u_exact(t))**2 * dG, mesh=mesh))
        print(f"dt={dt:.1e}, L2 error={err:.3e}")
        dtvals.append(dt)
        errors.append(err)
        
        
    dtvals = np.array(dtvals)
    errors = np.array(errors)
    plt.loglog(dtvals, errors, 'o-', label=f'h={maxh}')
    if maxh==0.025:
       s[maxh]=errors[-1]
    if maxh==0.05:
        s[maxh]=errors[-1]
        
print(s[0.025]/s[0.05])
C = errors[0] / dtvals[0]
if theta == 1:
    plt.loglog(dtvals, C * dtvals, '--', label='O(dt)')
elif theta == 0.5:
    plt.loglog(dtvals, C * dtvals**2, '--', label='$O(dt^2)$')
plt.xlabel('$\Delta t$')
plt.ylabel('$||u(T)-u_h(T)||_{L^2}$')
plt.title(r'$L^2$ error vs $\Delta t$')
plt.legend()
plt.show()
	
	

