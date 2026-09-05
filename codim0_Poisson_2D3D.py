from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh
from ngsolve.meshes import Make1DMesh
from netgen import gui
from ngsolve import Draw

MESH = "square"             # "square" (2D unit square) or "la" (3D LA mesh) 
maxh = 0.1               # mesh size for the 2D unit square
n = 10			#no of elements for line mesh
order = 2                   # H1 polynomial order

if MESH == "square":
    from netgen.geom2d import unit_square
    # mesh=MakeStructured2DMesh(quads= False, nx = 10, ny = 10)
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    filename = "square_output"
    name = "square_solution"
elif MESH == "la":
    mesh = Mesh("03_fastl_LA.vol")
    filename = "la_output"
    name="la_solution"
else:
    raise ValueError(f"unknown MESH {MESH!r}; use 'square' or 'la'")

#la boundaries: ('right_pulmonary_veins', 'left_pulmonary_veins', 'mitral_valve_ring', 'endocardium', 'epicardium')

fes = H1(mesh, order=1, dirichlet=".*")

u = fes.TrialFunction()
v = fes.TestFunction()

f = LinearForm(fes)
f += 32*(y*(1-y)+x*(1-x))*v*dx

a = BilinearForm(fes)
a += grad(u)*grad(v)*dx

a.Assemble()
f.Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

u_exact= 16*x*(1-x)*y*(1-y)
error_L2 = sqrt(Integrate((gfu-u_exact)**2, mesh))
print("L2 error =", error_L2)
exit()
Draw (gfu, mesh, "Solution", autoscale=True)



vtk = VTKOutput(mesh, coefs=[gfu], names=[name], filename=filename, subdivision=0)
vtk.Do()




