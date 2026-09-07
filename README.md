Project Code

CutFEM implementation for a 1D curve embedded on the surface of a 3D heart mesh.
Code provided covers the progression from codim-0 to codim-2, accounting for any major codimension and geometry changes.

Codimension-0:
[Snippet1](standalone_1D_convection.py) is the (only) standalone implementation of FEM to solve convection-diffusion in 1D.
[Snippet2](codim0_Poisson_2D3D.py) is the NGSolve implementation of FEM to solve the Poisson equation on 2D and 3D meshes.

Codimension-1 (TraceFEM):
[Snippet3](codim1_CD_circle.py) solves the convection-diffusion equation on a circle embedded in a 2D background mesh.

Codimension-2 (CutFEM):
[Snippet4](codim2_CD_line_or_corkscrew.py) solves the convection-diffusion equation on line/corkscrew geometries embedded in a 3D cylinder mesh.
[Snippet5](codim2_heart_ring.py) solves the convection-diffusion equation on a ring on the surface of a 3D heart mesh.
[Snippet6](codim2_heart_sin.py) solves the convection-diffusion equation on an oscillating curve on the surface of a 3D heart mesh.
