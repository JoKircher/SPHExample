using CairoMakie


f = Figure()
Axis(f[1, 1])

density!(randn(200))

f

using Meshes
# using MeshViz

# choose Makie backend
import GLMakie as Mke

grid = CartesianGrid(30, 30)
vals = rand(1:3, 30, 30)

# visualize grid
# viz(grid, color=vec(vals))

# visualize centroids
Meshes.viz(centroid.(grid))
viz!()

### Kernel plotten ####

using Plots

Plots.plotly()

### Gauss Kernel plotten 
A = collect(0:0.02:2.)
# h = (1.2 * sqrt(2) * 0.02 ^2)
h = 1

function GaussW(q)
    return (1 / (pi * h^2)) * exp(-q^2)
    # return exp(-q^2)
end

testG = GaussW.(A)
Plots.plot(A, testG)

### Wendland Kernel plotten

function WendlandW(q)
    return 7 / (4 * π ) *(1-q/2)^4*(2*q + 1)
    # return (1-q/2)^4*(2*q + 1)
end



testW = WendlandW.(A)



Plots.plot(A, testW)

### Ableitung Wendland plotten

function dtWendlandW(q)
    return 7 / (4 * π ) * 5/8 * (q-2)^3*q #/ (8*(q+0.01^2))
end

testdt = dtWendlandW.(A)

Plots.plot!(A, testdt)