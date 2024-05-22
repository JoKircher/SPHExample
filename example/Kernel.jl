using Plots

# Plots.plotly()
Plots.gr()

### Gauss Kernel plotten 
A = collect(-2.:0.02:2.)
h = 1

function GaussW(q)
    return (1 / (pi * h^2)) * exp(-q^2)
end

testG = GaussW.(A)
Plots.plot(A, testG)

### Wendland Kernel plotten

function WendlandW(q)
    return 7 / (4 * π ) *(1-(q/2))^4*(2*q + 1)
end

testW = WendlandW.(abs.(A))

Plots.plot!(A, testW)
# fig = Plots.plot(A, testW;grid=false, legend=false, axis=([],false))
savefig("Wendland.png")
### Ableitung Wendland plotten

function dtWendlandW(q)
    return (7 / (4 * π ))* -5*q*(1-(abs(q)/2))^3 
end

testdt = dtWendlandW.(A)

Plots.plot!(A, testdt)

x = rand(-2.0:0.1:2, 10)
# x =[0.5, -0.1, 1.7, -1.2]
B = rand(50:100, 10)
# B = [0.5, 2, 10, 15]

sca = WendlandW.(abs.(x))

C = sca .* B

C = C ./ maximum(C)

Plots.plot(A, testW)
Plots.scatter!(x, sca, ma=C) 