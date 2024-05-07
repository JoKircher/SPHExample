using Plots

# Plots.plotly()
Plots.gr()

### Gauss Kernel plotten 
A = collect(-2.:0.02:2.)
# h = (1.2 * sqrt(2) * 0.02 ^2)
h = 1

function GaussW(q)
    return (1 / (pi * h^2)) * exp(-q^2)
    # return 1/(1*sqrt(2*pi)) * exp(-0.5(abs(q)-1/1))^2
end

testG = GaussW.(A)
Plots.plot(A, testG)

### Wendland Kernel plotten

function WendlandW(q)
    return 7 / (4 * π ) *(1-(q/2))^4*(2*q + 1)
    # return (1-q/2)^4*(2*q + 1)
end

testW = WendlandW.(abs.(A))

Plots.plot!(A, testW)
# fig = Plots.plot(A, testW;grid=false, legend=false, axis=([],false))
savefig("Wendland.png")
### Ableitung Wendland plotten

function dtWendlandW(q)
    # return 7 / (4 * π ) * 5/8 * (abs(q)-2)^3*q #/ (8*(q+0.01^2))
    return (7 / (4 * π ))* -5*q*(1-(abs(q)/2))^3 
end

testdt = dtWendlandW.(A)

Plots.plot!(A, testdt)