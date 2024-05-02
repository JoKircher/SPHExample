module SimulationEquations

export EquationOfState, EquationOfStateGamma7, Pressure!, DensityEpsi!, LimitDensityAtBoundary!, ConstructGravitySVector, InverseHydrostaticEquationOfState

using StaticArrays
using Parameters
using FastPow

"""
    function EquationOfStateGamma7

Calculating the pressure from density with the Tait-Equation using the @fastpow macro

# Parameters
- `ρ`: density
- `c₀`: speed of sound
- `ρ₀`: Reference density

# Return
- pressure

# Example
```julia
EquationOfStateGamma7(Density[i],c₀,ρ₀)
```
""" 
@inline function EquationOfStateGamma7(ρ,c₀,ρ₀)
    return @fastpow ((c₀^2*ρ₀)/7) * ((ρ/ρ₀)^7 - 1)
end

"""
    function EquationOfState

Calculating the pressure from density with the Tait-Equation

# Parameters
- `ρ`: density
- `c₀`: speed of sound
- `γ`: adiabatic index
- `ρ₀`: Reference density

# Return
- pressure

# Example
```julia
EquationOfState(Density[i],c₀,γ,ρ₀)
```
"""
function EquationOfState(ρ,c₀,γ,ρ₀)
    return ((c₀^2*ρ₀)/γ) * ((ρ/ρ₀)^γ - 1)
end

"""
    function Pressure!(Press, Density, SimulationConstants)

Calculating the pressure for the given density

# Parameters
- `Press`: Pressure array where new pressure is stored in place
- `Density`: Density array from which pressure is calculated
- `SimulationConstants`: SimulationConstants struct from which constants are loaded

# Return
- `Press`: in-place changed pressure array

# Example
```julia
Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
```
"""
@inline function Pressure!(Press, Density, SimulationConstants)
    @unpack c₀,γ,ρ₀ = SimulationConstants
    @inbounds Base.Threads.@threads for i ∈ eachindex(Press,Density)
        # Press[i] = EquationOfState(Density[i],c₀,γ,ρ₀)
        Press[i] = EquationOfStateGamma7(Density[i],c₀,ρ₀)
    end
end

"""
    function DensityEpsi!(Density, dρdtIₙ⁺,ρₙ⁺,Δt)

This is to handle the special factor multiplied on density in the time stepping procedure, when using symplectic time stepping.

# Parameters
- `Density`: Particle Density array
- `dρdtI`: Density derivative
- `ρₙ⁺`: Half step density
- `Δt`: time delta

# Return
- `Density`: in-place changed density array

# Example
```julia
DensityEpsi!(Density, dρdtI, ρₙ⁺, dt)
```
"""
@inline function DensityEpsi!(Density, dρdtIₙ⁺,ρₙ⁺,Δt)
    @inbounds for i in eachindex(Density)
        epsi = - (dρdtIₙ⁺[i] / ρₙ⁺[i]) * Δt
        Density[i] *= (2 - epsi) / (2 + epsi)
    end
end

"""
    function LimitDensityAtBoundary!(Density, ρ₀, MotionLimiter)

Limit the density of boundary partices to ρ₀

# Parameters
- `Density`: Particle Density array
- `ρ₀`: reference density
- `MotionLimiter`: Identifies Boundary and fluid particles

# Return
- `Density`: in-place changed density array

# Example
```julia
LimitDensityAtBoundary!(Density, SimConstants.ρ₀, MotionLimiter)
```
"""
@inline function LimitDensityAtBoundary!(Density,ρ₀, MotionLimiter)
    @inbounds for i in eachindex(Density)
        if (Density[i] < ρ₀) * !Bool(MotionLimiter[i])
            Density[i] = ρ₀
        end
    end
end

"""
    function ConstructGravitySVector(_::SVector{N, T}, value) where {N, T}

Only fluid particles are influenced by gravity

# Parameters

# Return
- `Gravity`: vector of gravity influence on particles

# Example
```julia
ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
```
"""
@inline function ConstructGravitySVector(_::SVector{N, T}, value) where {N, T}
    return SVector{N, T}(ntuple(i -> i == N ? value : 0, N))
end

"""
    function Estimate7thRoot(x)

Function to estimate the 7th root of the input x. https://discourse.julialang.org/t/can-this-be-written-even-faster-cpu/109924/28

# Parameters
- `x`: input

# Return
- `t`: 7th root of `x`

# Example
```julia
Estimate7thRoot( 1 + (P * invCb))
```
"""
@inline function Estimate7thRoot(x)
    # initial guess based on fast inverse sqrt trick but adjusted to compute x^(1/7)
    t = copysign(reinterpret(Float64, 0x36cd000000000000 + reinterpret(UInt64,abs(x))÷7), x)
    @fastmath for _ in 1:2
        # newton's method for t^3 - x/t^4 = 0
        t2 = t*t
        t3 = t2*t
        t4 = t2*t2
        xot4 = x/t4
        t = t - t*(t3 - xot4)/(4*t3 + 3*xot4)
    end
    t
end
@inline InverseHydrostaticEquationOfState(ρ₀, P, invCb) = ρ₀ * ( Estimate7thRoot( 1 + (P * invCb)) - 1)

end
