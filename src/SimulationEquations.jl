module SimulationEquations

export Wᵢⱼ, ∑ⱼWᵢⱼ!, Optim∇ᵢWᵢⱼ, ∑ⱼ∇ᵢWᵢⱼ, Pressure, ∂Πᵢⱼ∂t, ∂ρᵢ∂tDDT!, ∂vᵢ∂t

using CellListMap
using StaticArrays
using LinearAlgebra
using Parameters

# Function to calculate Kernel Value
function Wᵢⱼ(αD,q)
    return αD*(1-q/2)^4*(2*q + 1)
end

# Function to calculate kernel value in both "particle i" format and "list of interactions" format
# Please notice how when using CellListMap since it is based on a "list of interactions", for each 
# interaction we must add the contribution to both the i'th and j'th particle!
function ∑ⱼWᵢⱼ!(Kernel, list,SimulationConstants)
    @unpack αD,H = SimulationConstants
    for (iter,L) in enumerate(list)
        i = L[1]; j = L[2]; d = L[3]

        q = d/H

        W = Wᵢⱼ(αD,q)

        Kernel[i] += W
        Kernel[j] += W
    end

    return nothing
end

# Original implementation of kernel gradient
# function ∇ᵢWᵢⱼ(αD,q,xᵢⱼ,h)
#     # Skip distances outside the support of the kernel:
#     if q < 0.0 || q > 2.0
#         return SVector(0.0,0.0,0.0)
#     end

#     gradWx = αD * 1/h * (5*(q-2)^3*q)/8 * (xᵢⱼ[1] / (q*h+1e-6))
#     gradWy = αD * 1/h * (5*(q-2)^3*q)/8 * (xᵢⱼ[2] / (q*h+1e-6))
#     gradWz = αD * 1/h * (5*(q-2)^3*q)/8 * (xᵢⱼ[3] / (q*h+1e-6)) 

#     return SVector(gradWx,gradWy,gradWz)
# end

# This is a much faster version of ∇ᵢWᵢⱼ
function Optim∇ᵢWᵢⱼ(αD,q,xᵢⱼ,h) 
    # Skip distances outside the support of the kernel:
    if 0 < q < 2
        Fac = αD*5*(q-2)^3*q / (8h*(q*h+1e-6)) 
    else
        Fac = 0.0 # or return zero(xᵢⱼ) 
    end
    return Fac .* xᵢⱼ
end

# Function to calculate kernel gradient value in both "particle i" format and "list of interactions" format
# Please notice how when using CellListMap since it is based on a "list of interactions", for each 
# interaction we must add the contribution to both the i'th and j'th particle!
function ∑ⱼ∇ᵢWᵢⱼ!(KernelGradientI, list,points,SimulationConstants)
    @unpack αD,H = SimulationConstants
 
    sumWgL = zeros(SVector{3,Float64},length(list))
    for (iter,L) in enumerate(list)
        i = L[1]; j = L[2]; d = L[3]

        xᵢⱼ = points[i] - points[j]

        q = d/H

        Wg = Optim∇ᵢWᵢⱼ(αD,q,xᵢⱼ,H)

        KernelGradientI[i] +=  Wg
        KernelGradientI[j] += -Wg

        sumWgL[iter] = Wg
    end

    return sumWgL
end

# Equation of State in Weakly-Compressible SPH
function Pressure(ρ,c₀,γ,ρ₀)
    return ((c₀^2*ρ₀)/γ) * ((ρ/ρ₀)^γ - 1)
end

# The artificial viscosity term
function ∂Πᵢⱼ∂t(list,points,ρ,v,WgL,SimulationConstants)

    @unpack H, α, c₀, m₀, η² = SimulationConstants

    N    = length(points)

    viscI = zeros(SVector{3,Float64},N)
    viscL = zeros(SVector{3,Float64},length(list))
    for (iter,L) in enumerate(list)
        i = L[1]; j = L[2];
        
        ρᵢ    = ρ[i]
        ρⱼ    = ρ[j]
        vᵢⱼ   = v[i] - v[j]
        xᵢⱼ   = points[i] - points[j]
        ρᵢⱼ   = (ρᵢ+ρⱼ)*0.5

        cond      = dot(vᵢⱼ,xᵢⱼ)

        cond_bool = cond < 0

        μᵢⱼ = H*cond/(dot(xᵢⱼ,xᵢⱼ)+η²)
        Πᵢⱼ = cond_bool*(-α*c₀*μᵢⱼ)/ρᵢⱼ
        
        viscI[i] += -Πᵢⱼ*m₀*WgL[iter]
        viscI[j] +=  Πᵢⱼ*m₀*WgL[iter]

        viscL[iter] = -Πᵢⱼ*m₀*WgL[iter]
    end

    return viscI,viscL
end

# The density derivative function WITHOUT density diffusion
function ∂ρᵢ∂t(list,points,m,ρ,v,WgL)
    N    = length(points)

    dρdtI = zeros(N)
    dρdtL = zeros(length(list))
    for (iter,L) in enumerate(list)
        i = L[1]; j = L[2]

        ρᵢ    = ρ[i]
        ρⱼ    = ρ[j]
        vᵢⱼ   = v[i] - v[j]
        ∇ᵢWᵢⱼ = WgL[iter]

        dρdtI[i] += ρᵢ*dot((m/ρⱼ)*vᵢⱼ,∇ᵢWᵢⱼ)
        dρdtI[j] += ρⱼ*dot((m/ρᵢ)*-vᵢⱼ,-∇ᵢWᵢⱼ)

        dρdtL[iter] = ρᵢ*dot((m/ρⱼ)*vᵢⱼ,∇ᵢWᵢⱼ)
    end

    return dρdtI,dρdtL
end

# The density derivative function INCLUDING density diffusion
function ∂ρᵢ∂tDDT!(dρdtI, list,points,ρ,v,WgL,MotionLimiter, SimulationConstants)
    @unpack H,m₀,δᵩ,c₀,γ,g,ρ₀,η² = SimulationConstants

    for (iter,L) in enumerate(list)
        i = L[1]; j = L[2];

        xᵢⱼ   = points[i] - points[j]
        ρᵢ    = ρ[i]
        ρⱼ    = ρ[j]
        vᵢⱼ   = v[i] - v[j]
        ∇ᵢWᵢⱼ = WgL[iter]

        Cb    = (c₀^2*ρ₀)/γ

        r²    = dot(xᵢⱼ,xᵢⱼ)

        DDTgz = ρ₀*g/Cb
        DDTkh = 2*H*δᵩ
        # Do note that in a lot of papers they write "ij"
        # BUT it should be ji for the direction to match (in dot3)
        # the density direction
        # For particle i
        drz   = xᵢⱼ[2]
        rh    = 1 + DDTgz*drz
        drhop = ρ₀* ^(rh,1/γ) - ρ₀
        visc_densi = DDTkh*c₀*(ρⱼ-ρᵢ-drhop)/(r²+η²)
        dot3  = dot(-xᵢⱼ,∇ᵢWᵢⱼ)
        delta_i = visc_densi*dot3*m₀/ρⱼ

        # For particle j
        drz   = -xᵢⱼ[2]
        rh    = 1 + DDTgz*drz
        drhop = ρ₀* ^(rh,1/γ) - ρ₀
        visc_densi = DDTkh*c₀*(ρᵢ-ρⱼ-drhop)/(r²+η²)
        dot3  = dot(xᵢⱼ,-∇ᵢWᵢⱼ)
        delta_j = visc_densi*dot3*m₀/ρᵢ

        dρdtI[i] += dot(m₀*vᵢⱼ,∇ᵢWᵢⱼ)+delta_i*MotionLimiter[i]
        dρdtI[j] += dot(m₀*-vᵢⱼ,-∇ᵢWᵢⱼ)+delta_j*MotionLimiter[j]
    end

    return nothing
end

# The momentum equation without any dissipation - we add the dissipation using artificial viscosity (∂Πᵢⱼ∂t)
function ∂vᵢ∂t(list,points,ρ,WgL, SimulationConstants)
    @unpack m₀, c₀,γ,ρ₀ = SimulationConstants

    N    = length(points)

    dvdtI = fill(SVector(0.0,0.0,0.0),N)
    dvdtL = fill(SVector(0.0,0.0,0.0),length(list))
    for (iter,L) in enumerate(list)
        i = L[1]; j = L[2]

        ρᵢ    = ρ[i]
        ρⱼ    = ρ[j]
        Pᵢ    = Pressure(ρᵢ,c₀,γ,ρ₀)
        Pⱼ    = Pressure(ρⱼ,c₀,γ,ρ₀)
        ∇ᵢWᵢⱼ = WgL[iter]

        Pfac  = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)


        dvdt  = - m₀ * Pfac *  ∇ᵢWᵢⱼ

        dvdtI[i]    +=  dvdt
        dvdtI[j]    +=  -dvdt
        
        dvdtL[iter] =   dvdt
    end

    return dvdtI,dvdtL
end

end