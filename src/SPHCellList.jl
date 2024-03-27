module SPHCellList

export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!, ComputeInteractions!

using Parameters, FastPow, StaticArrays, Base.Threads
import LinearAlgebra: dot

using ..SimulationEquations
using ..AuxillaryFunctions

    function ConstructStencil(v::Val{d}) where d
        n_ = CartesianIndices(ntuple(_->-1:1,v))
        half_length = length(n_) ÷ 2
        n  = n_[1:half_length]

        return n
    end

    @inline function ExtractCells!(Particles, CutOff)
        Cells  = @views Particles.Cells
        Points = @views Particles.Position
        Base.Threads.@threads for i ∈ eachindex(Particles)
            Cells[i]  =  CartesianIndex(@. Int(fld(Points[i], CutOff)) ...)
            Cells[i] +=  2 * one(Cells[i])  # + CartesianIndex(1,1) + CartesianIndex(1,1) #+ ZeroOffset + HalfPad
        end
        return nothing
    end

    ###=== Function to update ordering
    function UpdateNeighbors!(Particles, CutOff, SortingScratchSpace, ParticleRanges, UniqueCells)
        ExtractCells!(Particles, CutOff)

        sort!(Particles, by = p -> p.Cells; scratch=SortingScratchSpace)

        Cells = @views Particles.Cells
        @. ParticleRanges             = zero(eltype(ParticleRanges))
        IndexCounter                  = 1
        ParticleRanges[IndexCounter]  = 1
        UniqueCells[IndexCounter]     = Cells[1]

        for i in 2:length(Cells)
            if Cells[i] != Cells[i-1] # Equivalent to diff(Cells) != 0
                IndexCounter                += 1
                ParticleRanges[IndexCounter] = i
                UniqueCells[IndexCounter]    = Cells[i]
            end
        end
        ParticleRanges[IndexCounter + 1]  = length(ParticleRanges)

        return IndexCounter 
    end

    function ComputeInteractions!(SimConstants, SimParticles, Kernel, KernelGradient, dρdtI, dvdtI, i, j, ViscosityTreatment, BoolDDT, OutputKernelValues)
        Position      = @views SimParticles.Position
        Density       = @views SimParticles.Density
        Pressure      = @views SimParticles.Pressure
        Velocity      = @views SimParticles.Velocity
        MotionLimiter = @views SimParticles.MotionLimiter
        
        @unpack ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants
    
        xᵢⱼ  = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)                 #evaluate(SqEuclidean(), Position[i], Position[j]) from Distances.jl seemed slower
        if  xᵢⱼ² <= H²
            dᵢⱼ  = sqrt(xᵢⱼ²) #Using sqrt is what takes a lot of time?
            # Unsure what is faster, min should do less operations?
            q         = min(dᵢⱼ * h⁻¹, 2.0) #clamp(dᵢⱼ * h⁻¹,0.0,2.0)
            invd²η²   = inv(dᵢⱼ*dᵢⱼ+η²)
            ∇ᵢWᵢⱼ     = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 
            ρᵢ        = Density[i]
            ρⱼ        = Density[j]
        
            vᵢ        = Velocity[i]
            vⱼ        = Velocity[j]
            vᵢⱼ       = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ) # = dot(vᵢⱼ , -∇ᵢWᵢⱼ)
            dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  density_symmetric_term
            dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  density_symmetric_term
    
            # Density diffusion
            if BoolDDT
                Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
                ρᵢⱼᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
                Pⱼᵢᴴ  = -Pᵢⱼᴴ
                ρⱼᵢᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pⱼᵢᴴ, Cb⁻¹)
            
                ρⱼᵢ   = ρⱼ - ρᵢ
                MLcond = MotionLimiter[i] * MotionLimiter[j]
                ddt_symmetric_term =  δᵩ * h * c₀ * 2 * invd²η² * dot(-xᵢⱼ,  ∇ᵢWᵢⱼ) * MLcond #  dot(-xᵢⱼ,  ∇ᵢWᵢⱼ) =  dot( xᵢⱼ, -∇ᵢWᵢⱼ)
                Dᵢ  = ddt_symmetric_term * (m₀/ρⱼ) * ( ρⱼᵢ - ρᵢⱼᴴ)
                Dⱼ  = ddt_symmetric_term * (m₀/ρᵢ) * (-ρⱼᵢ - ρⱼᵢᴴ)
            else
                Dᵢ  = 0.0
                Dⱼ  = 0.0
            end
            dρdtI[i] += dρdt⁺ + Dᵢ
            dρdtI[j] += dρdt⁻ + Dⱼ
    
    
            Pᵢ      =  Pressure[i]
            Pⱼ      =  Pressure[j]
            Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
            dvdt⁺   = - m₀ * Pfac *  ∇ᵢWᵢⱼ
            dvdt⁻   = - dvdt⁺
    
            if ViscosityTreatment == :ArtificialViscosity
                ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
                cond      = dot(vᵢⱼ, xᵢⱼ)
                cond_bool = cond < 0.0
                μᵢⱼ       = h*cond * invd²η²
                Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
                Πⱼ        = - Πᵢ
            else
                Πᵢ        = zero(xᵢⱼ)
                Πⱼ        = Πᵢ
            end
        
            if ViscosityTreatment == :Laminar || ViscosityTreatment == :LaminarSPS
                # 4 comes from 2 divided by 0.5 from average density
                # should divide by ρᵢ eq 6 DPC
                # ν₀∇²uᵢ = (1/ρᵢ) * ( (4 * m₀ * (ρᵢ * ν₀) * dot( xᵢⱼ, ∇ᵢWᵢⱼ)  ) / ( (ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²) ) ) *  vᵢⱼ
                # ν₀∇²uⱼ = (1/ρⱼ) * ( (4 * m₀ * (ρⱼ * ν₀) * dot(-xᵢⱼ,-∇ᵢWᵢⱼ)  ) / ( (ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²) ) ) * -vᵢⱼ
                visc_symmetric_term = (4 * m₀ * ν₀ * dot( xᵢⱼ, ∇ᵢWᵢⱼ)) / ((ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²))
                # ν₀∇²uᵢ = (1/ρᵢ) * visc_symmetric_term *  vᵢⱼ * ρᵢ
                # ν₀∇²uⱼ = (1/ρⱼ) * visc_symmetric_term * -vᵢⱼ * ρⱼ
                ν₀∇²uᵢ =  visc_symmetric_term *  vᵢⱼ
                ν₀∇²uⱼ = -ν₀∇²uᵢ #visc_symmetric_term * -vᵢⱼ
            else
                ν₀∇²uᵢ = zero(xᵢⱼ)
                ν₀∇²uⱼ = ν₀∇²uᵢ
            end
        
            if ViscosityTreatment == :LaminarSPS 
                Iᴹ       = diagm(one.(xᵢⱼ))
                #julia> a .- a'
                # 3×3 SMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
                # 0.0  0.0  0.0
                # 0.0  0.0  0.0
                # 0.0  0.0  0.0
                # Strain *rate* tensor is the gradient of velocity
                Sᵢ = ∇vᵢ =  (m₀/ρⱼ) * (vⱼ - vᵢ) * ∇ᵢWᵢⱼ'
                norm_Sᵢ  = sqrt(2 * sum(Sᵢ .^ 2))
                νtᵢ      = (SmagorinskyConstant * dx)^2 * norm_Sᵢ
                trace_Sᵢ = sum(diag(Sᵢ))
                τᶿᵢ      = 2*νtᵢ*ρᵢ * (Sᵢ - (1/3) * trace_Sᵢ * Iᴹ) - (2/3) * ρᵢ * BlinConstant * dx^2 * norm_Sᵢ^2 * Iᴹ
                Sⱼ = ∇vⱼ =  (m₀/ρᵢ) * (vᵢ - vⱼ) * -∇ᵢWᵢⱼ'
                norm_Sⱼ  = sqrt(2 * sum(Sⱼ .^ 2))
                νtⱼ      = (SmagorinskyConstant * dx)^2 * norm_Sⱼ
                trace_Sⱼ = sum(diag(Sⱼ))
                τᶿⱼ      = 2*νtⱼ*ρⱼ * (Sⱼ - (1/3) * trace_Sⱼ * Iᴹ) - (2/3) * ρⱼ * BlinConstant * dx^2 * norm_Sⱼ^2 * Iᴹ
        
                
                dτdtᵢ = (m₀/(ρⱼ * ρᵢ)) * (τᶿᵢ + τᶿⱼ) *  ∇ᵢWᵢⱼ # MATHEMATICALLY THIS IS DOT PRODUCT TO GO FROM TENSOR TO VECTOR, BUT USE * IN JULIA THIS TIME
                dτdtⱼ = (m₀/(ρᵢ * ρⱼ)) * (τᶿᵢ + τᶿⱼ) * -∇ᵢWᵢⱼ # MATHEMATICALLY THIS IS DOT PRODUCT TO GO FROM TENSOR TO VECTOR, BUT USE * IN JULIA THIS TIME
            else
                dτdtᵢ  = zero(xᵢⱼ)
                dτdtⱼ  = dτdtᵢ
            end
        
            dvdtI[i] += dvdt⁺ + Πᵢ + ν₀∇²uᵢ + dτdtᵢ
            dvdtI[j] += dvdt⁻ + Πⱼ + ν₀∇²uⱼ + dτdtⱼ
    
            if OutputKernelValues
                Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)
                Kernel[i]         += Wᵢⱼ
                Kernel[j]         += Wᵢⱼ
                KernelGradient[i] +=  ∇ᵢWᵢⱼ
                KernelGradient[j] += -∇ᵢWᵢⱼ
            end
        end
    
        return nothing
    end

# Neither Polyester.@batch per core or thread is faster
###=== Function to process each cell and its neighbors
    function NeighborLoop!(SimConstants, SimParticles, ParticleRanges, Stencil, Kernel, KernelGradient, dρdtI, dvdtI, UniqueCells, IndexCounter, ViscosityTreatment, BoolDDT, OutputKernelValues)
        UniqueCells = view(UniqueCells, 1:IndexCounter)
        @inbounds Base.Threads.@threads for iter ∈ eachindex(UniqueCells)
            CellIndex = UniqueCells[iter]

            StartIndex = ParticleRanges[iter] 
            EndIndex   = ParticleRanges[iter+1] - 1

            @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex
                @inline ComputeInteractions!(SimConstants, SimParticles, Kernel, KernelGradient, dρdtI, dvdtI, i, j, ViscosityTreatment, BoolDDT, OutputKernelValues)
            end

            @inbounds for S ∈ Stencil
                SCellIndex = CellIndex + S

                # Returns a range, x:x for exact match and x:(x-1) for no match
                # utilizes that it is a sorted array and requires no isequal constructor,
                # so I prefer this for now
                NeighborCellIndex = searchsorted(UniqueCells, SCellIndex)

                if length(NeighborCellIndex) != 0
                    StartIndex_       = ParticleRanges[NeighborCellIndex[1]] 
                    EndIndex_         = ParticleRanges[NeighborCellIndex[1]+1] - 1

                    @inbounds for i = StartIndex:EndIndex, j = StartIndex_:EndIndex_
                        @inline ComputeInteractions!(SimConstants, SimParticles, Kernel, KernelGradient, dρdtI, dvdtI, i, j, ViscosityTreatment, BoolDDT, OutputKernelValues)
                    end
                end
            end
        end

        return nothing
    end

end
