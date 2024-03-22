using SPHExample
using BenchmarkTools
using StaticArrays
using Parameters
using StructArrays
import LinearAlgebra: dot, norm, diagm, diag, cond, det
using LoopVectorization
using FastPow
import CellListMap: InPlaceNeighborList, update!, neighborlist!
import ProgressMeter: next!
using Formatting
using Bumper
using TimerOutputs
using Distances

function SPHExample.LimitDensityAtBoundary!(Density,ρ₀, MotionLimiter)
    for i in eachindex(Density)
        if (Density[i] < ρ₀) * !Bool(MotionLimiter[i])
            Density[i] = ρ₀
        end
    end
end

import Base.Threads: nthreads, @threads
include("../src/ProduceVTP.jl")

#https://discourse.julialang.org/t/can-this-be-written-even-faster-cpu/109924/28
@inline function fancy7th(x)
    # todo tune the magic constant
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
@inline faux_fancy(ρ₀, P, invCb) = ρ₀ * ( fancy7th( 1 + (P * invCb)) - 1)

function update_arr1_bumper!(arr1,indices)
    @no_escape begin
        temp  = @alloc(eltype(arr1),length(arr1))

        temp .= @view arr1[indices]
        arr1 .= temp
        
    end
end


function ConstructGravitySVector(_::SVector{N, T}, value) where {N, T}
    return SVector{N, T}(ntuple(i -> i == N ? value : 0, N))
end

function ConstructStencil(v::Val{d}) where d
    n_ = CartesianIndices(ntuple(_->-1:1,v))
    half_length = length(n_) ÷ 2
    n  = n_[1:half_length]

    return n
end

###=== Extract Cells
function ExtractCells!(Cells, Points, CutOff)
    for i ∈ eachindex(Cells)
        Cells[i] =  CartesianIndex(@. Int(fld(Points[i], CutOff)) ...) + CartesianIndex(1,1) + CartesianIndex(1,1) #+ ZeroOffset + HalfPad
    end
    return nothing
end

###===

###=== SimStep
@inline function dρᵢdt_dρⱼdt_(ρᵢ,ρⱼ,m₀,vᵢⱼ,∇ᵢWᵢⱼ)
    #original: - ρᵢ * dot((m₀/ρⱼ) *  -vᵢⱼ ,  ∇ᵢWᵢⱼ)
    symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ) # = dot(vᵢⱼ , -∇ᵢWᵢⱼ)
    dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  symmetric_term
    dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  symmetric_term

    return dρdt⁺, dρdt⁻
end

@fastpow function EquationOfStateGamma7(ρ,c₀,ρ₀)
    return ((c₀^2*ρ₀)/7) * ((ρ/ρ₀)^7 - 1)
end


function SimStepLocalCell(Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, StartIndex, EndIndex, MotionLimiter, ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H², Cb⁻¹, BoolDDT=true)

    @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex

        xᵢⱼ² = evaluate(SqEuclidean(), Position[i], Position[j])

        if  xᵢⱼ² <= H²
            xᵢⱼ  = Position[i] - Position[j]
            
            dᵢⱼ  = sqrt(xᵢⱼ²) #Using sqrt is what takes a lot of time?
            q    = clamp(dᵢⱼ * h⁻¹,0.0,2.0)
            # Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)

            invd²η² = inv(dᵢⱼ*dᵢⱼ+η²)

            ∇ᵢWᵢⱼ = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 

            ρᵢ        = Density[i]
            ρⱼ        = Density[j]
        
            vᵢ        = Velocity[i]
            vⱼ        = Velocity[j]
            vᵢⱼ       = vᵢ - vⱼ

            dρdt⁺, dρdt⁻ = dρᵢdt_dρⱼdt_(ρᵢ,ρⱼ,m₀,vᵢⱼ,∇ᵢWᵢⱼ)

            # Density diffusion
            if BoolDDT
                Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
                ρᵢⱼᴴ  = faux_fancy(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
                Pⱼᵢᴴ  = -Pᵢⱼᴴ
                ρⱼᵢᴴ  = faux_fancy(ρ₀, Pⱼᵢᴴ, Cb⁻¹)
            
                ρⱼᵢ   = ρⱼ - ρᵢ

                MLcond = MotionLimiter[i] * MotionLimiter[j]
                ddt_symmetric_term =  δᵩ * h * c₀ * 2 * invd²η² * dot(-xᵢⱼ,  ∇ᵢWᵢⱼ) * MLcond #  dot(-xᵢⱼ,  ∇ᵢWᵢⱼ) =  dot( xᵢⱼ, -∇ᵢWᵢⱼ)
                Dᵢ  = ddt_symmetric_term * (m₀/ρⱼ) * ( ρⱼᵢ - ρᵢⱼᴴ)
                Dⱼ  = ddt_symmetric_term * (m₀/ρᵢ) * (-ρⱼᵢ - ρⱼᵢᴴ)

                dρdtI[i] += dρdt⁺ + Dᵢ
                dρdtI[j] += dρdt⁻ + Dⱼ
            else
                dρdtI[i] += dρdt⁺
                dρdtI[j] += dρdt⁻
            end

            Pᵢ        =  EquationOfStateGamma7(ρᵢ,c₀,ρ₀)
            Pⱼ        =  EquationOfStateGamma7(ρⱼ,c₀,ρ₀)
            Pfac      = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)

            ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
            cond      = dot(vᵢⱼ, xᵢⱼ)
            cond_bool = cond < 0.0
            μᵢⱼ       = h*cond * invd²η²
            Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
            Πⱼ        = - Πᵢ

            dvdt⁺ = - m₀ * (Pfac) *  ∇ᵢWᵢⱼ + Πᵢ
            dvdt⁻ = - dvdt⁺ + Πⱼ

            dvdtI[i] +=  dvdt⁺
            dvdtI[j] +=  dvdt⁻

            # Kernel[i] += Wᵢⱼ
            # Kernel[j] += Wᵢⱼ

            # KernelGradient[i] +=  ∇ᵢWᵢⱼ
            # KernelGradient[j] += -∇ᵢWᵢⱼ
        end
    end

    return nothing
end


function SimStepNeighborCell(Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, StartIndex, EndIndex, StartIndex_, EndIndex_, MotionLimiter, ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H², Cb⁻¹, BoolDDT=true)
    @inbounds for i = StartIndex:EndIndex, j = StartIndex_:EndIndex_

        xᵢⱼ² = evaluate(SqEuclidean(), Position[i], Position[j])

        if  xᵢⱼ² <= H²
            xᵢⱼ  = Position[i] - Position[j]

            dᵢⱼ  = sqrt(xᵢⱼ²) #Using sqrt is what takes a lot of time?
            q    = clamp(dᵢⱼ * h⁻¹,0.0,2.0)
            # Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)

            invd²η² = inv(dᵢⱼ*dᵢⱼ+η²)

            ∇ᵢWᵢⱼ = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 

            ρᵢ        = Density[i]
            ρⱼ        = Density[j]
        
            vᵢ        = Velocity[i]
            vⱼ        = Velocity[j]
            vᵢⱼ       = vᵢ - vⱼ

            dρdt⁺, dρdt⁻ = dρᵢdt_dρⱼdt_(ρᵢ,ρⱼ,m₀,vᵢⱼ,∇ᵢWᵢⱼ)
            # Density diffusion
            if BoolDDT
                Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
                ρᵢⱼᴴ  = faux_fancy(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
                Pⱼᵢᴴ  = -Pᵢⱼᴴ
                ρⱼᵢᴴ  = faux_fancy(ρ₀, Pⱼᵢᴴ, Cb⁻¹)
            
                ρⱼᵢ   = ρⱼ - ρᵢ

                # Dᵢ  = δᵩ * h * c₀ * (m₀/ρⱼ) * 2 * ( ρⱼᵢ - ρᵢⱼᴴ) * invd²η² * dot(-xᵢⱼ,  ∇ᵢWᵢⱼ) * MLcond
                # Dⱼ  = δᵩ * h * c₀ * (m₀/ρᵢ) * 2 * (-ρⱼᵢ - ρⱼᵢᴴ) * invd²η² * dot( xᵢⱼ, -∇ᵢWᵢⱼ) * MLcond
                MLcond = MotionLimiter[i] * MotionLimiter[j]
                ddt_symmetric_term =  δᵩ * h * c₀ * 2 * invd²η² * dot(-xᵢⱼ,  ∇ᵢWᵢⱼ) * MLcond #  dot(-xᵢⱼ,  ∇ᵢWᵢⱼ) =  dot( xᵢⱼ, -∇ᵢWᵢⱼ)
                Dᵢ  = ddt_symmetric_term * (m₀/ρⱼ) * ( ρⱼᵢ - ρᵢⱼᴴ)
                Dⱼ  = ddt_symmetric_term * (m₀/ρᵢ) * (-ρⱼᵢ - ρⱼᵢᴴ)


                dρdtI[i] += dρdt⁺ + Dᵢ
                dρdtI[j] += dρdt⁻ + Dⱼ
            else
                dρdtI[i] += dρdt⁺
                dρdtI[j] += dρdt⁻
            end

            Pᵢ        =  EquationOfStateGamma7(ρᵢ,c₀,ρ₀)
            Pⱼ        =  EquationOfStateGamma7(ρⱼ,c₀,ρ₀)
            Pfac      = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)

            ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
            cond      = dot(vᵢⱼ, xᵢⱼ)
            cond_bool = cond < 0.0
            μᵢⱼ       = h*cond * invd²η²
            Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
            Πⱼ        = - Πᵢ

            dvdt⁺ = - m₀ * (Pfac) *  ∇ᵢWᵢⱼ + Πᵢ
            dvdt⁻ = - dvdt⁺ + Πⱼ

            dvdtI[i] +=  dvdt⁺
            dvdtI[j] +=  dvdt⁻

            # Kernel[i] += Wᵢⱼ
            # Kernel[j] += Wᵢⱼ

            # KernelGradient[i] +=  ∇ᵢWᵢⱼ
            # KernelGradient[j] += -∇ᵢWᵢⱼ
        end
    end

    return nothing
end

###===

###=== Function to update ordering
#https://cuda.juliagpu.org/stable/tutorials/performance/
function UpdateNeighbors!(Cells, CutOff, SortedIndices, Position, Density, Acceleration, Velocity, GravityFactor, MotionLimiter, ParticleRanges, UniqueCells)
    ExtractCells!(Cells,Position,CutOff)

    sortperm!(SortedIndices,Cells)

    update_arr1_bumper!(Cells, SortedIndices)
    update_arr1_bumper!(Position, SortedIndices)
    update_arr1_bumper!(Density, SortedIndices)
    update_arr1_bumper!(Acceleration, SortedIndices)
    update_arr1_bumper!(Velocity, SortedIndices)    
    update_arr1_bumper!(GravityFactor, SortedIndices)    
    update_arr1_bumper!(MotionLimiter, SortedIndices)    

    # These two are equivalent lol
    # @time ParticleSplitter[findall(.!iszero.(diff(Cells))) .+ 1] .= true
    IndexCounter = 1
    for i in 2:length(Cells)
        if Cells[i] != Cells[i-1] # Equivalent to diff(Cells) != 0
            ParticleRanges[IndexCounter] = i
            UniqueCells[IndexCounter]    = Cells[i]
            IndexCounter += 1
        end
    end

    return IndexCounter
end
###===

###=== Function to process each cell and its neighbors
#https://cuda.juliagpu.org/stable/tutorials/performance/
# 192 bytes and 4 allocs from launch config
# INLINE IS SO IMPORTANT 10X SPEED
function NeighborLoop!(SimConstants, ParticleRanges, Stencil, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI,  MotionLimiter, Cells, UniqueCells, IndexCounter)
    @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H², Cb⁻¹ = SimConstants

    UniqueCells = view(UniqueCells, 1:IndexCounter)
    for iter ∈ eachindex(UniqueCells)
        CellIndex = UniqueCells[iter]

        StartIndex = ParticleRanges[iter] 
        EndIndex   = ParticleRanges[iter+1] - 1

        @inline SimStepLocalCell(Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, StartIndex, EndIndex, MotionLimiter, ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H², Cb⁻¹)

        @inbounds for S ∈ Stencil
            SCellIndex = CellIndex + S

            # Returns a range, x:x for exact match and x:(x-1) for no match
            # utilizes that it is a sorted array and requires no isequal constructor,
            # so I prefer this for now
            NeighborCellIndex = searchsorted(UniqueCells, SCellIndex)

            if length(NeighborCellIndex) != 0
                StartIndex_       = ParticleRanges[NeighborCellIndex[1]] 
                EndIndex_         = ParticleRanges[NeighborCellIndex[1]+1] - 1

                @inline SimStepNeighborCell(Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, StartIndex, EndIndex, StartIndex_, EndIndex_, MotionLimiter, ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H², Cb⁻¹)
            end
        end
    end

    return nothing
end

function SimulationLoop(SimMetaData, SimConstants, Cells, Stencil,  ParticleRanges, UniqueCells, SortedIndices, Position, Kernel, KernelGradient, Density, Velocity, Acceleration, dρdtI, dvdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, dρdtIₙ⁺, GravityFactor, MotionLimiter)
    dt  = Δt(Position, Velocity, Acceleration, SimConstants)
    dt₂ = dt * 0.5

    IndexCounter = UpdateNeighbors!(Cells, SimConstants.H, SortedIndices, Position, Density, Acceleration, Velocity, GravityFactor, MotionLimiter, ParticleRanges, UniqueCells)

    ResetArrays!(Kernel, KernelGradient, dρdtI, dvdtI)

    NeighborLoop!(SimConstants, ParticleRanges, Stencil, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI,  MotionLimiter, Cells, UniqueCells, IndexCounter)

    @inbounds for i in eachindex(Position)
        dvdtI[i]        +=  ConstructGravitySVector(dvdtI[i], SimConstants.g * GravityFactor[i])
        Velocityₙ⁺[i]    =  Velocity[i]   + dvdtI[i]  *  dt₂ * MotionLimiter[i]
        Positionₙ⁺[i]    =  Position[i]   + Velocityₙ⁺[i]   * dt₂  * MotionLimiter[i]
        ρₙ⁺[i]           =  Density[i]    + dρdtI[i]       *  dt₂
    end

    LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, MotionLimiter)

    ResetArrays!(Kernel, KernelGradient, dρdtI, dρdtIₙ⁺, Acceleration)

    NeighborLoop!(SimConstants, ParticleRanges, Stencil, Positionₙ⁺, Kernel, KernelGradient, ρₙ⁺, Velocityₙ⁺, dρdtIₙ⁺, Acceleration, MotionLimiter, Cells, UniqueCells, IndexCounter)

    DensityEpsi!(Density, dρdtIₙ⁺, ρₙ⁺, dt)

    LimitDensityAtBoundary!(Density, SimConstants.ρ₀, MotionLimiter)

    @inbounds for i in eachindex(Position)
        Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
        Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
        Position[i]       += (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
    end

    SimMetaData.Iteration      += 1
    SimMetaData.CurrentTimeStep = dt
    SimMetaData.TotalTime      += dt
end

###===

function RunSimulation(;FluidCSV::String,
    BoundCSV::String,
    SimMetaData::SimulationMetaData{Dimensions, FloatType},
    SimConstants::SimulationConstants,
    ViscosityTreatment = :LaminarSPS,
    BoolDDT = true,
    BoolShifting = true
    ) where {Dimensions,FloatType}

    if !isdir(SimMetaData.SaveLocation)
        mkdir(SimMetaData.SaveLocation)
    end
    
    foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))

    # Unpack the relevant simulation meta data
    @unpack HourGlass, SaveLocation, SimulationName, SilentOutput, ThreadsCPU = SimMetaData;

    # Unpack simulation constants
    @unpack ρ₀, dx, h, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H = SimConstants

    # Load in the fluid and boundary particles. Return these points and both data frames
    # @inline is a hack here to remove all allocations further down due to uncertainty of the points type at compile time
    @inline points, density_fluid, density_bound  = LoadParticlesFromCSV(Dimensions,FloatType, FluidCSV,BoundCSV)
    NumberOfPoints = length(points)
    Position = convert(Vector{SVector{Dimensions,FloatType}},points.V)
    Density  = deepcopy([density_bound; density_fluid])

    GravityFactor = [ zeros(size(density_bound,1)) ; -ones(size(density_fluid,1)) ]
    
    MotionLimiter = [ zeros(size(density_bound,1)) ;  ones(size(density_fluid,1)) ]

    Acceleration    = similar(Position)
    Velocity        = similar(Position)
    Kernel          = similar(Density)
    KernelGradient  = similar(Position)

    dρdtI  = similar(Density)

    dvdtI  = similar(Position)

    Velocityₙ⁺ = similar(Position)
    Positionₙ⁺ = similar(Position)
    ρₙ⁺        = zeros(FloatType, NumberOfPoints)
    dρdtIₙ⁺    = zeros(FloatType, NumberOfPoints)

    Pressureᵢ  = zeros(FloatType, NumberOfPoints)

    Cells      = similar(Position, CartesianIndex{Dimensions})

    ParticleRanges  = zeros(Int, length(Cells) + 1)
    UniqueCells     = zeros(CartesianIndex{Dimensions}, length(Cells))

    SortedIndices   = similar(Cells, Int)

    Stencil         = ConstructStencil(Val(Dimensions))

    # Ensure zero, similar does not!
    ResetArrays!(Acceleration, Velocity, Kernel, KernelGradient, Cells, SortedIndices, dρdtI, dvdtI, Positionₙ⁺, Velocityₙ⁺)

    SaveLocation_ = SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(0,6,"0") * ".vtp"
    Pressure!(Pressureᵢ,Density,SimConstants)
    PolyDataTemplate(SaveLocation_, to_3d(Position), ["Kernel", "KernelGradient", "Density", "Pressure","Velocity", "Acceleration"], Kernel, KernelGradient, Density, Pressureᵢ, Velocity, Acceleration)

    # Normal run and save data
    generate_showvalues(Iteration, TotalTime) = () -> [(:(Iteration),format(FormatExpr("{1:d}"),  Iteration)), (:(TotalTime),format(FormatExpr("{1:3.3f}"), TotalTime))]
    OutputCounter = 0.0
    OutputIterationCounter = 0
    @time @inbounds while true

        @timeit HourGlass "1 SimulationLoop" SimulationLoop(SimMetaData, SimConstants, Cells, Stencil, ParticleRanges, UniqueCells, SortedIndices, Position, Kernel, KernelGradient, Density, Velocity, Acceleration, dρdtI, dvdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, dρdtIₙ⁺, GravityFactor, MotionLimiter)

        OutputCounter += SimMetaData.CurrentTimeStep
        if OutputCounter >= SimMetaData.OutputEach
            OutputCounter = 0.0
            OutputIterationCounter += 1

            SaveLocation_ = SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(OutputIterationCounter,6,"0") * ".vtp"
            Pressure!(Pressureᵢ,Density,SimConstants)
            # @timeit HourGlass "2 Output Data"  PolyDataTemplate(SaveLocation_, to_3d(Position), ["Kernel", "KernelGradient", "Density", "Pressure","Velocity", "Acceleration"], Kernel, KernelGradient, Density, Pressureᵢ, Velocity, Acceleration)
        end
        # @timeit HourGlass "3 Next TimeStep"  next!(SimMetaData.ProgressSpecification; showvalues = generate_showvalues(SimMetaData.Iteration , SimMetaData.TotalTime))

        if SimMetaData.TotalTime >= SimMetaData.SimulationTime + 1e-3
            break
        end
    end
    show(HourGlass,sortby=:name)
end

to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]


let
    Dimensions = 2
    FloatType  = Float64

    SimMetaData  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="Test", 
        SaveLocation="E:/SecondApproach/TESTING_CPU",
        SimulationTime=1,
        OutputEach=0.01,
    )

    SimConstantsWedge = SimulationConstants{FloatType}(c₀=42.48576250492629)

    @profview RunSimulation(
        FluidCSV     = "./input/still_wedge_mdbc/StillWedge_Dp0.02_Fluid.csv",
        BoundCSV     = "./input/still_wedge_mdbc/StillWedge_Dp0.02_Bound.csv",
        SimMetaData  = SimMetaData,
        SimConstants = SimConstantsWedge
    )
end