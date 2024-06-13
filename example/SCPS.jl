using SPHExample
using StaticArrays
import StructArrays: StructArray
import LinearAlgebra: dot, norm, diagm, diag, cond, det
import Parameters: @unpack
import FastPow: @fastpow
import ProgressMeter: next!, finish!
using Format
using TimerOutputs
using Logging, LoggingExtras
using HDF5
using Base.Threads

"""
    function ComputeInteractions!(SimMetaData, SimConstants, Position, KernelThreaded, KernelGradientThreaded, Density, Pressure, Velocity, dρdtI, dvdtI, i, j, MotionLimiter, ichunk)

Function to process each interaction

# Parameters
- `SimMetaData`: Simulation Meta data.
- `SimConstants`: Simulation constants.
- `Position`: Particle postions.
- `KernelThreaded`: Threaded Kernel values.
- `KernelGradientThreaded`: Threaded Kernel gradient values.
- `Density`: Particle densities.
- `Pressure`: Particle pressures.
- `Velocity`: Particle Velocities.
- `dρdtI`: Density derivative values.
- `dvdtI`: Velocity derivative values.
- `i`: Index of particle `i`
- `j`: Index of particle `j`
- `MotionLimiter`: Identifies Boundary and fluid particles.
- `ichunk`: Thread chunks, part allocated to each thread.

# Example
```julia
ComputeInteractions!(SimMetaData, SimConstants, Position, Kernel, KernelGradient, Density, Pressure, Velocity, dρdtI, dvdtI, i, j, MotionLimiter, ichunk)
```
"""
function ComputeInteractions!(SimMetaData, SimConstants, Position, KernelThreaded, KernelGradientThreaded, Density, Pressure, Velocity, dρdtI, dvdtI, i, j, MotionLimiter, ichunk)
    @unpack FlagViscosityTreatment, FlagDensityDiffusion, FlagOutputKernelValues = SimMetaData
    @unpack ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants

    xᵢⱼ  = Position[i] - Position[j]
    xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)              
    if  xᵢⱼ² <= H²
        dᵢⱼ  = sqrt(abs(xᵢⱼ²))

        q         = min(dᵢⱼ * h⁻¹, 2.0)
        invd²η²   = inv(dᵢⱼ*dᵢⱼ+η²)
        ∇ᵢWᵢⱼ     = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 
        ρᵢ        = Density[i]
        ρⱼ        = Density[j]
    
        vᵢ        = Velocity[i]
        vⱼ        = Velocity[j]
        vᵢⱼ       = vᵢ - vⱼ
        density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
        dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  density_symmetric_term
        dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  density_symmetric_term

        # Density diffusion
        if FlagDensityDiffusion
            Dᵢ, Dⱼ = diffusion_term(ρ₀, g, xᵢⱼ,Cb⁻¹,ρⱼ, ρᵢ, dᵢⱼ, η², h, c₀, m₀, ∇ᵢWᵢⱼ, δᵩ)
        else
            Dᵢ  = 0.0
            Dⱼ  = 0.0
        end
        MLcond = MotionLimiter[i] * MotionLimiter[j]
        dρdtI[ichunk][i] += dρdt⁺ + Dᵢ * MLcond
        dρdtI[ichunk][j] += dρdt⁻ + Dⱼ * MLcond


        Pᵢ      =  Pressure[i]
        Pⱼ      =  Pressure[j]
        Pfac    = ((Pᵢ/ρᵢ^2)+ (Pⱼ/ρⱼ^2))
        dvdt⁺   = - m₀ * Pfac *  ∇ᵢWᵢⱼ
        dvdt⁻   = - dvdt⁺

        if FlagViscosityTreatment == :ArtificialViscosity
            Πᵢ, Πⱼ = viscosity_term(ρᵢ, ρⱼ, vᵢⱼ, xᵢⱼ, h, invd²η², m₀, α, c₀, ∇ᵢWᵢⱼ)
        else
            Πᵢ        = zero(xᵢⱼ)
            Πⱼ        = Πᵢ
        end
    
        dvdtI[ichunk][i] += dvdt⁺ + Πᵢ 
        dvdtI[ichunk][j] += dvdt⁻ + Πⱼ

        if FlagOutputKernelValues
            Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)
            KernelThreaded[ichunk][i]         += Wᵢⱼ
            KernelThreaded[ichunk][j]         += Wᵢⱼ
            KernelGradientThreaded[ichunk][i] +=  ∇ᵢWᵢⱼ
            KernelGradientThreaded[ichunk][j] += -∇ᵢWᵢⱼ
        end

    end

    return nothing
end

"""
    function reduce_sum!(target_array, arrays)

Reduce threaded arrays

# Parameters
- `target_array`: array where data should be reduced to.
- `arrays`: array with unreduced data.

# Example
```julia
reduce_sum!(dρdtI, dρdtIThreaded)
```
"""
@inline function reduce_sum!(target_array, arrays)
    for array in arrays
        target_array .+= array
    end
end

"""
    function SimulationLoop(ComputeInteractions!, SimMetaData, SimConstants, SimParticles, Stencil,  ParticleRanges, UniqueCells, SortingScratchSpace, Kernel, KernelThreaded, KernelGradient, KernelGradientThreaded, dρdtI, dρdtIThreaded, AccelerationThreaded, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, InverseCutOff)

Function to control the simulation loop

# Parameters
- `ComputeInteractions!`: Function to process interactions.
- `SimMetaData`: Simulation Meta data.
- `SimConstants`: Simulation constants.
- `SimParticles`: Particles in the simulation.
- `Stencil`: Kernel stencil.
- `ParticleRanges`: range of possible particles in neighborhood.
- `UniqueCells`: Identify closeby cells.
- `SortingScratchSpace`: Sorting algorithm.
- `Kernel`: Kernel values.
- `KernelThreaded`: Threaded Kernel values.
- `KernelGradient`: Kernel gradient values.
- `KernelGradientThreaded`: Threaded Kernel gradient values.
- `dρdtI`: Density derivative values.
- `dρdtIThreaded`: Threaded Density derivative values.
- `AccelerationThreaded`: Threaded Acceleration values.
- `Velocityₙ⁺`: Half step Velocities.
- `Positionₙ⁺`: Half step Positions.
- `ρₙ⁺`: Half step densities.
- `InverseCutOff`: Multiplicative inverse of CutOff distance.

# Example
```julia
SimulationLoop(ComputeInteractions!, SimMetaData, SimConstants, SimParticles, Stencil, ParticleRanges, UniqueCells, SortingScratchSpace, Kernel, KernelThreaded, KernelGradient, KernelGradientThreaded, dρdtI, dρdtIThreaded, AccelerationThreaded, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, InverseCutOff)
```
"""

@inbounds function SimulationLoop(ComputeInteractions!, SimMetaData, SimConstants, SimParticles, Stencil,  ParticleRanges, UniqueCells, SortingScratchSpace, Kernel, KernelThreaded, KernelGradient, KernelGradientThreaded, dρdtI, dρdtIThreaded, AccelerationThreaded, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, InverseCutOff)
    Position      = SimParticles.Position
    Density       = SimParticles.Density
    Pressure      = SimParticles.Pressure
    Velocity      = SimParticles.Velocity
    Acceleration  = SimParticles.Acceleration
    GravityFactor = SimParticles.GravityFactor
    MotionLimiter = SimParticles.MotionLimiter

    @timeit SimMetaData.HourGlass "01 Update TimeStep"  dt  = Δt(Position, Velocity, Acceleration, SimConstants)
    dt₂ = dt * 0.5

    @timeit SimMetaData.HourGlass "02 Calculate IndexCounter" IndexCounter = UpdateNeighbors!(SimParticles, InverseCutOff, SortingScratchSpace,  ParticleRanges, UniqueCells)
   
    @timeit SimMetaData.HourGlass "03 ResetArrays"                           ResetArrays!(Kernel, KernelGradient, dρdtI, Acceleration); ResetArrays!.(KernelThreaded, KernelGradientThreaded, dρdtIThreaded, AccelerationThreaded)

    Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
    @timeit SimMetaData.HourGlass "04 First NeighborLoop"                    NeighborLoop!(ComputeInteractions!, SimMetaData, SimConstants, ParticleRanges, Stencil, Position, KernelThreaded, KernelGradientThreaded, Density, Pressure, Velocity, dρdtIThreaded, AccelerationThreaded,  MotionLimiter, UniqueCells, IndexCounter)
    @timeit SimMetaData.HourGlass "04A Reduction"                            reduce_sum!(dρdtI, dρdtIThreaded)
    @timeit SimMetaData.HourGlass "04B Reduction"                            reduce_sum!(Acceleration, AccelerationThreaded)

    @timeit SimMetaData.HourGlass "05 Update To Half TimeStep" @inbounds for i in eachindex(Position)
        Acceleration[i]  +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
        Positionₙ⁺[i]     =  Position[i]   + Velocity[i]   * dt₂  * MotionLimiter[i]
        Velocityₙ⁺[i]     =  Velocity[i]   + Acceleration[i]  *  dt₂ * MotionLimiter[i]
        ρₙ⁺[i]            =  Density[i]    + dρdtI[i]       *  dt₂
    end

    @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"  LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, MotionLimiter)

    @timeit SimMetaData.HourGlass "07 ResetArrays"                  ResetArrays!(Kernel, KernelGradient, dρdtI, Acceleration); ResetArrays!.(KernelThreaded, KernelGradientThreaded, dρdtIThreaded, AccelerationThreaded)

    Pressure!(SimParticles.Pressure, ρₙ⁺,SimConstants)
    @timeit SimMetaData.HourGlass "08 Second NeighborLoop"          NeighborLoop!(ComputeInteractions!, SimMetaData, SimConstants, ParticleRanges, Stencil, Positionₙ⁺, KernelThreaded, KernelGradientThreaded, ρₙ⁺, Pressure, Velocityₙ⁺, dρdtIThreaded, AccelerationThreaded, MotionLimiter, UniqueCells, IndexCounter)
    @timeit SimMetaData.HourGlass "08A Reduction"                   reduce_sum!(dρdtI, dρdtIThreaded)
    @timeit SimMetaData.HourGlass "08B Reduction"                   reduce_sum!(Acceleration, AccelerationThreaded)

    @timeit SimMetaData.HourGlass "09 Final LimitDensityAtBoundary" LimitDensityAtBoundary!(Density, SimConstants.ρ₀, MotionLimiter)

    @timeit SimMetaData.HourGlass "10 Final Density"                DensityEpsi!(Density, dρdtI, ρₙ⁺, dt)


    @timeit SimMetaData.HourGlass "11 Update To Final TimeStep"  @inbounds for i in eachindex(Position)
        Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
        Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
        Position[i]       +=  (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
    end

    SimMetaData.Iteration      += 1
    SimMetaData.CurrentTimeStep = dt
    SimMetaData.TotalTime      += dt

    if SimMetaData.FlagOutputKernelValues
        reduce_sum!(Kernel, KernelThreaded)
        reduce_sum!(KernelGradient, KernelGradientThreaded)
    end
    
    return nothing
end

"""
    function RunSimulation(;FluidCSV::String,
        BoundCSV::String,
        SimMetaData::SimulationMetaData{Dimensions, FloatType},
        SimConstants::SimulationConstants,
        SimLogger::SimulationLogger
        ) where {Dimensions,FloatType}

Function to process each cell and its neighbors

# Parameters
- `FluidCSV::String`: Location of fluid particle .csv file.
- `BoundCSV::String`: Location of boundary particle .csv file.
- `SimMetaData::SimulationMetaData{Dimensions, FloatType}`: Simulation meta data.
- `SimConstants::SimulationConstants`: Simulation constants.
- `SimLogger::SimulationLogger`: Simulation logger.

# Example
```julia
RunSimulation(
    FluidCSV           = "./input/dam_break_2d/DamBreak2d_Dp0.02_Fluid_OneLayer.csv",
    BoundCSV           = "./input/dam_break_2d/DamBreak2d_Dp0.02_Bound_ThreeLayers.csv",
    SimMetaData        = SimMetaDataDamBreak,
    SimConstants       = SimConstantsDamBreak,
    SimLogger          = SimLogger
)
```
"""
function RunSimulation(;FluidCSV::String,
    BoundCSV::String,
    SimMetaData::SimulationMetaData{Dimensions, FloatType},
    SimConstants::SimulationConstants,
    SimLogger::SimulationLogger
    ) where {Dimensions,FloatType}

    if SimMetaData.FlagLog
        InitializeLogger(SimLogger,SimConstants,SimMetaData)
    end

    # If save directory is not already made, make it
    if !isdir(SimMetaData.SaveLocation)
        mkdir(SimMetaData.SaveLocation)
    end
    
    # Delete previous result files
    foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))
    # https://discourse.julialang.org/t/find-what-has-locked-held-a-file/23278
    GC.gc()
    try
        foreach(rm, filter(endswith(".vtkhdf"), readdir(SimMetaData.SaveLocation,join=true)))
    catch
    end

    # Unpack the relevant simulation meta data
    @unpack HourGlass, SimulationName, SilentOutput, ThreadsCPU = SimMetaData;

    # Load in particles
    SimParticles, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, Kernel, KernelGradient = AllocateDataStructures(Dimensions,FloatType, FluidCSV,BoundCSV)
    
    
    NumberOfPoints = length(SimParticles)::Int 
    @inline Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)

    @inline begin
        n_copy = Base.Threads.nthreads()
        KernelThreaded         = [copy(Kernel)         for _ in 1:n_copy]
        KernelGradientThreaded = [copy(KernelGradient) for _ in 1:n_copy]
        dρdtIThreaded          = [copy(dρdtI)          for _ in 1:n_copy]
        AccelerationThreaded   = [copy(KernelGradient) for _ in 1:n_copy]
    end

    # Produce sorting related variables
    ParticleRanges         = zeros(Int, NumberOfPoints + 1)
    UniqueCells            = zeros(CartesianIndex{Dimensions}, NumberOfPoints)
    Stencil                = ConstructStencil(Val(Dimensions))
    _, SortingScratchSpace = Base.Sort.make_scratch(nothing, eltype(SimParticles), NumberOfPoints)

    # Produce data saving functions
    SaveLocation_ = SimMetaData.SaveLocation * "/" * SimulationName
    SaveLocation  = (Iteration) -> SaveLocation_ * "_" * lpad(Iteration,6,"0") * ".vtkhdf"

    fid_vector    = Vector{HDF5.File}(undef, Int(SimMetaData.SimulationTime/SimMetaData.OutputEach + 1))

    SaveFile   = (Index) -> SaveVTKHDF(fid_vector, Index, SaveLocation(Index),to_3d(SimParticles.Position),["Kernel", "KernelGradient", "Density", "Pressure","Velocity", "Acceleration", "BoundaryBool" , "ID"], Kernel, KernelGradient, SimParticles.Density, SimParticles.Pressure, SimParticles.Velocity, SimParticles.Acceleration, Int.(SimParticles.BoundaryBool), SimParticles.ID)
    SimMetaData.OutputIterationCounter += 1
    @inline SaveFile(SimMetaData.OutputIterationCounter)
    

    InverseCutOff = Val(1/(SimConstants.H))

    # Normal run and save data
    generate_showvalues(Iteration, TotalTime, TimeLeftInSeconds) = () -> [(:(Iteration),format(FormatExpr("{1:d}"),  Iteration)), (:(TotalTime),format(FormatExpr("{1:3.3f}"), TotalTime)), (:(TimeLeftInSeconds),format(FormatExpr("{1:3.1f} [s]"), TimeLeftInSeconds))]
    
    # This is for some reason to trick the compiler to avoid dispatch error on SimulationLoop due to SimParticles
    # @inline on SimulationLoop directly slows down code
    f = () -> SimulationLoop(ComputeInteractions!, SimMetaData, SimConstants, SimParticles, Stencil, ParticleRanges, UniqueCells, SortingScratchSpace, Kernel, KernelThreaded, KernelGradient, KernelGradientThreaded, dρdtI, dρdtIThreaded, AccelerationThreaded, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, InverseCutOff)

    @inbounds while true

        @inline f()

        if SimMetaData.TotalTime >= SimMetaData.OutputEach * SimMetaData.OutputIterationCounter
            @timeit HourGlass "12A Output Data" SaveFile(SimMetaData.OutputIterationCounter + 1)

            if SimMetaData.FlagLog
                LogStep(SimLogger, SimMetaData, HourGlass)
                SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
            end

            SimMetaData.OutputIterationCounter += 1
        end

        if !SilentOutput
            TimeLeftInSeconds = (SimMetaData.SimulationTime - SimMetaData.TotalTime) * (TimerOutputs.tottime(HourGlass)/1e9 / SimMetaData.TotalTime)
            @timeit HourGlass "13 Next TimeStep" next!(SimMetaData.ProgressSpecification; showvalues = generate_showvalues(SimMetaData.Iteration , SimMetaData.TotalTime, TimeLeftInSeconds))
        end

        if SimMetaData.TotalTime > SimMetaData.SimulationTime
            
            if SimMetaData.FlagLog
                LogFinal(SimLogger, HourGlass)
                close(SimLogger.LoggerIo)
            end

            finish!(SimMetaData.ProgressSpecification)
            show(HourGlass,sortby=:name)
            show(HourGlass)

            @timeit HourGlass "12B Close hdfvtk output files"  close.(fid_vector)

            break
        end
    end
end

"""
    function viscosity_term(ρᵢ, ρⱼ, vᵢⱼ, xᵢⱼ, h, invd²η², m₀, α, c₀, ∇ᵢWᵢⱼ)

function that models the artificial viscosity term.

# Parameters:
- `ρᵢ`: Density of particle i.
- `ρⱼ`: Density of particle j.
- `vᵢⱼ`: relative velocity between particle i and j.
- `xᵢⱼ`: relative position between particle i and j.
- `h`: smoothing length.
- `invd²η²`: Muliplicative inverse of inv(xᵢⱼ²+η²) aka. 1/(xᵢⱼ²+η²)
- `m₀`: Mass of particle
- `α`: Artificial viscosity parameter. Default is 0.01.
- `c₀`: Speed of sound.
- ` ∇ᵢWᵢⱼ`: Kernel value of particle i.

"""
function viscosity_term(ρᵢ, ρⱼ, vᵢⱼ, xᵢⱼ, h, invd²η², m₀, α, c₀, ∇ᵢWᵢⱼ)
    # TODO add viscosity term
end

"""
    function diffusion_term(ρ₀, g, xᵢⱼ,Cb⁻¹,ρⱼ, ρᵢ, dᵢⱼ, η², h, c₀, m₀, ∇ᵢWᵢⱼ, δᵩ)

function that models the artificial diffusion term.

# Parameters:
- `ρ₀`: rest density.
- `g`: Gravitational constant (positive).
- `xᵢⱼ`: relative position between particle i and j.
- `Cb⁻¹`: Muliplicative inverse of pressure coefficient inv(c₀^2 * ρ₀)/γ) aka. 1/((c₀^2 * ρ₀)/γ
- `ρⱼ`: Density of particle j.
- `ρᵢ`: Density of particle i.
- `dᵢⱼ`: distance between partilce i and j.
- `η²`: eta, error term.
- `h`: smoothing length.
- `c₀`: Speed of sound.
- `m₀`: Mass of particle.
- ` ∇ᵢWᵢⱼ`: Kernel value of particle i.
- `δᵩ`: Coefficient for density diffusion.
"""
function diffusion_term(ρ₀, g, xᵢⱼ,Cb⁻¹,ρⱼ, ρᵢ, dᵢⱼ, η², h, c₀, m₀, ∇ᵢWᵢⱼ, δᵩ)
    # TODO add diffusion term
end

let
    Dimensions = 2
    FloatType  = Float64

    SimMetaDataDamBreak  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="Test", 
        SaveLocation="C:/Users/kirchejo/Repos/SPHExample/results/v0.5_other_symmetry_visco/",
        SimulationTime=1.2,
        OutputEach=0.01,
        FlagDensityDiffusion=false,
        FlagViscosityTreatment = :None,
        FlagOutputKernelValues=false,
        FlagLog=true
    )

    SimConstantsDamBreak = SimulationConstants{FloatType}(dx=0.02,c₀=88.14487860902641, δᵩ = 0.1, CFL=0.2, α = 0.02)

    SimLogger = SimulationLogger(SimMetaDataDamBreak.SaveLocation)

    RunSimulation(
        FluidCSV           = "./input/dam_break_2d/DamBreak2d_Dp0.02_Fluid_OneLayer.csv",
        BoundCSV           = "./input/dam_break_2d/DamBreak2d_Dp0.02_Bound_ThreeLayers.csv",
        SimMetaData        = SimMetaDataDamBreak,
        SimConstants       = SimConstantsDamBreak,
        SimLogger          = SimLogger
    )
end
 