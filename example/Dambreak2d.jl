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

# Really important to overload default function, gives 10x speed up?
# Overload the default function to do what you please
# TODO Per particle/per Interaction
function ComputeInteractions!(SimMetaData, SimConstants, Position, KernelThreaded, KernelGradientThreaded, Density, Pressure, Velocity, dρdtI, dvdtI, i, j, MotionLimiter, ichunk)
    @unpack FlagViscosityTreatment, FlagDensityDiffusion, FlagOutputKernelValues = SimMetaData
    @unpack ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, ν₀, dx = SimConstants

    xᵢⱼ  = Position[i] - Position[j]
    xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)              
    if  xᵢⱼ² <= H²
        #https://discourse.julialang.org/t/sqrt-abs-x-is-even-faster-than-sqrt/58154/2
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
            # Dᵢ, Dⱼ = diff(ρ₀, g, -xᵢⱼ, Cb⁻¹, ρᵢ, ρⱼ, dᵢⱼ, η², δᵩ, h, c₀, m₀, ∇ᵢWᵢⱼ) # TODO what does MotionLimiter do
            Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
            ρᵢⱼᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
            Pⱼᵢᴴ  = -Pᵢⱼᴴ
            ρⱼᵢᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pⱼᵢᴴ, Cb⁻¹)

            ρⱼᵢ   = ρⱼ - ρᵢ

            Ψᵢⱼ   = 2( ρⱼᵢ  - ρᵢⱼᴴ) * (-xᵢⱼ)/(dᵢⱼ^2 + η²)
            Ψⱼᵢ   = 2(-ρⱼᵢ  - ρⱼᵢᴴ) * ( xᵢⱼ)/(dᵢⱼ^2 + η²) 

            MLcond = MotionLimiter[i] * MotionLimiter[j]
            Dᵢ    =  δᵩ * h * c₀ * (m₀/ρⱼ) * dot(Ψᵢⱼ ,  ∇ᵢWᵢⱼ) * MLcond
            Dⱼ    =  δᵩ * h * c₀ * (m₀/ρᵢ) * dot(Ψⱼᵢ , -∇ᵢWᵢⱼ) * MLcond
            # Dᵢ *= MLcond
            # Dⱼ *= MLcond
            
        else
            Dᵢ  = 0.0
            Dⱼ  = 0.0
        end
        dρdtI[ichunk][i] += dρdt⁺ + Dᵢ
        dρdtI[ichunk][j] += dρdt⁻ + Dⱼ


        Pᵢ      =  Pressure[i]
        Pⱼ      =  Pressure[j]
        Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
        dvdt⁺   = - m₀ * Pfac *  ∇ᵢWᵢⱼ
        dvdt⁻   = - dvdt⁺

        if FlagViscosityTreatment == :ArtificialViscosity
            # Πᵢ,Πⱼ = visco(ρᵢ, ρⱼ, vᵢⱼ, xᵢⱼ, invd²η², α, c₀, ∇ᵢWᵢⱼ, h, m₀)
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

# Reduce threaded arrays
@inline function reduce_sum!(target_array, arrays)
    for array in arrays
        target_array .+= array
    end
end
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

    # if mod(SimMetaData.Iteration,10) == 0
        @timeit SimMetaData.HourGlass "02 Calculate IndexCounter" IndexCounter = UpdateNeighbors!(SimParticles, InverseCutOff, SortingScratchSpace,  ParticleRanges, UniqueCells)
    # else
        # IndexCounter = findfirst(isequal(0), ParticleRanges) - 2
    # end

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

###===
function RunSimulation(;FluidCSV::String,
    BoundCSV::String,
    SimMetaData::SimulationMetaData{Dimensions, FloatType},
    SimConstants::SimulationConstants,
    SimLogger::SimulationLogger
    ) where {Dimensions,FloatType}

     # If save directory is not already made, make it
     if !isdir(SimMetaData.SaveLocation)
        mkdir(SimMetaData.SaveLocation)
    end

    if SimMetaData.FlagLog
        InitializeLogger(SimLogger,SimConstants,SimMetaData)
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
    
    
    NumberOfPoints = length(SimParticles)::Int #Have to type declare, else error?
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
    SimMetaData.OutputIterationCounter += 1 #Since a file has been saved
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

            # This should not be counted in actual run 
            @timeit HourGlass "12B Close hdfvtk output files"  close.(fid_vector)

            break
        end
    end
end

