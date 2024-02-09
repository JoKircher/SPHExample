using Revise

using SPHExample
using CSV
using DataFrames
using Printf
using StaticArrays
using CellListMap
using LinearAlgebra
using TimerOutputs
using Parameters
import ProgressMeter: Progress, next!
using Formatting
using StructArrays



"""
    RunSimulation(;SimulationMetaData::SimulationMetaData, SimulationConstants::SimulationConstants)

Run a Smoothed Particle Hydrodynamics (SPH) simulation using specified metadata and simulation constants.

This function initializes the simulation environment, loads particle data, and runs the simulation iteratively until the maximum number of iterations is reached. It outputs simulation results at specified intervals.

## Arguments
- `SimulationMetaData::SimulationMetaData`: A struct containing metadata for the simulation, including the simulation name, save location, maximum iterations, output iteration frequency, and other settings.
- `SimulationConstants::SimulationConstants`: A struct containing constants used in the simulation, such as reference density, initial particle distance, smoothing length, initial mass, normalization constant for kernel, artificial viscosity alpha value, gravity, speed of sound, gamma for the pressure equation of state, initial time step, coefficient for density diffusion, and CFL number.

## Variable Explanation
- `FLUID_CSV`: Path to CSV file containing fluid particles. See "input" folder for examples.
- `BOUND_CSV`: Path to CSV file containing boundary particles. See "input" folder for examples.
- `ρ₀`: Reference density.
- `dx`: Initial particle distance. See "dp" in CSV files. For 3D simulations, a typical value might be 0.0085.
- `H`: Smoothing length.
- `m₀`: Initial mass, calculated as reference density multiplied by initial particle distance to the power of simulation dimensions.
- `mᵢ = mⱼ = m₀`: All particles have the same mass.
- `αD`: Normalization constant for the kernel.
- `α`: Artificial viscosity alpha value.
- `g`: Gravity (positive value).
- `c₀`: Speed of sound, which must be 10 times the highest velocity in the simulation.
- `γ`: Gamma, most commonly 7 for water, used in the pressure equation of state.
- `dt`: Initial time step.
- `δᵩ`: Coefficient for density diffusion, typically 0.1.
- `CFL`: CFL number for the simulation.

## Example
```julia
#See SimulationMetaData and SimulationConstants for all possible inputs
SimMetaData  = SimulationMetaData(SimulationName="MySimulation", SaveLocation=raw"path/to/results", MaxIterations=101)
SimConstants = SimulationConstants{SimMetaData.FloatType, SimMetaData.IntType}()
RunSimulation(
    FluidCSV = "./input/FluidPoints_Dp0.02.csv",
    BoundCSV = "./input/BoundaryPoints_Dp0.02.csv",
    SimulationMetaData = SimMetaData,
    SimulationConstants = SimConstants
)
```
"""
function RunSimulation(;FluidCSV::String,
                        BoundCSV::String,
                        SimMetaData::SimulationMetaData{FloatType},
                        SimConstants::SimulationConstants,
) where FloatType
    # Unpack the relevant simulation meta data
    @unpack HourGlass, SaveLocation, SimulationName, MaxIterations, OutputIteration, SilentOutput, ThreadsCPU = SimMetaData;

    # Unpack simulation constants
    @unpack ρ₀, dx, h, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η² = SimConstants

    # Load in the fluid and boundary particles. Return these points and both data frames
    points, density_fluid, density_bound  = LoadParticlesFromCSV(FloatType, FluidCSV,BoundCSV)

    # Generate simulation data results array
    FinalResults = SimulationDataResults{3,FloatType}(NumberOfParticles = length(points))
    @unpack Kernel, KernelGradient, Density, Position, Acceleration, Velocity = FinalResults
    # Initialize Arrays
    #Position .= deepcopy(points)
    Density  .= deepcopy([density_fluid;density_bound])

    GravityContribution = SVector(0.0,g,0.0)

    # Read this as "GravityFactor * g", so -1 means negative acceleration for fluid particles
    # 1 means boundary particles push back against gravity
    GravityFactor = [-ones(size(density_fluid,1)) ; ones(size(density_bound,1))]
    GravityContributionArray = map((x)->x * GravityContribution,GravityFactor) 

    # MotionLimiter is what allows fluid particles to move, while not letting the velocity of boundary
    # particles change
    MotionLimiter = [ ones(size(density_fluid,1)) ; zeros(size(density_bound,1))]

    # Based on MotionLimiter we assess which particles are boundary particles
    BoundaryBool  = .!Bool.(MotionLimiter)

    # Save the initial particle layout with dummy values
    create_vtp_file(SimMetaData,SimConstants,FinalResults)

    # Preallocate simulation arrays
    SizeOfParticlesI1 = size(Density)
    SizeOfParticlesI3 = size(Position)
    TypeOfParticleI3  = eltype(Position)

    dρdtI             = zeros(FloatType,         SizeOfParticlesI1)
    dvdtI             = zeros(TypeOfParticleI3,  SizeOfParticlesI3)
  
    ρₙ⁺               = zeros(FloatType,         SizeOfParticlesI1)
    vₙ⁺               = zeros(TypeOfParticleI3,  SizeOfParticlesI3)

    Positionₙ⁺ˣ         = zeros(FloatType,  SizeOfParticlesI1)
    Positionₙ⁺ʸ         = zeros(FloatType,  SizeOfParticlesI1)
    Positionₙ⁺ᶻ         = zeros(FloatType,  SizeOfParticlesI1)
    Positionₙ⁺          = StructArray{TypeOfParticleI3}(( Positionₙ⁺ˣ, Positionₙ⁺ʸ, Positionₙ⁺ᶻ))
  
    dρdtIₙ⁺           = zeros(FloatType,         SizeOfParticlesI1)
  
    
    xᵢⱼˣ               = zeros(FloatType,  SizeOfParticlesI1)
    xᵢⱼʸ               = zeros(FloatType,  SizeOfParticlesI1)
    xᵢⱼᶻ               = zeros(FloatType,  SizeOfParticlesI1)
    xᵢⱼ                = StructArray{TypeOfParticleI3}(( xᵢⱼˣ, xᵢⱼʸ, xᵢⱼᶻ))

    Positionˣ         = getindex.(points,1)
    Positionʸ         = getindex.(points,2)
    Positionᶻ         = getindex.(points,3)
    Position          = StructArray{TypeOfParticleI3}(( Positionˣ, Positionʸ, Positionᶻ))

    I                 = zeros(Int64,   SizeOfParticlesI1)
    J                 = zeros(Int64,   SizeOfParticlesI1)
    D                 = zeros(Float64, SizeOfParticlesI1)
    list_me           = StructArray{Tuple{Int64,Int64,Float64}}((I,J,D))

    KernelGradientL   = zeros(TypeOfParticleI3,  SizeOfParticlesI3)
    drhopLp           = zeros(FloatType,         SizeOfParticlesI1)
    drhopLn           = zeros(FloatType,         SizeOfParticlesI1) 
         
    Pressureᵢ         = zeros(FloatType,         SizeOfParticlesI1)

    # Initialize the system system.nb.list
    system  = InPlaceNeighborList(x=Position, cutoff=2*h, parallel=true)

    # Define Progress spec
    show_vals(x) = [(:(Iteration),format(FormatExpr("{1:d}"), x.Iteration)), (:(TotalTime),format(FormatExpr("{1:3.3f}"),x.TotalTime))]

    @inbounds for SimMetaData.Iteration = 1:MaxIterations
        # Be sure to update and retrieve the updated neighbour system.nb.list at each time step
        @timeit HourGlass "0 | Update Neighbour system.nb.list" begin
            update!(system,Position)
            neighborlist!(system)
            resize!(list_me, system.nb.n)
            list_me .= system.nb.list
        end
        
        @timeit HourGlass "0 | Reset arrays to zero and resize L arrays" begin
            # Clean up arrays, Vector{T} and Vector{SVector{3,T}} must be cleansed individually,
            # to avoid run time dispatch errors
            ResetArrays!(Kernel, dρdtI,dρdtIₙ⁺,KernelGradient,dvdtI, Acceleration)
            # Resize KernelGradientL based on length of neighborsystem.nb.list
            ResizeBuffers!(KernelGradientL, xᵢⱼ, drhopLp, drhopLn; N = system.nb.n)
        end

        @timeit HourGlass "1 | Update xᵢⱼ, kernel values and kernel gradient" begin
            updatexᵢⱼ!(xᵢⱼˣ, xᵢⱼʸ, xᵢⱼᶻ, I, J,  Positionˣ, Positionʸ, Positionᶻ)
            # Here we output the kernel value for each particle
            ∑ⱼWᵢⱼ!(Kernel, system.nb.list, SimConstants)
            # Here we output the kernel gradient value for each particle and also the kernel gradient value
            # based on the pair-to-pair interaction system.nb.list, for use in later calculations.
            # Other functions follow a similar format, with the "I" and "L" ending
            ∑ⱼ∇ᵢWᵢⱼ!(KernelGradient, KernelGradientL, system.nb.list, xᵢⱼ, SimConstants)
        end

        # Then we calculate the density derivative at time step "n"
        @timeit HourGlass "2| DDT" ∂ρᵢ∂tDDT!(dρdtI,system.nb.list,xᵢⱼ,xᵢⱼʸ,Density,Velocity,KernelGradientL,MotionLimiter,drhopLp,drhopLn, SimConstants)

        # We calculate viscosity contribution and momentum equation at time step "n"
        @timeit HourGlass "2| Pressure" map!(x -> Pressure(x, c₀, γ, ρ₀), Pressureᵢ, Density)
        @timeit HourGlass "2| ∂vᵢ∂t!"   ∂vᵢ∂t!(dvdtI, system.nb.list, Density, KernelGradientL,Pressureᵢ, SimConstants)
        @timeit HourGlass "2| ∂Πᵢⱼ∂t!"  ∂Πᵢⱼ∂t!(dvdtI, system.nb.list, xᵢⱼ,Density,Velocity,KernelGradientL, SimConstants)
        @timeit HourGlass "2| Gravity"  dvdtI   .+=    GravityContributionArray

        # Based on the density derivative at "n", we calculate "n+½"
        @timeit HourGlass "2| ρₙ⁺" @. ρₙ⁺  = Density  + dρdtI * (dt/2)
        # We make sure to limit the density of boundary particles in such a way that they cannot produce suction
        @timeit HourGlass "2| LimitDensityAtBoundary!(ρₙ⁺)" LimitDensityAtBoundary!(ρₙ⁺,BoundaryBool,ρ₀)

        # We now calculate velocity and position at "n+½"
        @timeit HourGlass "2| vₙ⁺" @. vₙ⁺          = Velocity   + dvdtI * (dt/2) * MotionLimiter
        @timeit HourGlass "2| Positionₙ⁺" @. Positionₙ⁺   = Position   + vₙ⁺ * (dt/2)   * MotionLimiter
        @timeit HourGlass "2| updatexᵢⱼ!" updatexᵢⱼ!(xᵢⱼˣ, xᵢⱼʸ, xᵢⱼᶻ, I, J, Positionₙ⁺ˣ, Positionₙ⁺ʸ, Positionₙ⁺ᶻ)

        # Density derivative at "n+½" - Note that we keep the kernel gradient values calculated at "n" for simplicity
        @timeit HourGlass "2| DDT2" ∂ρᵢ∂tDDT!(dρdtIₙ⁺,system.nb.list,xᵢⱼ,xᵢⱼʸ,ρₙ⁺,vₙ⁺,KernelGradientL,MotionLimiter, drhopLp, drhopLn, SimConstants)

        # Viscous contribution and momentum equation at "n+½"
        @timeit HourGlass "2| Pressure2" map!(x -> Pressure(x, c₀, γ, ρ₀), Pressureᵢ, ρₙ⁺)
        @timeit HourGlass "2| ∂vᵢ∂t!2" ∂vᵢ∂t!(Acceleration, system.nb.list, ρₙ⁺, KernelGradientL, Pressureᵢ, SimConstants) 
        @timeit HourGlass "2| ∂Πᵢⱼ∂t!2" ∂Πᵢⱼ∂t!(Acceleration,system.nb.list, xᵢⱼ ,ρₙ⁺,vₙ⁺, KernelGradientL, SimConstants)
        @timeit HourGlass "2| Acceleration2" Acceleration .+= GravityContributionArray

        # Factor for properly time stepping the density to "n+1" - We use the symplectic scheme as done in DualSPHysics
        @timeit HourGlass "2| DensityEpsi!" DensityEpsi!(Density,dρdtIₙ⁺,ρₙ⁺,dt)

        # Clamp boundary particles minimum density to avoid suction
        @timeit HourGlass "2| LimitDensityAtBoundary!(Density)" LimitDensityAtBoundary!(Density,BoundaryBool,ρ₀)

        # Update Velocity in-place and then use the updated value for Position
        @timeit HourGlass "2| Velocity" @. Velocity += Acceleration * dt * MotionLimiter
        @timeit HourGlass "2| Position" @. Position += ((Velocity + (Velocity - Acceleration * dt * MotionLimiter)) / 2) * dt * MotionLimiter

        # Automatic time stepping control
        @timeit HourGlass "3| Calculating time step" begin
            dt =  Δt(FinalResults,SimConstants)
            SimMetaData.CurrentTimeStep = dt
            SimMetaData.TotalTime      += dt
        end
        
        @timeit HourGlass "4| OutputVTP" OutputVTP(SimMetaData,SimConstants,FinalResults)

        next!(SimMetaData.ProgressSpecification; showvalues = show_vals(SimMetaData))
    end

    
    # Print the timings in the default way
    show(HourGlass,sortby=:name)
    show(HourGlass)
    disable_timer!(HourGlass)
end


# Initialize SimulationMetaData
begin
    T = Float64
    SimMetaData  = SimulationMetaData{T}(
                                    SimulationName="MySimulation", 
                                    SaveLocation=raw"E:\SecondApproach\Results", 
                                    MaxIterations=10001
    )
    # Initialze the constants to use
    SimConstants = SimulationConstants{T}()
    # Clean up folder before running (remember to make folder before hand!)
    foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))

    # println(
    #     # And here we run the function - enjoy!
    #     @report_opt target_modules=(@__MODULE__,) RunSimulation(
    #         FluidCSV     = "./input/FluidPoints_Dp0.02.csv",
    #         BoundCSV     = "./input/BoundaryPoints_Dp0.02.csv",
    #         SimMetaData  = SimMetaData,
    #         SimConstants = SimConstants
    #     )
    # )

    # And here we run the function - enjoy!
    @profview RunSimulation(
        FluidCSV     = "./input/FluidPoints_Dp0.02.csv",
        BoundCSV     = "./input/BoundaryPoints_Dp0.02.csv",
        SimMetaData  = SimMetaData,
        SimConstants = SimConstants
    )
end
