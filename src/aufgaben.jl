function ausgabe()
    println("Hilfe")
end

function visco(ρᵢ, ρⱼ, vᵢⱼ, xᵢⱼ, invd²η², α, c₀, ∇ᵢWᵢⱼ, h, m₀)
    ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
    cond      = dot(vᵢⱼ, xᵢⱼ)
    cond_bool = cond < 0.0
    μᵢⱼ       = h*cond * invd²η²
    Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
    Πⱼ        = - Πᵢ
    return Πᵢ, Πⱼ
end

include("../example/Dambreak2d.jl")


let
    Dimensions = 2
    FloatType  = Float64

    SimMetaDataDamBreak  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="Test", 
        SaveLocation="C:/Users/kirchejo/Repos/SPHExample/results/v0.5_full_scps/",
        SimulationTime=2,
        OutputEach=0.01,
        FlagDensityDiffusion=true,
        FlagViscosityTreatment = :ArtificialViscosity,
        FlagOutputKernelValues=false,
        FlagLog=true
    )

    SimConstantsDamBreak = SimulationConstants{FloatType}(dx=0.02,c₀=88.14487860902641, δᵩ = 0.1, CFL=0.2, α = 0.02)

    SimLogger = SimulationLogger(SimMetaDataDamBreak.SaveLocation)
    ausgabe()
    RunSimulation(
        FluidCSV           = "./input/dam_break_2d/DamBreak2d_Dp0.02_Fluid_OneLayer.csv",
        BoundCSV           = "./input/dam_break_2d/DamBreak2d_Dp0.02_Bound_ThreeLayers.csv",
        SimMetaData        = SimMetaDataDamBreak,
        SimConstants       = SimConstantsDamBreak,
        SimLogger          = SimLogger
    )
end