function visco(ρᵢ, ρⱼ, vᵢⱼ, xᵢⱼ, invd²η², α, c₀, ∇ᵢWᵢⱼ, h, m₀)
    ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
    cond      = dot(vᵢⱼ, xᵢⱼ)
    cond_bool = cond < 0.0
    μᵢⱼ       = h*cond * invd²η²
    Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
    Πⱼ        = - Πᵢ
    return Πᵢ, Πⱼ
end

function diffusions(ρ₀, g, xᵢⱼ, Cb⁻¹, ρᵢ, ρⱼ, dᵢⱼ, η², δᵩ, h, c₀, m₀, ∇ᵢWᵢⱼ)
    Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
    ρᵢⱼᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
    Pⱼᵢᴴ  = -Pᵢⱼᴴ
    ρⱼᵢᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pⱼᵢᴴ, Cb⁻¹)

    ρⱼᵢ   = ρⱼ - ρᵢ

    Ψᵢⱼ   = 2( ρⱼᵢ  - ρᵢⱼᴴ) * (-xᵢⱼ)/(dᵢⱼ^2 + η²)
    Ψⱼᵢ   = 2(-ρⱼᵢ  - ρⱼᵢᴴ) * ( xᵢⱼ)/(dᵢⱼ^2 + η²) 

    Dᵢ    =  δᵩ * h * c₀ * (m₀/ρⱼ) * dot(Ψᵢⱼ ,  ∇ᵢWᵢⱼ)
    Dⱼ    =  δᵩ * h * c₀ * (m₀/ρᵢ) * dot(Ψⱼᵢ , -∇ᵢWᵢⱼ)

    return Dᵢ, Dⱼ
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
        FlagViscosityTreatment = true,
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