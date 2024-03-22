module PreProcess

export LoadParticlesFromCSV, LoadParticlesFromCSV_StaticArrays, LoadBoundaryNormals

using CSV
using DataFrames
using StaticArrays
using ..SimulationEquations

# This function loads in a CSV file of particles. Please note that it is simply a Vector{SVector{3,Float64}}
# so you can definely make your own! It was just much easier to use a particle distribution layout generated by
# for example DualSPHysics, than making code for this example to do that
function LoadParticlesFromCSV(dims, float_type, fluid_csv,boundary_csv)
    DF_FLUID = CSV.read(fluid_csv, DataFrame)
    DF_BOUND = CSV.read(boundary_csv, DataFrame)

    P1F = DF_FLUID[!,"Points:0"]
    P2F = DF_FLUID[!,"Points:1"]
    P3F = DF_FLUID[!,"Points:2"]
    P1B = DF_BOUND[!,"Points:0"]
    P2B = DF_BOUND[!,"Points:1"]
    P3B = DF_BOUND[!,"Points:2"]

    points1          = Vector{float_type}()
    points2          = Vector{float_type}()
    points3          = Vector{float_type}()
    density_fluid    = Vector{float_type}()
    density_bound    = Vector{float_type}()

    # Since the particles are produced in DualSPHysics
    for i = 1:length(P1B)
        if dims == 3
            push!(points1,P1B[i])
            push!(points2,P2B[i])
            push!(points3,P3B[i])
        elseif dims == 2
            push!(points1,P1B[i])
            push!(points2,P3B[i])
        end
        push!(density_bound,DF_BOUND.Rhop[i])
    end
    

    for i = 1:length(P1F)
        if dims == 3
            push!(points1,P1F[i])
            push!(points2,P2F[i])
            push!(points3,P3F[i])
        elseif dims == 2
            push!(points1,P1F[i])
            push!(points2,P3F[i])
        end
        push!(density_fluid,DF_FLUID.Rhop[i])
    end

    if dims == 3
        points = DimensionalData(points1,points2,points3)
    elseif dims == 2
        points = DimensionalData(points1,points2)
    end

    return points,density_fluid,density_bound
end

function LoadParticlesFromCSV_StaticArrays(dims, float_type, fluid_csv, boundary_csv)
    DF_FLUID = CSV.read(fluid_csv, DataFrame)
    DF_BOUND = CSV.read(boundary_csv, DataFrame)

    P1F = DF_FLUID[!, "Points:0"]
    P2F = DF_FLUID[!, "Points:1"]
    P3F = DF_FLUID[!, "Points:2"] 
    P1B = DF_BOUND[!, "Points:0"]
    P2B = DF_BOUND[!, "Points:1"]
    P3B = DF_BOUND[!, "Points:2"]

    points = Vector{SVector{dims,float_type}}()
    density_fluid = Vector{float_type}()
    density_bound = Vector{float_type}()

    for (P1, P2, P3, DF, density) in [(P1B, P2B, P3B, DF_BOUND, density_bound), (P1F, P2F, P3F, DF_FLUID, density_fluid)]
        for i = 1:length(P1)
            point = dims == 3 ? SVector{dims,float_type}(P1[i], P2[i], P3[i]) : SVector{dims,float_type}(P1[i], P3[i])
            push!(points, point)
            push!(density, DF.Rhop[i])
        end
    end

    return points, density_fluid, density_bound
end


function LoadBoundaryNormals(dims, float_type, path_mdbc)
    # Read the CSV file into a DataFrame
    df = CSV.read(path_mdbc, DataFrame)

    normals       = Vector{SVector{dims,float_type}}()
    points        = Vector{SVector{dims,float_type}}()
    ghost_points  = Vector{SVector{dims,float_type}}()

    # Loop over each row of the DataFrame
    for i in 1:size(df, 1)
        # Extract the "Normal" fields into an SVector
        if dims == 3
            normal = SVector{dims,float_type}(df[i, "Normal:0"], df[i, "Normal:1"], df[i, "Normal:2"])
            point  = SVector{dims,float_type}(df[i, "Points:0"], df[i, "Points:1"], df[i, "Points:2"])
        elseif dims == 2
            normal = SVector{dims,float_type}(df[i, "Normal:0"], df[i, "Normal:2"])
            point  = SVector{dims,float_type}(df[i, "Points:0"], df[i, "Points:2"])
        end

        push!(normals, normal)
        push!(points,  point)
        push!(ghost_points,  point+normal)

    end

    return points, ghost_points, normals
end

end

