module SPHCellList

export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!, ComputeInteractions!

using Parameters, FastPow, StaticArrays, Base.Threads, ChunkSplitters
import LinearAlgebra: dot

using ..SimulationEquations
using ..AuxillaryFunctions

    """
        function ConstructStencil(v::Val{d}) where d

    Construction of the stencil of the kernel shape

    # Parameters
    - `SimLogger`: Logger that handles the output of the logger file.
    - `SimMetaData`: Simulation Meta data
    - `HourGlass`: Keeps track of execution time.

    # Example
    ```julia
    Stencil                = ConstructStencil(Val(Dimensions))
    ```
    """
    function ConstructStencil(v::Val{d}) where d
        n_ = CartesianIndices(ntuple(_->-1:1,v))
        half_length = length(n_) ÷ 2
        n  = n_[1:half_length]

        return n
    end

    """
        function ExtractCells!(Particles, ::Val{InverseCutOff}) where InverseCutOff

    Construction of the grid containing possible neigbors

    # Parameters
    - `Particles`: particles in the simulation.
    - `::Val{InverseCutOff}`: Cutoff distance of kernel.

    # Example
    ```julia
    ExtractCells!(Particles, CutOff)
    ```
    """
    @inline function ExtractCells!(Particles, ::Val{InverseCutOff}) where InverseCutOff
        # Replace unsafe_trunc with trunc if this ever errors
        function map_floor(x)
            unsafe_trunc(Int, muladd(x,InverseCutOff,2))
        end

        Cells  = @views Particles.Cells
        Points = @views Particles.Position
        @threads for i ∈ eachindex(Particles)
            t = map(map_floor, Tuple(Points[i]))
            Cells[i] = CartesianIndex(t)
        end
        return nothing
    end

    """
        function UpdateNeighbors!(Particles, CutOff, SortingScratchSpace, ParticleRanges, UniqueCells)

    Update the neigborhood of the particles

    # Parameters
    - `Particles`: particles in the simulation.
    - `CutOff`: Cutoff distance of kernel.
    - `SortingScratchSpace`: Sorting algorithm.
    - `ParticleRanges`: range of possible particles in neighborhood
    - `UniqueCells`: Identify closeby cells.

    # Example
    ```julia
    IndexCounter = UpdateNeighbors!(SimParticles, InverseCutOff, SortingScratchSpace,  ParticleRanges, UniqueCells)
    ```
    """
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

    """
        function NeighborLoop!(ComputeInteractions!, SimMetaData, SimConstants, ParticleRanges, Stencil, Position, Kernel, KernelGradient, Density, Pressure, Velocity, dρdtI, dvdtI,  MotionLimiter, UniqueCells, IndexCounter)

            Function to process each cell and its neighbors

    # Parameters
    - `ComputeInteractions!`: Function to process interactions.
    - `SimMetaData`: Simulation Meta data.
    - `SimConstants`: Simulation constants.
    - `ParticleRanges`: range of possible particles in neighborhood
    - `Stencil`: Kernel stencil.
    - `Position`: Particle postions.
    - `Kernel`: Kernel values.
    - `KernelGradient`: Kernel gradient values.
    - `Density`: Particle densities.
    - `Pressure`: Particle pressures.
    - `Velocity`: Particle Velocities.
    - `dρdtI`: Density derivative values.
    - `dvdtI`: Velocity derivative values.
    - `MotionLimiter`: Identifies Boundary and fluid particles.
    - `UniqueCells`: Identify closeby cells.
    - `IndexCounter`: List of possible neigbors.

    # Example
    ```julia
    NeighborLoop!(ComputeInteractions!, SimMetaData, SimConstants, ParticleRanges, Stencil, Position, KernelThreaded, KernelGradientThreaded, Density, Pressure, Velocity, dρdtIThreaded, AccelerationThreaded,  MotionLimiter, UniqueCells, IndexCounter)
    ```
    """
    function NeighborLoop!(ComputeInteractions!, SimMetaData, SimConstants, ParticleRanges, Stencil, Position, Kernel, KernelGradient, Density, Pressure, Velocity, dρdtI, dvdtI,  MotionLimiter, UniqueCells, IndexCounter)
        UniqueCells = view(UniqueCells, 1:IndexCounter)
        @threads for (ichunk, inds) in enumerate(chunks(UniqueCells; n=nthreads()))
            for iter in inds
                CellIndex = UniqueCells[iter]

                StartIndex = ParticleRanges[iter] 
                EndIndex   = ParticleRanges[iter+1] - 1

                @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex
                    @inline ComputeInteractions!(SimMetaData, SimConstants, Position, Kernel, KernelGradient, Density, Pressure, Velocity, dρdtI, dvdtI, i, j, MotionLimiter, ichunk)
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
                            @inline ComputeInteractions!(SimMetaData, SimConstants, Position, Kernel, KernelGradient, Density, Pressure, Velocity, dρdtI, dvdtI, i, j, MotionLimiter, ichunk)
                        end
                    end
                end
            end
        end

        return nothing
    end

end
