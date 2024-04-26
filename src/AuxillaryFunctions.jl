module AuxillaryFunctions
using StaticArrays

export ResetArrays!, to_3d

"""
    function ResetArrays!(arrays...)

Initialize all given Arrays with zeros

# Parameters
- `arrays...`: Arrays that should be reset   
"""
ResetArrays!(arrays...) = foreach(a -> fill!(a, zero(eltype(a))), arrays)

"""
    function to_3d(vec_2d)

Turns 2d vector into a 3d vector by adding 0.0 in the third dimension

# Parameters
- `vec_2d`: 2d vector   
"""
to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]

end