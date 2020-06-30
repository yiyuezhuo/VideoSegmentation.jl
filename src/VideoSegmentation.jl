module VideoSegmentation

using Flux
using Flux: @adjoint, @functor

# using CUDA

using CuArrays # compatible with Flux.gpu
using CuArrays: @cuda
using CUDAnative
using CUDAdrv

# const CuArrayCompat = Union{CUDA.CuArray, CuArrays.CuArray}

#gpu(x) = CUDA.functional() ? fmap(CUDA.cu, x) : x
# borrow from Flux, we replace `CuArrays.cu` with `CUDA.cu`

include("utils.jl")
include("cuda_utils.jl")
include("flownet_s.jl")
include("flownet_sr.jl")
include("models.jl")
include("apply_flow.jl")

greet() = print("Hello World!")

end # module
