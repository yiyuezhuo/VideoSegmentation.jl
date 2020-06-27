module VideoSegmentation

using Flux
using Flux: @adjoint
using CUDA

include("utils.jl")
include("cuda_utils.jl")
include("flownet_s.jl")
include("flownet_sr.jl")
include("models.jl")
include("apply_flow.jl")

greet() = print("Hello World!")

end # module
