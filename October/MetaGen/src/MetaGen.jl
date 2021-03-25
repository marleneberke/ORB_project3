module MetaGen

using Gen
using GenRFS
using Distributions

function __init__()
    @load_generated_functions
end

include("declaring_structs.jl")
include("geometry_optics.jl")
include("custom_distributions.jl")
include("receptive_fields.jl")
include("video.jl")
include("metacog.jl")

end # module
