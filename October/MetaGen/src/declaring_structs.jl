Base.@kwdef struct Coordinate
    x::Float64
    y::Float64
    z::Float64
end

Base.@kwdef struct Video_Params
    lambda_objects::Float64 = 10.0
    possible_objects::Vector{Int64} = collect(1:91)
    probs_possible_objects = collect(ones(91)./91)
    v::Matrix{Float64} = zeros(91, 2)
    x_min::Float64 = -10
    y_min::Float64 = -10
    z_min::Float64 = -10
    x_max::Float64 = 10
    y_max::Float64 = 10
    z_max::Float64 = 10
    num_receptive_fields::Int64 = 1
    #camera parameters that don't change
    image_dim_x::Int64 = 320
    image_dim_y::Int64 = 240
    horizontal_FoV::Float64 = 60
    vertical_FoV::Float64 = 40
end

#The camera params that change in a video
Base.@kwdef struct Camera_Params
    camera_location::Coordinate
    camera_focus::Coordinate
end

Base.@kwdef struct Receptive_Field
    p1::Tuple{Int64, Int64} #upper left
    p2::Tuple{Int64, Int64} #lower right
end

export Coordinate
export Camera_Params
export Permanent_Camera_Params
export Receptive_Field
