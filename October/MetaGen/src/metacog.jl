#make the matrix v
@gen function make_visual_system(params::Video_Params)
	v = Matrix{Float64}(undef, length(params.possible_objects), 2)
	#alpha = 2
	#beta = 10
	#alpha_hit = 10
	#beta_hit = 2
	for j = 1:length(params.possible_objects)
		#set lambda when target absent
		#v[j,1] = @trace(Gen.beta(alpha, beta), (:fa, j)) #leads to fa rate of around 0.1
		v[j,1] = @trace(trunc_normal(0.002, 0.005, 0.0, 1.0), (:lambda_fa, j)) #these are lambdas per receptive field
		#set miss rate when target present
		v[j,2] = @trace(trunc_normal(0.25, 0.5, 0.0, 1.0), (:miss_rate, j))
	end
	return v
end
################################################################################


@gen (static) function metacog(num_videos::Int64, num_frames::Int64)

	params = Video_Params()

    #sample parameters
    #set up visual system's parameters
    #Determining visual system V


	n = length(params.possible_objects)
	v = Matrix{Float64}(undef, n, 2)

	#v[:,1] = @trace(Map(trunc_normal(0.002, 0.005, 0.0, 1.0), collect(1:n)), :lambda_fa)
	#v[:,2] = @trace(Map(trunc_normal(0.25, 0.5, 0.0, 1.0), collect(1:n)), :miss_rate)

	#temp = @trace(Map(foo)(collect(1:n)), :lambda_fa)

	v = @trace(make_visual_system(params), :v_matrix)

	receptive_fields = make_receptive_fields()

	params = Video_Params(v = v, num_receptive_fields = length(receptive_fields))

    fs = fill(num_frames, num_videos) #number of frames per video
    ps = fill(params, num_videos)
	receptive_fieldses = fill(receptive_fields, num_videos)

    @trace(video_map(fs, ps, receptive_fieldses), :videos)

end

export metacog

################################################################################
#set up receptive_field. make sure this matches the one in MetaGen metacog
function make_receptive_fields()

    #square receptive fields. hardcoded for the 240 x 320 image
    pixels = 80

	#layer 1 of receptive fields. 3x4, inside the image
    n_horizontal = 4
    n_vertical = 3
    n = n_horizontal*n_vertical
    receptive_fields_layer_1 = Vector{Receptive_Field}(undef, n) #of length n
    for h = 1:n_horizontal
        for v = 1:n_vertical
            receptive_fields_layer_1[n_vertical*(h-1)+v] = Receptive_Field(p1 = ((h-1)*pixels, (v-1)*pixels), p2 = (h*pixels, v*pixels))
        end
    end

	#layer 2 of receptive fields. 4x5 overlay
	n_horizontal = 5
    n_vertical = 4
    n = n_horizontal*n_vertical
    receptive_fields_layer_2 = Vector{Receptive_Field}(undef, n) #of length n
    for h = 1:n_horizontal
        for v = 1:n_vertical
            receptive_fields_layer_2[n_vertical*(h-1)+v] = Receptive_Field(p1 = ((h-1.5)*pixels, (v-1.5)*pixels), p2 = ((h-0.5)*pixels, (v-0.5)*pixels))
        end
    end

	#layer 3 of receptive fields. 3x5 overlay, tiled horizontally
	n_horizontal = 5
    n_vertical = 3
    n = n_horizontal*n_vertical
    receptive_fields_layer_3 = Vector{Receptive_Field}(undef, n) #of length n
    for h = 1:n_horizontal
        for v = 1:n_vertical
            receptive_fields_layer_3[n_vertical*(h-1)+v] = Receptive_Field(p1 = ((h-1.5)*pixels, (v-1)*pixels), p2 = ((h-0.5)*pixels, v*pixels))
        end
    end

	#layer 4 of receptive fields. 4x4 overlay, tiled vertically
	n_horizontal = 4
    n_vertical = 4
    n = n_horizontal*n_vertical
    receptive_fields_layer_4 = Vector{Receptive_Field}(undef, n) #of length n
    for h = 1:n_horizontal
        for v = 1:n_vertical
            receptive_fields_layer_4[n_vertical*(h-1)+v] = Receptive_Field(p1 = ((h-1)*pixels, (v-1.5)*pixels), p2 = (h*pixels, (v-0.5)*pixels))
        end
    end
	
	receptive_fields = vcat(receptive_fields_layer_1, receptive_fields_layer_2, receptive_fields_layer_3, receptive_fields_layer_4)

    return receptive_fields
end

export make_receptive_fields
