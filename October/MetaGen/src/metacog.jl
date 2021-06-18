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
