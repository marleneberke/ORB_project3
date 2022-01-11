
################################################################################
function threeD2twoD(threeDobjects::Vector{Any}, v::Int64, f::Int64,
		params::Video_Params, camera_trajectories::Matrix{Camera_Params})
	paramses = fill(params, length(threeDobjects))
	camera_paramses = fill(camera_trajectories[v, f], length(threeDobjects))
	threeDobjects_2D = map(render, paramses, camera_paramses, threeDobjects)
	threeDobjects_2D = Array{Detection2D}(threeDobjects_2D)
	threeDobjects_2D = filter(p -> within_frame(p), threeDobjects_2D)
	#categories = last.(threeDobjects_2D)
end

################################################################################
function threeD2twoD(threeDobjects::Vector{Tuple{Float64, Float64, Float64, Int64}}, v::Int64, f::Int64,
		params::Video_Params, camera_trajectories::Matrix{Camera_Params})
	paramses = fill(params, length(threeDobjects))
	camera_paramses = fill(camera_trajectories[v, f], length(threeDobjects))
	threeDobjects_2D = map(render, paramses, camera_paramses, threeDobjects)
	threeDobjects_2D = Array{Detection2D}(threeDobjects_2D)
	threeDobjects_2D = filter(p -> within_frame(p), threeDobjects_2D)
	#categories = last.(threeDobjects_2D)
end

function cost_fxn(obs::Detection2D, gt::Detection2D)
    d = sqrt((obs[1] - gt[1])^2 + (obs[2] - gt[2])^2)
    return minimum([1., (d / 362.)^2]) #don't want cost greater than 1 for a pairing
    #362 is diagonal of a 256 by 256 image
end

function calculate_matrix(gt::Vector{Detection2D}, obs::Vector{Detection2D})
    matrix = Matrix{Float64}(undef, length(obs), length(gt)) #obs are like workers, gt like tasks
    for i = 1:length(obs)
        for j = 1:length(gt)
            matrix[i,j] = cost_fxn(obs[i], gt[j])
        end
    end
    return matrix
end

#for just one frame, calculate similarity between gt and obs
function similarity(gt::Vector{Detection2D}, obs::Vector{Detection2D})
    gt_categories = last.(gt)
    obs_categories = last.(obs)
    #J = jaccard_similarity(gt_categories, obs_categories)
    union_val = 0
    weighted_intersection = 0

    for category in 1:params.n_possible_objects
        gt_index = findall(gt_categories .== category)
        obs_index = findall(obs_categories .== category)

        n_matches = 0
        if !isempty(gt_index) && !isempty(obs_index)
            cost_matrix = calculate_matrix(gt[gt_index], obs[obs_index])
            #println(cost_matrix)
            #println(size(cost_matrix))
            assignment, cost = hungarian(cost_matrix)

            n_matches = sum(assignment!=0) #how many things got matched up
            weight = n_matches - cost

            weighted_intersection = weighted_intersection + weight
        end
        union_val = union_val + length(obs_index) + length(gt_index) - n_matches #subtract n_matches to avoid double counting
    end

    if union_val == 0 && weighted_intersection == 0
        return 1.
    else
        return weighted_intersection/union_val
    end
end
################################################################################
#this one is for getting sim for the NN
function similarity_2D(num_videos::Int64, num_frames::Int64, params::Video_Params, dict::Any,
    ground_truth_world_states::Vector{Any}, threshold::Float64, top_n = Inf)

    receptive_fields = make_receptive_fields(params)

    objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, office_subset, num_videos, num_frames, threshold, top_n)

    sim_NN = zeros(num_videos)
    for v = 1:num_videos
        for f = 1:num_frames
            gt = threeD2twoD(ground_truth_world_states[v], v, f, params, camera_trajectories)
            NN = objects_observed[v,f]
            sim_NN[v] = sim_NN[v] + similarity(gt, NN)
        end
        sim_NN[v] = sim_NN[v]/num_frames
    end
    return sim_NN
end


################################################################################
#this one is for getting sim and confidence interval for a DataFrame
function similarity_2D(num_videos::Int64, num_frames::Int64, params::Video_Params, dict::Any,
    ground_truth_world_states::Vector{Any}, df::DataFrame, num_bootstraps::Int64, cil::Float64)

    receptive_fields = make_receptive_fields(params)

    objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, office_subset, num_videos, num_frames, threshold)

    sim = zeros(num_videos)
    for v = 1:num_videos
        for f = 1:num_frames
			gt = threeD2twoD(ground_truth_world_states[v], v, f, params, camera_trajectories)
			model = threeD2twoD(df[v, "inferred_best_world_state"], v, f, params, camera_trajectories)
            sim[v] = sim[v] + similarity(gt, model)
        end
        sim[v] = sim[v]/num_frames
    end

	#let's make confidence intervals
	from = findfirst(names(df).=="world_state_1")
	to = length(names(df))
	data = select(df, from:to)#now we have just the columns we need
	num_particles = convert(Int64, size(data)[2]/2)
	lower_ci = Array{Float64}(undef, num_videos)
	upper_ci = Array{Float64}(undef, num_videos)
	sim_values = zeros(num_videos, num_particles)

	#bootstrapping happens for each video
	for v = 1:num_videos
		weights = Array{Float64}(undef, num_particles)
		for j = 1:num_particles
			str = "weight_" * string(j)
			weights[j] = df[v, str]
			for f = 1:num_frames
				gt = threeD2twoD(ground_truth_world_states[v], v, f, params, camera_trajectories)
				str = "world_state_" * string(j)
				model = threeD2twoD(df[v, str], v, f, params, camera_trajectories)
				sim_values[v, j] = sim_values[v, j] + similarity(gt, model)
			end
			sim_values[v,j] = sim_values[v,j]/num_frames
		end
		new_df = DataFrame(weights = weights, values = sim_values[v,:])
		#println("new_df ", new_df)
		bs = bootstrap(highest_weighted_world_state, new_df, BasicSampling(num_bootstraps))
		#println("bs ", bs)
		bci = confint(bs, PercentileConfInt(cil))
		#println("bci ", bci)
		lower_ci[v] = bci[1][2]
		upper_ci[v] = bci[1][3]
	end
    return sim, lower_ci, upper_ci
end
