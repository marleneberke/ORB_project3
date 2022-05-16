#writes one CSV file for how an ideal observer would update the v matrix
#another CSV file for the ground truth V

using Hungarian
using CSV
using DataFrames
using JSON
using Pipe: @pipe
using MetaGen
using Bootstrap

function gt_V(path::String)

	#could equally use input dictionary
	#dict = @pipe "output.json" |> open |> read |> String |> JSON.parse
	dict = @pipe (path * "output.json") |> open |> read |> String |> JSON.parse

	threshold = 0.0
	top_n = 5
	sigma = 200

	params = Video_Params(n_possible_objects = length(office_subset), sigma = sigma)
	receptive_fields = make_receptive_fields(params)
	objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, office_subset, num_videos, num_frames, threshold, top_n)

	###############################################################################

	alphas, betas = calc_gt_alpha_beta(dict, objects_observed, params)

	################################################################################
	#ground truth
	file = open(string(path * "../ground_truth_V.csv"), "w")

	print(file, "video_number&")
	for i = 1:params.n_possible_objects
		print(file, "gt_fa_", string(i), "&")
		print(file, "gt_m_", string(i), "&")
	end
	print(file, "\n")

	gt_v = fill(0.0, (params.n_possible_objects,2))
	gt_v[:,1] = alphas[:,1] ./ (betas[:,1]) #had been ((betas[:,1] .- 1).^2) in denominator
	gt_v[:,2] = alphas[:,2] ./ ((alphas[:,2]) .+ (betas[:,2]))
	println(gt_v)
	println("halls ", gt_v[:,1])
	println("misses ", gt_v[:,2])

	for v=1:num_videos
		print_helper(file, v, gt_v)
	end
	close(file)
end

function update_alpha_beta_gt_bb(alphas::Array{Int64,2}, betas::Array{Int64,2}, obs::Vector{Detection2D}, gt::Vector{Dict}, params::Video_Params)
	gt_categories = Vector{Int64}(undef, length(gt))
	for i in 1:length(gt)
		gt_categories[i] = gt[i]["label"]
	end
    obs_categories = last.(obs)
	detections = zeros(params.n_possible_objects)
    misses = zeros(params.n_possible_objects)
    halls = zeros(params.n_possible_objects)

    for category in 1:params.n_possible_objects
		gt_index = findall(gt_categories .== category)
        obs_index = findall(obs_categories .== category)

		#possibility of detection
        if !isempty(gt_index) && !isempty(obs_index)
			cost_matrix = calculate_matrix(gt[gt_index], obs[obs_index])
            assignment, cost = hungarian(cost_matrix)
            for i in 1:length(assignment)
				if assignment[i] != 0 #if the observation is assigned to a gt
					detected = is_in_bb(obs[obs_index][i], gt[gt_index][assignment[i]])
					detections[category] = detections[category] + detected
				end
			end
        end
		misses[category] = length(gt_index) - detections[category]
		halls[category] = length(obs_index) - detections[category]
    end

	#update hallucination stuff
	alphas[:, 1] = alphas[:, 1] .+ halls
	betas[:, 1] = betas[:, 1] .+ 1 #add one for each frame

	#update miss stuff
	alphas[:, 2] = alphas[:, 2] .+ misses
	betas[:, 2] = betas[:, 2] .+ detections
	return alphas, betas
end

#Set up the output file
function print_helper(file, v::Int64, avg_v::Matrix{Float64})
	print(file, v, "&")
	n_rows,_ = size(avg_v)
	for i = 1:n_rows
		print(file, avg_v[i,1], "&")
		print(file, avg_v[i,2], "&")
	end
	print(file, "\n")
end

function calc_gt_alpha_beta(dict::Array{Any}, objects_observed::Matrix{Array{Detection2D}}, params::Video_Params)
	#could make these zeros
	alphas = fill(0, (params.n_possible_objects,2))
	betas = fill(0, (params.n_possible_objects,2))
	avg_v = fill(0.0, (params.n_possible_objects,2))

	for v = 1:num_videos
		#println("v ", v)
		#println("gt_objects ", gt_objects[v])
		for f = 1:num_frames
			gt_objects_2D = get_gt_with_bb(dict, v, f)
			alphas, betas = update_alpha_beta_gt_bb(alphas, betas, objects_observed[v,f], gt_objects_2D, params)
		end
	end
	return alphas, betas
end
