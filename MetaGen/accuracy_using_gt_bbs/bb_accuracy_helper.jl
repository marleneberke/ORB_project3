using Hungarian
using DataFrames

COCO_CLASSES = ["person", "bicycle", "car", "motorcycle",
			"airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
			"N/A", "stop sign","parking meter", "bench", "bird", "cat", "dog", "horse",
			"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
			"umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
			"sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
			"surfboard", "tennis racket","bottle", "N/A", "wine glass", "cup", "fork", "knife",
			"spoon","bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
			"hot dog", "pizza","donut", "cake", "chair", "couch", "potted plant", "bed",
			"N/A", "dining table","N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
			"remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
			"book","clock", "vase", "scissors", "teddy bear", "hair drier",
			"toothbrush"]
Dict_gt_to_office = Dict("chair" => "chair", "bowl" => "bowl", "plant" => "potted plant", "tv" => "tv", "umbrella" => "umbrella")

#helper function for getting ground-truth labels and bounding box for a frame
#Returns a list of tuples, one tuple for each thing in the ground-truth of the frame
function get_gt_with_bb(dict::Array{Any}, v::Int64, f::Int64)
	gt = Dict[]
	for i = 1:length(dict[v]["views"][f]["labels2D"])
		cat_name = dict[v]["views"][f]["labels2D"][i]["category_name"]
		office_name = get(Dict_gt_to_office, cat_name, "NA") #office name will be NA if it's not an entry in the Dict
		label = findfirst(office_subset .== office_name)
		top_left = dict[v]["views"][f]["labels2D"][i]["top_left"]
		bottom_right = dict[v]["views"][f]["labels2D"][i]["bottom_right"]
		center = dict[v]["views"][f]["labels2D"][i]["center"]
		if !isnothing(label)
			push!(gt, Dict("top_left" => top_left, "bottom_right" => bottom_right, "center" => center, "label" => label))
		end
	end
	return gt
end

################################################################################

function within_frame(x::Float64, y::Float64) #1280, 720 for imagenet
    x >= 0 && x <= 256 && y >= 0 && y <= 256 #hard-codded frame size
end

function within_frame(p::Detection2D)
    p[1] >= 0 && p[1] <= 256 && p[2] >= 0 && p[2] <= 256 #hard-codded frame size
end

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

################################################################################
function is_in_bb(obs::Detection2D, gt::Dict)
	in_x = (obs[1] <= gt["bottom_right"][1]) && (obs[1] >= gt["top_left"][1])
	in_y = (obs[2] <= gt["bottom_right"][2]) && (obs[2] >= gt["top_left"][2])
	return in_x && in_y
end
################################################################################

function cost_fxn(obs::Detection2D, gt::Dict)
    d = sqrt((obs[1] - gt["center"][1])^2 + (obs[2] - gt["center"][2])^2)
    return minimum([1., (d / 362.)^2]) #don't want cost greater than 1 for a pairing
    #362 is diagonal of a 256 by 256 image
end

function calculate_matrix(gt::Vector{Dict}, obs::Vector{Detection2D})
    matrix = Matrix{Float64}(undef, length(obs), length(gt)) #obs are like workers, gt like tasks
    for i = 1:length(obs)
        for j = 1:length(gt)
            matrix[i,j] = cost_fxn(obs[i], gt[j])
        end
    end
    return matrix
end

#for just one frame, calculate similarity between gt and obs
function similarity(gt::Vector{Dict}, obs::Vector{Detection2D})
	gt_categories = Vector{Int64}(undef, length(gt))
	for i in 1:length(gt)
		gt_categories[i] = gt[i]["label"]
	end
    obs_categories = last.(obs)
    #J = jaccard_similarity(gt_categories, obs_categories)
    union_val = zeros(params.n_possible_objects) #keep track of these separately per category. then add together at the end
    num_matches = zeros(params.n_possible_objects)

    for category in 1:params.n_possible_objects
		gt_index = findall(gt_categories .== category)
        obs_index = findall(obs_categories .== category)

        if !isempty(gt_index) && !isempty(obs_index)
			cost_matrix = calculate_matrix(gt[gt_index], obs[obs_index])
            assignment, cost = hungarian(cost_matrix)

            for i in 1:length(assignment)
				if assignment[i] != 0 #if the observation is assigned to a gt
					num_matches[category] = num_matches[category] + is_in_bb(obs[obs_index][i], gt[gt_index][assignment[i]])
				end
			end
        end
        union_val[category] = length(obs_index) + length(gt_index) - num_matches[category] #subtract num_matches to avoid double counting
    end

	if sum(union_val) == 0 && sum(num_matches) != 0
		println("dividing by 0")
	end

    if sum(union_val) == 0 && sum(num_matches) == 0
        return 1.
    else
        return sum(num_matches)/sum(union_val)
    end
end
################################################################################
#this one is for getting sim for the NN
function similarity_2D(num_videos::Int64, num_frames::Int64, params::Video_Params,
    dict_with_gt::Vector{Any}, threshold::Float64, top_n = Inf)

    receptive_fields = make_receptive_fields(params)

    objects_observed, camera_trajectories = make_observations_office(dict_with_gt, receptive_fields, office_subset, num_videos, num_frames, threshold, top_n)
	#objects_observed, camera_trajectories = make_observations(dict, receptive_fields, num_videos, num_frames, threshold, top_n)

    sim_NN = zeros(num_videos)
    for v = 1:num_videos
        for f = 1:num_frames
            gt = get_gt_with_bb(dict_with_gt, v, f)
            NN = objects_observed[v,f]
            sim_NN[v] = sim_NN[v] + similarity(gt, NN)
        end
        sim_NN[v] = sim_NN[v]/num_frames
    end
    return sim_NN
end


################################################################################
#this one is for getting sim and confidence interval for a DataFrame
function similarity_2D(num_videos::Int64, num_frames::Int64, params::Video_Params,
    dict_with_gt::Vector{Any}, df::DataFrame)

    receptive_fields = make_receptive_fields(params)

    _, camera_trajectories = make_observations_office(dict_with_gt, receptive_fields, office_subset, num_videos, num_frames, threshold)

    sim = zeros(num_videos)
    for v = 1:num_videos
        for f = 1:num_frames
			gt = get_gt_with_bb(dict_with_gt, v, f)
			model = threeD2twoD(df[v, "inferred_best_world_state"], v, f, params, camera_trajectories)
            sim[v] = sim[v] + similarity(gt, model)
        end
        sim[v] = sim[v]/num_frames
    end
    return sim
end

################################################################################
#parse the dataframe and return a vector of world states
function new_parse_data(data::DataFrame, num_videos::Int64)
    world_states = Array{Any}(undef, num_videos) #length(online_names) should be the same as num_videos
    for i = 1:num_videos
        world_states[i] = eval(Meta.parse(data[i, "inferred_best_world_state"])) #1 is for the first row. we only have 1 row
    end
    return world_states
end

################################################################################
#parse the dataframe and return a dataframe of world states
function new_parse_data(data::DataFrame, num_videos::Int64, num_particles::Int64)
	println("num_videos ", num_videos)
	world_states = Array{Any}(undef, num_videos) #length(online_names) should be the same as num_videos
	for i = 1:num_videos
        world_states[i] = eval(Meta.parse(data[i, "inferred_best_world_state"])) #1 is for the first row. we only have 1 row
    end
	data[!, "inferred_best_world_state"] = world_states

	for j = 1:num_particles
		temp = Array{Any}(undef, num_videos) #length(online_names) should be the same as num_videos
		str = "world_state_" * string(j)
		for i = 1:num_videos
	        temp[i] = eval(Meta.parse(data[i, str])) #1 is for the first row. we only have 1 row
	    end
		data[!, str] = temp
	end

    return data
end
