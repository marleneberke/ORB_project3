function within_frame(p::Detection2D)
    p[1] >= 0 && p[1] <= 256 && p[2] >= 0 && p[2] <= 256 #hard-codded frame size because I can't figure out how to use two arguments and filter
end

#check if p1 and p2 are withing radius r of each other. Euclidean space
function within_radius(p1::Detection2D, p2::Detection2D, r::Float64)
    sqrt((p1[1]-p2[1])^2 + (p1[2]-p2[2])^2) <= r
end

# #first approximation
# function update_alpha_beta(lesioned::Bool, alphas_old::Matrix{Int64}, betas_old::Matrix{Int64}, observations_2D, real_detections::Array{Detection2D}, params::Video_Params)
#     alphas = deepcopy(alphas_old)
#     betas = deepcopy(betas_old)
#
#     if lesioned
#         return alphas, betas
#     end
#
#     observations_2D = filter!(within_frame, observations_2D)
#     real_detections = filter!(within_frame, real_detections)
#
#     #println(real_detections)
#
#     #lets only do this by category
#     #real_detections_cats = last.(real_detections)
#     #observations_2D_cats = last.(observations_2D)
#
#     #see if actually detected. update miss rate
#     observations_2D_edited = deepcopy(observations_2D)
#     #observations_2D_cats_edited = copy(observations_2D_cats)
#     for i = 1:length(real_detections)
#         real_detection = real_detections[i]
#         cat = real_detection[3] #category
#         alphas[cat, 2] = alphas[cat, 2] + 1 #increase alpha for detection/miss rate
#         j = 1
#         while j <= length(observations_2D_edited) #basically a for loop over observations_2D_edited while it changes sizes
#             obs = observations_2D_edited[j]
#             if obs[3] == cat && within_radius(real_detection, obs, params.sigma)#if same category and within a distance of each other. 40 matches std on gaussian distribution for detection location
#                 observations_2D_edited = deleteat!(observations_2D_edited, j)
#                 betas[cat, 2] = betas[cat, 2] + 1 #increase beta for detection/miss rate
#                 #keep j the same because something was deleted at j
#             else
#                 j = j+1
#             end
#         end
#     end
#
#
#     #everything left in observations_2D_edited must have been hallucinated
#     for i in 1:length(observations_2D_edited)
#         alphas[observations_2D_edited[i][3], 1] = alphas[observations_2D_edited[i][3], 1] + 1
#     end
#
#     #update beta for hallucinations
#     betas[:, 1] = betas[:, 1] .+ 1 #add one for each frame
#     return alphas, betas
# end

################################################################################
"""
Returns the nearest ground-truth 2D object of the same category of the observation
as long as it's within a radius of sigma pixels.
If there's no match, returns NA.
"""
function find_nearest(obs_2D::Detection2D, gt_2D_objs::Array{Detection2D}, params)
    cat = obs_2D[3] #category
    gt_2D_objs_cat = filter(x -> x[3]==cat, gt_2D_objs) #filter to those matching category

    nearest = missing
    dist = Inf
    for i = 1:length(gt_2D_objs_cat)
        gt = gt_2D_objs_cat[i]
        d = sqrt((obs_2D[1] - gt[1])^2 + (obs_2D[2] - gt[2])^2) #Euclidean distance
        if d < dist
            nearest = gt
            dist = d
        end
    end

    if dist <= params.sigma
        return nearest
    else
        return missing
    end
end


#Match each detection to closest ground-truth object of that same category
function update_alpha_beta(lesioned::Bool, alphas_old::Matrix{Int64}, betas_old::Matrix{Int64}, observations_2D, real_detections::Array{Detection2D}, params::Video_Params)
    alphas = deepcopy(alphas_old)
    betas = deepcopy(betas_old)

    if lesioned
        return alphas, betas
    end

    observations_2D = filter!(within_frame, observations_2D)
    gt_2D_objs = filter!(within_frame, real_detections)

    #track how many times each ground-truth object gets detected
    num_times_detected = zeros(length(gt_2D_objs))
    for i = 1:length(observations_2D)
        obs_2D = observations_2D[i]
        cat = obs_2D[3] #category
        nearest_gt = find_nearest(obs_2D, gt_2D_objs, params)
        if !ismissing(nearest_gt) #matched detection with gt
            num_times_detected[findfirst(x->x==nearest_gt, gt_2D_objs)] += 1 #increment by 1
        else #each unmatched detection is a hallucination
            alphas[cat, 1] += 1
        end
    end

    for i = 1:length(gt_2D_objs)
        gt = gt_2D_objs[i]
        cat = gt[3] #category
        alphas[cat, 2] += 1 #always "missed" once, like detections stopping
        betas[cat, 2] += num_times_detected[i] #a detection
    end

    #update beta for hallucinations
    betas[:, 1] .+= 1 #add one for each frame
    return alphas, betas
end

################################################################################

# #Better version below uses Hungarian algorithm. Think there's a bug
# function cost_fxn(obs::Union{Any, Detection2D}, gt::Detection2D)
#     d = sqrt((obs[1] - gt[1])^2 + (obs[2] - gt[2])^2)
#     return minimum([1., (d / 362.)^2]) #don't want cost greater than 1 for a pairing
#     #362 is diagonal of a 256 by 256 image
# end
#
# function calculate_matrix(gt::Vector{Detection2D}, obs::Union{Vector{Any}, Vector{Detection2D}})
#     matrix = Matrix{Float64}(undef, length(obs), length(gt)) #obs are like workers, gt like tasks
#     for i = 1:length(obs)
#         for j = 1:length(gt)
#             matrix[i,j] = cost_fxn(obs[i], gt[j])
#         end
#     end
#     return matrix
# end
#
# #Better implementation using Hungarian algorithm for matching
# function update_alpha_beta(lesioned::Bool, alphas_old::Matrix{Int64}, betas_old::Matrix{Int64}, observations_2D, real_detections::Array{Detection2D}, params::Video_Params)
#     alphas = deepcopy(alphas_old)
#     betas = deepcopy(betas_old)
#
#     if lesioned
#         return alphas, betas
#     end
#
#     obs = filter!(within_frame, observations_2D)
#     gt = filter!(within_frame, real_detections)
#
#     obs_categories = last.(obs)
#     gt_categories = last.(gt)
#
#     detections = zeros(params.n_possible_objects)
#     misses = zeros(params.n_possible_objects)
#     halls = zeros(params.n_possible_objects)
#
#     for category in 1:params.n_possible_objects
# 		gt_index = findall(gt_categories .== category)
#         obs_index = findall(obs_categories .== category)
#
# 		#possibility of detection
#         if !isempty(gt_index) && !isempty(obs_index)
# 			cost_matrix = calculate_matrix(gt[gt_index], obs[obs_index])
#             assignment, cost = hungarian(cost_matrix)
#             for i in 1:length(assignment)
# 				if assignment[i] != 0 #if the observation is assigned to a gt
# 					detected = within_radius(obs[obs_index][i], gt[gt_index][assignment[i]], params.sigma)
# 					detections[category] = detections[category] + detected
# 				end
# 			end
#         end
# 		misses[category] = length(gt_index) - detections[category]
# 		halls[category] = length(obs_index) - detections[category]
#     end
#
# 	#update hallucination stuff
# 	alphas[:, 1] = alphas[:, 1] .+ halls
# 	betas[:, 1] = betas[:, 1] .+ 1 #add one for each frame
#
# 	#update miss stuff
# 	alphas[:, 2] = alphas[:, 2] .+ misses
# 	betas[:, 2] = betas[:, 2] .+ detections
#     return alphas, betas
# end

"""given a 3D detection, return BernoulliElement over a 2D detection"""
function render(params::Video_Params, camera_params::Camera_Params, object_3D::Object3D)
    cat = object_3D[4]
    object = Coordinate(object_3D[1], object_3D[2], object_3D[3])
    x, y = get_image_xy(camera_params, params, object)
    return (x, y, cat)
end

"""
    gen_camera(params::Video_Params)

Independently samples a camera location and camera focus from a
uniform distribution
"""
@gen (static) function gen_camera(params::Video_Params)
    #camera location
    camera_location_x = @trace(uniform(params.x_min,params.x_max), :camera_location_x)
    camera_location_y = @trace(uniform(params.y_min,params.y_max), :camera_location_y)
    camera_location_z = @trace(uniform(params.z_min,params.z_max), :camera_location_z)

    #camera focus focus
    camera_focus_x = @trace(uniform(params.x_min,params.x_max), :camera_focus_x)
    camera_focus_y = @trace(uniform(params.y_min,params.y_max), :camera_focus_y)
    camera_focus_z = @trace(uniform(params.z_min,params.z_max), :camera_focus_z)

    camera_params = Camera_Params(Coordinate(camera_location_x,camera_location_y,camera_location_z), Coordinate(camera_focus_x,camera_focus_y,camera_focus_z))
    return camera_params
end

# """
#     gen_camera(params::Video_Params)
#
# Places the camera at 0,0,0 and samples the camera focus from a Gaussian 5,0,0
# with sd (1, 2, 2)
# """
# @gen (static) function gen_camera_imagenet(params::Video_Params)
#     #camera location
#     camera_location_x = 0.00000001
#     camera_location_y = 0.000000001
#     camera_location_z = 0.00000000001
#
#     #camera focus focus
#     camera_focus_x = @trace(gaussian(5.0000000000001, 1.0), :camera_focus_x)
#     camera_focus_y = @trace(gaussian(0.00000000000001, 2.0), :camera_focus_y)
#     camera_focus_z = @trace(gaussian(0.00000000000001, 2.0), :camera_focus_z)
#
#     camera_params = Camera_Params(Coordinate(camera_location_x,camera_location_y,camera_location_z), Coordinate(camera_focus_x,camera_focus_y,camera_focus_z))
#     return camera_params
# end




"""
Generates the next frame given the current frame.

state is Tuple{Array{Any,1}, Matrix{Int64}, Matrix{Int64}}
"""
@gen (static) function frame_kernel(current_frame::Int64, state::Any, lesioned::Bool, params::Video_Params, v::Matrix{Real}, receptive_fields::Vector{Receptive_Field})
    ####Update 2D real objects

    ####Update camera location and pointing
    camera_params = @trace(gen_camera(params), :camera)

    #get locations of the objects in the image. basically want to input the list
    #of observations_3D [(x,y,z,cat), (x,y,z,cat)] and get out the [(x_image,y_image,cat)]
    scene = state[1]
    n_real_objects = length(scene)
    paramses = fill(params, n_real_objects)
    #vs = fill(v, n_real_objects)
    camera_paramses = fill(camera_params, n_real_objects)
    real_detections = map(render, paramses, camera_paramses, scene)
    real_detections = Array{Detection2D}(real_detections)
    #observations_2D will be what we condition on

    rfs_vec = get_rfs_vec(real_detections, params, v)

    #for loop over receptive fields
    #@show maximum(map(length, rfs_vec))
    #could re-write with map
    #@trace(Gen.Map(rfs)(rfs_vec), :observations_2D) #gets no method matching error
    observations_2D = @trace(rfs(rfs_vec), :observations_2D) #dirty shortcut because we only have one receptive field atm
    alphas, betas = update_alpha_beta(lesioned, state[2], state[3], observations_2D, real_detections, params)
    state = (scene, alphas, betas) #just keep sending the scene in.
    return state
end

frame_chain = Gen.Unfold(frame_kernel)

"""
Print
"""
function print_helper_alphas_betas(alphas::Matrix{Int64}, betas::Matrix{Int64})
    @show alphas
    @show betas
end

"""
Samples new values for lambda_fa.
"""
@gen (static) function update_lambda_fa(alpha::Int64, beta::Int64)
    #println("in update_lambda_fa")
    #println("alpha ", alpha)
    #println("beta ", beta)
    fa = @trace(gamma(alpha, 1/beta), :fa)
    #println("fa ", fa)
    return fa
end

"""
Samples new values for miss_rate.
"""
@gen (static) function update_miss_rate(alpha::Int64, beta::Int64)
    #println("in update_miss_rate")
    #println("alpha ", alpha)
    #println("beta ", beta)
    miss = @trace(beta(alpha, beta), :miss)
    #println("miss ", miss)
    return miss
end

"""
Samples a new v based on the previous v.
"""
@gen (static) function update_v_matrix(alphas::Matrix{Int64}, betas::Matrix{Int64})
    #v = Matrix{Real}(undef, dim(previous_v_matrix))
    #a = print_helper_alphas_betas(alphas, betas)
    fa = @trace(Map(update_lambda_fa)(alphas[:,1], betas[:,1]), :lambda_fa)
    miss = @trace(Map(update_miss_rate)(alphas[:,2], betas[:,2]), :miss_rate)
    #v[:, 1] = fa
    #v[:, 2] = miss
    v = hcat(fa, miss)
    return convert(Matrix{Real}, v)
    #return v
end

"""
Samples a new scene and a new v_matrix.
"""
@gen (static) function video_kernel(current_video::Int64, v_matrix_state::Any, lesioned::Bool, num_frames::Int64, params::Video_Params, receptive_fields::Array{Receptive_Field, 1})
    #for the scene. scenes are completely independent of each other
    #println("current video ", current_video)

    #rfs_element = GeometricElement{Object3D}(params.p_objects, object_distribution, (params,))
    #rfs_element = RFSElements{Object3D}([rfs_element]) #need brackets because rfs has to take an array
    init_scene = @trace(object_distribution_gaussian(params), :init_scene)
    #make the observations
    previous_v_matrix = v_matrix_state[1]
    previous_alphas = v_matrix_state[2]
    previous_betas = v_matrix_state[3]
    init_state = (init_scene, previous_alphas, previous_betas)

    state = @trace(frame_chain(num_frames, init_state, lesioned, params, previous_v_matrix, receptive_fields), :frame_chain)
    alphas = state[end][2]#not sure num_frames or end is better for index
    betas = state[end][3]
    #for the metacognition.
    # println("alphas ", alphas)
    # println("betas ", betas)
    v_matrix = @trace(update_v_matrix(alphas, betas), :v_matrix)
    v_matrix_state = (v_matrix, alphas, betas)
    return v_matrix_state
end

#video_chain = Gen.Unfold(video_kernel)
"""Creates scene chain"""
video_chain = Gen.Unfold(video_kernel)

"""Creates frame chain"""
frame_chain = Gen.Unfold(frame_kernel)

export video_chain
export render
#export within_frame
export update_alpha_beta
