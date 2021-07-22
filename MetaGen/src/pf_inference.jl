"""
    unfold_particle_filter(num_particles::Int, objects_observed::Matrix{Array{Detection2D}}, camera_trajectories::Matrix{Camera_Params}, num_receptive_fields::Int64)

Performs inference procedure.

# Arguments
- num_particles::Int
- objects_observed: indexed by ``(\\mathrm{scene, frame, receptive field, detection})``
- camera_trajectories: Indexed by ``(\\mathrm{scene, frame})``
- num_receptive_fields::Int64

array for each frame and array for each receptive field and array for those detections

# Returns
- A trace containing the inferred visual system and world state.


"""
function unfold_particle_filter(lesioned::Bool, num_particles::Int, objects_observed::Matrix{Array{Detection2D}}, camera_trajectories::Matrix{Camera_Params}, params::Video_Params, file)
    init_obs = Gen.choicemap()

    if lesioned
        for j = 1:params.n_possible_objects
            init_obs[:init_v_matrix => :lambda_fa => j => :fa] = 1.0
            init_obs[:init_v_matrix => :miss_rate => j => :miss] = 0.5
        end
    end

    #no video, no frames
    println("initialize pf")
    state = Gen.initialize_particle_filter(main, (0, 0, params), init_obs, num_particles)

    num_videos, num_frames = size(objects_observed)

    for v=1:num_videos

        println("v ", v)

        maybe_resample!(state, ess_threshold=num_particles)#used to be /2. now always resampling becasue I want to get rid of -Inf before they become NANs
        ess = effective_sample_size(normalize_weights(state.log_weights)[2])
        println("ess after resample ", ess)


        #want to do this for every frame at once. put a : instead of f
        #rf is for receptive field
        obs = Gen.choicemap()

        for f = 1:num_frames
            camera_params = camera_trajectories[v, f]
            obs[:videos => v => :frame_chain => f => :camera => :camera_location_x] = camera_params.camera_location.x
            obs[:videos => v => :frame_chain => f => :camera => :camera_location_y] = camera_params.camera_location.y
            obs[:videos => v => :frame_chain => f => :camera => :camera_location_z] = camera_params.camera_location.z
            obs[:videos => v => :frame_chain => f => :camera => :camera_focus_x] = camera_params.camera_focus.x
            obs[:videos => v => :frame_chain => f => :camera => :camera_focus_y] = camera_params.camera_focus.y
            obs[:videos => v => :frame_chain => f => :camera => :camera_focus_z] = camera_params.camera_focus.z

            #for rf = 1:num_receptive_fields
                #println("objects_observed[v, f][rf] ", objects_observed[v, f][rf])
                #println("type ", typeof(objects_observed[v, f][rf]))
            obs[:videos => v => :frame_chain => f => :observations_2D] = convert(Array{Any, 1}, objects_observed[v, f])
            #end
        end
        #def should be using map to replace for loops here
        #point is, condition on the camera trajectory and on the observations

        if lesioned
            for j = 1:params.n_possible_objects
                obs[:videos => v => :v_matrix => :lambda_fa => j => :fa] = 1.0
                obs[:videos => v => :v_matrix => :miss_rate => j => :miss] = 0.5
            end
        end

        Gen.particle_filter_step!(state, (v, num_frames, params), (UnknownChange(),), obs)

        ess = effective_sample_size(normalize_weights(state.log_weights)[2])
        println("ess after pf step ", ess)

        #optional rejuvination
        (perturb_params, n_objects_per_category) = get_probs_categories(objects_observed, params, v, num_frames)
        line_segments_per_category = get_line_segments_per_category(params, objects_observed, camera_trajectories, v, num_frames)
        #line_segments = get_line_segments(objects_observed, camera_trajectories, params, v, num_frames, num_receptive_fields, total_n_objects)
        #to-do: threads here
        Threads.@threads for i = 1:num_particles
            println("perturb particle i ", i)
            state.traces[i] = perturb(lesioned, state.traces[i], v, perturb_params, line_segments_per_category)
            println("done perturbing i ", i)
            println("trace ", state.traces[i][:videos => v => :init_scene])
            println("log score of this trace ", get_score(state.traces[i]))
            #visualize_trace(state.traces, i, camera_trajectories, v, 1, params)

            #visualize_trace(state.traces, i, camera_trajectories, v, 2, params)
        end

        #visualize_trace_with_heatmap(state.traces, camera_trajectories, v, 1, params )

        min_particle_weight = minimum(map(i -> get_score(state.traces[i]), collect(1:num_particles)))
        println("min log score ", min_particle_weight)

        max_particle_weight = maximum(map(i -> get_score(state.traces[i]), collect(1:num_particles)))
        println("max log score ", max_particle_weight)

        ess = effective_sample_size(normalize_weights(state.log_weights)[2])
        println("ess after rejuvination ", ess)

        if lesioned && v==num_videos
            print_Vs_and_Rs_to_file(file, state.traces, num_particles, params, v, true)
        else
            print_Vs_and_Rs_to_file(file, state.traces, num_particles, params, v)
        end
    end

    return Gen.sample_unweighted_traces(state, num_particles)

end

"""
    perturb(trace, v::Int64, perturb_params::Perturb_Params, line_segments_per_category::Array{Array{Line_Segment,1},1})
Does 500 MCMC steps (with different proposal functions) on the scene and on the v matrix.

"""

@gen function perturb(lesioned::Bool, trace, v::Int64, perturb_params::Perturb_Params, line_segments_per_category::Array{Array{Line_Segment,1},1})
    #acceptance_counter = 0
    #proposal_counter = 0

    println("before perturbations")
    println("v ", v)

    println("alpha ", get_retval(trace)[end][2][5,2])
    println("beta ", get_retval(trace)[end][3][5,2])

    #println("miss 2 ", trace[:init_v_matrix => :miss_rate => 2 => :miss])
    println("init miss 5 ", trace[:init_v_matrix => :miss_rate => 5 => :miss])
    #println("miss 2 ", trace[:videos => v => :v_matrix => :miss_rate => 2 => :miss])
    println("miss rate 5 ", trace[:videos => v => :v_matrix => :miss_rate => 5 => :miss])

    println("scene at v ", trace[:videos => v => :init_scene])

    for iter=1:5 #try 100 MH moves
        println("iter ", iter)
        #println("trace ", trace[:videos => v => :init_scene])

        trace = perturb_scene(trace, v, perturb_params, line_segments_per_category)
        if lesioned == false
            for iter2=1:10
                trace = perturb_v_matrix_mh(trace, v, perturb_params)
            end
        end
        # println("lambda_fa 2 ", trace[:v_matrix => (:lambda_fa, 2)])
        # println("miss 2 ", trace[:v_matrix => (:miss_rate, 2)])
        # println("lambda_fa 5 ", trace[:v_matrix => (:lambda_fa, 5)])
        # println("miss 5 ", trace[:v_matrix => (:miss_rate, 5)])
    end
    #println("acceptance_counter $(acceptance_counter/proposal_counter)")

    println("after perturbations")
    println("v ", v)

    println("alpha ", get_retval(trace)[end][2][5,2])
    println("beta ", get_retval(trace)[end][3][5,2])

    #println("miss 2 ", trace[:init_v_matrix => :miss_rate => 2 => :miss])
    println("init miss 5 ", trace[:init_v_matrix => :miss_rate => 5 => :miss])
    #println("miss 2 ", trace[:videos => v => :v_matrix => :miss_rate => 2 => :miss])
    println("miss rate 5 ", trace[:videos => v => :v_matrix => :miss_rate => 5 => :miss])

    println("scene at v ", trace[:videos => v => :init_scene])
    return trace
end

"""
    perturb_scene(trace, v::Int64, perturb_params::Perturb_Params, line_segments_per_category::Array{Array{Line_Segment,1},1})
Does 3 MCMC steps (with different proposal functions) on the scene.

"""

function perturb_scene(trace, v::Int64, perturb_params::Perturb_Params, line_segments_per_category::Array{Array{Line_Segment,1},1})
    trace, accepted = add_remove_kernel(trace, v, line_segments_per_category, perturb_params)
    #println("accepted? ", accepted)
    #println("trace ", trace[:videos => v => :init_scene])

    #only try changing location or category if there's at least one object in the scene
    if length(trace[:videos => v => :init_scene]) > 0
        trace, accepted = change_location_kernel(trace, v, 0.1, perturb_params)
        #println("accepted? ", accepted)
        #println("trace ", trace[:videos => v => :init_scene])

        trace, accepted = change_category_kernel(trace, v, perturb_params)
        #println("accepted? ", accepted)
        #println("trace ", trace[:videos => v => :init_scene])
    end
    return trace
end

"""
    perturb_v_matrix(trace, perturb_params::Perturb_Params)
Picks one element of the v matrix to perturb using metropolis hastings.

"""

#just pick an element of the matrix to perturb
function perturb_v_matrix_mh(trace, v::Int64, perturb_params::Perturb_Params)
    n = length(perturb_params.probs_possible_objects)
    i = categorical([0.5, 0.5])
    j = categorical(fill(1/n, n))

    selection = get_selection(v, j, i)
    trace, accepted = mh(trace, selection) #proposes new trace from the prior


    # if i == 1 #if perturbing fa
    #     trace, accepted = mh(trace, proposal_for_v_matrix_fa, (v, j))
    # else
    #     trace, accepted = mh(trace, proposal_for_v_matrix_miss, (v, j))
    # end

    #println("accepted? ", accepted)
    #println("trace ", trace[:v_matrix])
    return trace
end


function get_selection(v::Int64, j::Int64, i::Int64)
    if i == 1 #change lambda_fa
        if v == 1
            selection = select(:videos => v => :v_matrix => :lambda_fa => j => :fa,
            :init_v_matrix => :lambda_fa => j => :fa)
        else
            selection = select(:videos => v => :v_matrix => :lambda_fa => j => :fa,
            :videos => v-1 => :v_matrix => :lambda_fa => j => :fa)
        end
    else
        if v == 1
            selection = select(:videos => v => :v_matrix => :miss_rate => j => :miss,
            :init_v_matrix => :miss_rate => j => :miss)
        else
            selection = select(:videos => v => :v_matrix => :miss_rate => j => :miss,
            :videos => v-1 => :v_matrix => :miss_rate => j => :miss)
        end
    end
    return selection
end

"""
    perturb_v_matrix_hmc(trace, perturb_params::Perturb_Params)
Perturbs an element of the v_matrix using Hamiltonian MC

"""

#just pick an element of the matrix to perturb
function perturb_v_matrix_hmc(trace, v::Int64, perturb_params::Perturb_Params)
    n = length(perturb_params.probs_possible_objects)
    i = categorical([0.5, 0.5])
    j = categorical(fill(1/n, n))

    if i == 1 #if perturbing fa
        trace, accepted = hmc(trace, select(:videos => v => :v_matrix => :lambda_fa => j => :fa))
    else
        trace, accepted = hmc(trace, select(:videos => v => :v_matrix => :miss_rate => j => :miss))
    end

    #println("accepted? ", accepted)
    #println("trace ", trace[:v_matrix])

    return trace
end


"""
    proposal_for_v_matrix_fa(trace, j::Int64)
Performs on MH step on the false alarm rate for object of category j.

"""

@gen function proposal_for_v_matrix_fa(trace, v::Int64, j::Int64)
    std = 0.01
    choices = get_choices(trace)
    #centered on previous value
    @trace(trunc_normal(choices[:videos => v => :v_matrix => :lambda_fa => j => :fa], std, 0.0, 10000.0), :videos => v => :v_matrix => :lambda_fa => j => :fa) #had to make trunc_normal because of error otherwise
end

"""
    proposal_for_v_matrix_miss(trace, j::Int64)
Performs on MH step on the miss rate for object of category j.

"""

@gen function proposal_for_v_matrix_miss(trace, v::Int64, j::Int64)
    std = 0.1
    choices = get_choices(trace)
    #centered on previous value
    @trace(trunc_normal(choices[:videos => v => :v_matrix => :miss_rate => j => :miss], std, 0.0, 1.0), :videos => v => :v_matrix => :miss_rate => j => :miss)
end

"""
    get_probs_categories(objects_observed::Matrix{Array{Detection2D}}, params::Video_Params, v::Int64, num_frames::Int64, num_receptive_fields::Int64)

Returns a probability distribution over the object categories to be used
in the proposal functions.

Derived from the objects that the visual system observes.
"""

function get_probs_categories(objects_observed::Matrix{Array{Detection2D}}, params::Video_Params, v::Int64, num_frames::Int64)
    track_categories = zeros(length(params.possible_objects)) #each element will be the number of times that category was detected. adding 1
    probabilities = zeros(length(params.possible_objects))
    for f = 1:num_frames
        #for rf = 1:num_receptive_fields
        for (index, value) in enumerate(objects_observed[v, f])
            track_categories[value[3]] = track_categories[value[3]]+1#category
        end
        #end
    end
    #make it so that categories that were never observed collectively have 10% of the weight
    other_ten_percent = sum(track_categories)/9
    each = other_ten_percent/sum(track_categories.==0)
    probabilities = copy(track_categories)
    probabilities[track_categories.==0] .= each

    #in case of no observations, make uniform
    if sum(probabilities) == 0
        probabilities .= 1
    end

    return (Perturb_Params(probs_possible_objects = (probabilities)./sum(probabilities)), track_categories)
end

"""
    get_line_segments_per_category(params::Video_Params, objects_observed::Matrix{Array{Detection2D}}, camera_trajectories::Matrix{Camera_Params}, v::Int64, num_frames::Int64, num_receptive_fields::Int64)

Gets all line segments in 3d space that correspond to each 2D detection
in every frame of a scene.
"""
function get_line_segments_per_category(params::Video_Params, objects_observed::Matrix{Array{Detection2D}}, camera_trajectories::Matrix{Camera_Params}, v::Int64, num_frames::Int64)
    line_segments = Array{Array{Line_Segment, 1}}(undef, length(params.possible_objects))
    for j = 1:length(params.possible_objects)
        line_segments[j] = []
    end
    for f = 1:num_frames
        camera_params = camera_trajectories[v, f]
        #for rf = 1:num_receptive_fields
        for (index, value) in enumerate(objects_observed[v, f])
            #println("value ", value)
            line_segment = get_line_segment(camera_params, params, value)
            #println("line_segment in pf_inference ", line_segment)
            push!(line_segments[value[3]], line_segment)
        end
        #end
    end
    return line_segments
end

"""Duplicated from Gen library"""
function effective_sample_size(log_normalized_weights::Vector{Float64})
    log_ess = -logsumexp(2. * log_normalized_weights)
    return exp(log_ess)
end

"""Duplicated from Gen Library"""
function normalize_weights(log_weights::Vector{Float64})
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (log_total_weight, log_normalized_weights)
end

export unfold_particle_filter
