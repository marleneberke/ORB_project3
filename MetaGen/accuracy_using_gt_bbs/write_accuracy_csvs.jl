#script for pre-processing data. Ends with giving Jacard similarity between
#ground-truth world states and the mode inferred world state, and 95% CI on that.
#detections inside the bounding box are coded as 1, and outside are 0.

using Hungarian
using CSV
using DataFrames
using JSON
using Pipe: @pipe
using MetaGen
using Bootstrap

function write_accuracy_csvs(full_path::String, lesion_path::String, top_n::Int64, num_videos::Int64, num_training_videos::Int64, num_frames::Int64)

    online_data = CSV.read(full_path * "online_ws.csv", DataFrame; delim = "&")
    retro_data = CSV.read(full_path * "retro_ws.csv", DataFrame; delim = "&")
    lesioned_data = CSV.read(lesion_path * "retro_ws.csv", DataFrame; delim = "&")

    ################################################################################
    #could equally use input or output dictionary
    #dict = @pipe path * "../data_labelled_detr.json" |> open |> read |> String |> JSON.parse

    dict_with_gt = @pipe full_path * "output.json" |> open |> read |> String |> JSON.parse

    online_data = sort!(online_data, :video_number) #undoes whatever shuffle happened
    retro_data = sort!(retro_data, :video_number)

    #ground_truth_world_states = get_ground_truth(dict, num_videos)
    ################################################################################
    online_world_states = new_parse_data(online_data, num_training_videos, num_particles)
    retrospective_world_states = new_parse_data(retro_data, num_videos, num_particles)
    lesioned_world_states = new_parse_data(lesioned_data, num_videos, num_particles)

    sim_online = similarity_2D(num_training_videos, num_frames, params, dict_with_gt, online_world_states)

    sim_retrospective = similarity_2D(num_videos, num_frames, params, dict_with_gt, retrospective_world_states)

    sim_lesioned = similarity_2D(num_videos, num_frames, params, dict_with_gt, lesioned_world_states)

    #sim_NN_fitted = similarity_2D(num_videos, num_frames, params, dict, ground_truth_world_states, fitted_threshold)

    sim_NN_input = similarity_2D(num_videos, num_frames, params, dict_with_gt, 0.0, top_n)

    new_df = DataFrame(video = 1:num_videos,
    #order_run = retro_data[!, "order_run"],
    #video = retro_data[!, "video_number"],
    #video = lesioned_data[!, "video_number"],
    sim_online = vcat(sim_online, fill(NaN, num_videos - num_training_videos)),
    sim_retrospective = sim_retrospective,
    sim_lesioned = sim_lesioned,
    # sim_lesioned_lower_ci = sim_lesioned_lower_ci,
    # sim_lesioned_upper_ci = sim_lesioned_upper_ci,
    #sim_NN_fitted = sim_NN_fitted,
    sim_NN_input = sim_NN_input)

    CSV.write(full_path * "similarity2D.csv", new_df)
end

function write_accuracy_NN_only(json_path::String, top_n::Int64, num_videos::Int64, num_frames::Int64)
    dict_with_gt = @pipe json_path |> open |> read |> String |> JSON.parse

    sim_NN_input = similarity_2D(num_videos, num_frames, params, dict_with_gt, 0.0, top_n)

    new_df = DataFrame(video = 1:num_videos, sim_NN_input = sim_NN_input)
    CSV.write(json_path * "similarity2D.csv", new_df)
end
