#script for pre-processing data. Ends with giving Jacard similarity between
#ground-truth world states and the mode inferred world state, and 95% CI on that

using CSV
using DataFrames
using JSON
using Pipe: @pipe
using MetaGen
using Bootstrap

include("helper_function.jl")

path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/01_09/detr250/shuffle_0_detr/"
#path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/09_30/dataset_1_obj_model2_iclr/detr/shuffle_3_detr/"


online_data = CSV.read(path * "online_ws.csv", DataFrame; delim = "&")
retro_data = CSV.read(path * "retro_ws.csv", DataFrame; delim = "&")
#lesioned_data = CSV.read(path * "lesion_ws.csv", DataFrame; delim = "&")

################################################################################
#could equally use input or output dictionary
#dict = @pipe path * "../data_labelled_detr.json" |> open |> read |> String |> JSON.parse
dict = @pipe path * "output.json" |> open |> read |> String |> JSON.parse
#dict = @pipe "../../scratch_work_07_16_21/08_20/data_labelled.json" |> open |> read |> String |> JSON.parse

online_data = sort!(online_data, :video_number) #undoes whatever shuffle happened
retro_data = sort!(retro_data, :video_number)


num_videos = 100
num_frames = 20

num_particles = 100

num_training_videos = 50

fitted_threshold = 0.20
threshold = 0.0
top_n = 5

office_subset = ["chair", "bowl", "umbrella", "potted plant", "tv"]

#reorder the dictionary to match the order run
#order = retro_data[!, "video_number"]
#dict = dict[order]

ground_truth_world_states = get_ground_truth(dict, num_videos)

################################################################################
online_world_states = new_parse_data(online_data, num_training_videos, num_particles)
retrospective_world_states = new_parse_data(retro_data, num_videos, num_particles)
#lesioned_world_states = new_parse_data(lesioned_data, num_videos, num_particles)

################################################################################
#get just object categories
ground_truth_categories = extract_category(ground_truth_world_states)
online_categories = extract_category(online_world_states, num_training_videos, num_particles)
retrospective_categories = extract_category(retrospective_world_states, num_videos, num_particles)
#lesioned_categories = extract_category(lesioned_world_states, num_videos, num_particles)

#jaccard similarity
sim_online = jaccard_similarity(ground_truth_categories, online_categories, num_particles)
sim_retrospective = jaccard_similarity(ground_truth_categories, retrospective_categories, num_particles)
#sim_lesioned = jaccard_similarity(ground_truth_categories, lesioned_categories, num_particles)

#get CI
sim_online, sim_online_lower_ci, sim_online_upper_ci = add_confidence_interval(sim_online, 1000, 0.95)
sim_retrospective, sim_retrospective_lower_ci, sim_retrospective_upper_ci = add_confidence_interval(sim_retrospective, 1000, 0.95)
#sim_lesioned, sim_lesioned_lower_ci, sim_lesioned_upper_ci = add_confidence_interval(sim_lesioned, 1000, 0.95)

################################################################################
#make new dataframe. just has stuff on the similarity.

new_df = DataFrame(video = 1:num_videos,
    #order_run = retro_data[!, "order_run"],
    #video = retro_data[!, "video_number"],
    #video = lesioned_data[!, "video_number"],
    sim_online = vcat(sim_online, fill(NaN, num_videos - num_training_videos)),
    sim_online_lower_ci = vcat(sim_online_lower_ci, fill(NaN, num_videos - num_training_videos)),
    sim_online_upper_ci = vcat(sim_online_upper_ci, fill(NaN, num_videos - num_training_videos)),
    sim_retrospective = sim_retrospective,
    sim_retrospective_lower_ci = sim_retrospective_lower_ci,
    sim_retrospective_upper_ci = sim_retrospective_upper_ci,
    #sim_lesioned = sim_lesioned,
    #sim_lesioned_lower_ci = sim_lesioned_lower_ci,
    #sim_lesioned_upper_ci = sim_lesioned_upper_ci,
    )
#new_df = DataFrame(video = 1:num_videos, sim_online = sim_online)
CSV.write(path * "similarity3D.csv", new_df)

################################################################################
#repeat Jaccard sim in 2D

include("scripts/useful_functions.jl")

params = Video_Params(n_possible_objects = 5)

sim_online, sim_online_lower_ci, sim_online_upper_ci = jacccard_sim_2D(num_training_videos,
    num_frames, params, dict, ground_truth_world_states, online_world_states,
    1000, 0.95)

sim_retrospective, sim_retrospective_lower_ci, sim_retrospective_upper_ci = jacccard_sim_2D(num_videos,
    num_frames, params, dict, ground_truth_world_states, retrospective_world_states,
    1000, 0.95)

# sim_lesioned, sim_lesioned_lower_ci, sim_lesioned_upper_ci = jacccard_sim_2D(num_videos,
#     num_frames, params, dict, ground_truth_world_states, lesioned_world_states,
#     1000, 0.95)

sim_NN_fitted = jacccard_sim_2D(num_videos, num_frames, params, dict, ground_truth_world_states, fitted_threshold)

sim_NN_input = jacccard_sim_2D(num_videos, num_frames, params, dict, ground_truth_world_states, 0.0, top_n)

new_df = DataFrame(video = 1:num_videos,
    #order_run = retro_data[!, "order_run"],
    #video = retro_data[!, "video_number"],
    #video = lesioned_data[!, "video_number"],
    sim_online = vcat(sim_online, fill(NaN, num_videos - num_training_videos)),
    sim_online_lower_ci = vcat(sim_online_lower_ci, fill(NaN, num_videos - num_training_videos)),
    sim_online_upper_ci = vcat(sim_online_upper_ci, fill(NaN, num_videos - num_training_videos)),
    sim_retrospective = sim_retrospective,
    sim_retrospective_lower_ci = sim_retrospective_lower_ci,
    sim_retrospective_upper_ci = sim_retrospective_upper_ci,
    # sim_lesioned = sim_lesioned,
    # sim_lesioned_lower_ci = sim_lesioned_lower_ci,
    # sim_lesioned_upper_ci = sim_lesioned_upper_ci,
    sim_NN_fitted = sim_NN_fitted,
    sim_NN_input = sim_NN_input)

CSV.write(path * "similarity2D.csv", new_df)
