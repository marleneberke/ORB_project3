println("Starting")

using MetaGen
using JSON
import YAML
using Pipe: @pipe
using Random

config_path = ARGS[1]
config = YAML.load_file(config_path)
output_dir = "results_marlene/$(config["experiment_name"])_" * ENV["SLURM_JOB_ID"]
mkdir(output_dir)

include("useful_functions.jl")
dict = []
#for i = 0:config["batches_upto"]
	#to_add =  @pipe "$(config["input_file_dir"])$(i)_data_labelled.json" |> open |> read |> String |> JSON.parse
to_add =  @pipe "$(config["input_file"])" |> open |> read |> String |> JSON.parse
append!(dict, to_add)
#end
#dict = @pipe "../../scratch_work_07_16_21/0_data_labelled.json" |> open |> read |> String |> JSON.parse
#dict = @pipe "../../scratch_work_07_16_21/0_data_labelled.json" |> open |> read |> String |> JSON.parse


#Random.seed!(15)
#try to make objects_observed::Array{Array{Array{Array{Detection2D}}}} of observed objects.
#outer array is for scenes, then frames, the receptive fields, then last is an array of detections

################################################################################
num_videos = config["num_videos"]
num_frames = config["num_frames"]
threshold = config["threshold"]
top_n = config["top_n"]

categories_subset = config["categories_subset"]

sigma = config["sigma"]

params = Video_Params(n_possible_objects = length(categories_subset), sigma = sigma)

receptive_fields = make_receptive_fields(params)
objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, categories_subset, num_videos, num_frames, threshold, top_n)

num_particles = config["num_particles"]
mcmc_steps_outer = config["mcmc_steps_outer"]
mcmc_steps_inner = config["mcmc_steps_inner"]

left_off = config["left_off"]
path_to_v = config["path_to_v"]
avg_v_dict = @pipe path_to_v |> open |> read |> String |> JSON.parse
avg_v = reduce(hcat, avg_v_dict["avg_v"])
################################################################################
#Retrospective MetaGen
println("start retrospective")

#training set and test set
order = collect(left_off:num_videos)
input_objects_observed = vcat(objects_observed[order, :])
input_camera_trajectories = vcat(camera_trajectories[order, :])


#Set up the output file
retro_V_file = open(output_dir * "/retro_V.csv", "w")
file_header_V(retro_V_file, params)
retro_ws_file = open(output_dir * "/retro_ws.csv", "w")
file_header_ws(retro_ws_file, params, num_particles)

println("start retrospective")

traces, inferred_world_states, avg_v = unfold_particle_filter(avg_v, num_particles, mcmc_steps_outer, mcmc_steps_inner,
	input_objects_observed, input_camera_trajectories, params, retro_V_file, retro_ws_file, order)
close(retro_V_file)
close(retro_ws_file)

println("done with pf for retrospective")

################################################################################

################################################################################
#for writing an output file for a demo using MetaGen

###### add to dictionary
# out = write_to_dict(dict, camera_trajectories, inferred_world_states, num_videos, num_frames)
#
# #open("../../scratch_work_07_16_21/output_tiny_set_detections.json","w") do f
# open(output_dir * "/output.json","w") do f
# 	JSON.print(f,out)
# end
#
# println("finished writing json")
