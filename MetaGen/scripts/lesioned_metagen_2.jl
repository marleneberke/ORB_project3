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

#we only want the last 50 videos, and do not want to shuffle.
dict = dict[51:100]

################################################################################
#num_videos = config["num_videos"]
num_videos = config["num_videos"]
num_frames = config["num_frames"]
threshold = config["threshold"]

categories_subset = config["categories_subset"]

params = Video_Params(n_possible_objects = length(categories_subset))
top_n = 5

receptive_fields = make_receptive_fields(params)
objects_observed, camera_trajectories = make_observations_office(dict, receptive_fields, categories_subset, num_videos, num_frames, threshold, top_n)

num_particles = config["num_particles"]
mcmc_steps_outer = config["mcmc_steps_outer"]
mcmc_steps_inner = config["mcmc_steps_inner"]
#shuffle_type = config["shuffle_type"]
lesion_type = config["lesion_type"]
################################################################################
#lesion types are 0,1
v = zeros(length(params.possible_objects), 2)

#lesion type 1
#miss = 0, hallucination is mean of prior
if lesion_type == 1
	v[:,1] .= 1.0 #hallucination
	v[:,2] .= 0.000000000001 #miss rate
#hallucination = 0, miss is mean of prior
else
	v[:,1] .= 0.000000000001 #hallucination
	v[:,2] .= 0.5 #miss rate
end
################################################################################
#Lesioned MetaGen

#training set and test set
order = 1:50
input_objects_observed = objects_observed
input_camera_trajectories = camera_trajectories

#Set up the output file
lesion_V_file = open(output_dir * "/lesion_V.csv", "w")
file_header_V(lesion_V_file, params)
lesion_ws_file = open(output_dir * "/lesion_ws.csv", "w")
file_header_ws(lesion_ws_file, params, num_particles)

println("start lesioned")

traces, inferred_world_states, v = unfold_particle_filter(v, num_particles, mcmc_steps_outer, mcmc_steps_inner,
	input_objects_observed, input_camera_trajectories, params, lesion_V_file, lesion_ws_file, order)
close(retro_V_file)
close(retro_ws_file)

println("done with pf for lesioned")

#=

################################################################################
#run Lesioned MetaGen

#Set up the output file
lesioned_outfile = output_dir * "/lesioned_output.csv"
lesioned_file = open(lesioned_outfile, "w")
file_header(lesioned_file)

v = zeros(length(params.possible_objects), 2)
v[:,1] .= 1.0
v[:,2] .= 0.5
unfold_particle_filter(v, num_particles, mcmc_steps_outer, mcmc_steps_inner,
	objects_observed, camera_trajectories, params, lesioned_file)
close(lesioned_file)

println("done with pf for lesioned metagen")


=#

################################################################################
#for writing an output file for a demo using Retro MetaGen.

#undor re-ordering of inferred_world_states
inferred_world_states = inferred_world_states[order]

###### add to dictionary
out = write_to_dict(dict, camera_trajectories, inferred_world_states, num_videos, num_frames)

#open("../../scratch_work_07_16_21/output_tiny_set_detections.json","w") do f
open(output_dir * "/output.json","w") do f
	JSON.print(f,out)
end

println("finished writing json")
