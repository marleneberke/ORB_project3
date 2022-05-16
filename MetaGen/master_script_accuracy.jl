# # #For some reason, have to add CSV with "using CSV" before activating the MetaGen environment
# #
# #This is for fitting a threshold, then getting 2D accuracy.
# include("fitting_threshold.jl")
#
# NN = "detr"
# overall_path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/05_09_22/"*NN*"/Final_Results/Full/"
#
# #file_path = overall_path*"../data_labelled_"*NN*".json"
# file_path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/05_09_22/"*NN*"/data_labelled_"*NN*".json"
#
# num_training_videos = 50
# num_frames = 20
# office_subset = ["chair", "bowl", "umbrella", "potted plant", "tv"]
#
# fitted_threshold = fitting_threshold(file_path, num_training_videos, num_frames)
#
# include("bootstrap_preprocess_data_hungarian.jl")
#
# num_videos = 100
# top_n = 5
# threshold = 0.0 #threshold for MetaGen inputs
# num_particles = 100
# params = Video_Params(n_possible_objects = 5)
#
# for i = 0:3
#     path = overall_path * "/shuffle_" *string(i)* "_" *NN* "/"
#     write_accuracy_csvs(path, fitted_threshold, top_n, num_videos, num_training_videos, num_frames)
# end

################################################################################
include("accuracy_just_retro.jl")

NN = "detr"
#NN2 = "retinanet"
overall_path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/05_09_22/"*NN*"/Final_Results/Lesioned/"
#
json_file_path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/05_09_22/"*NN*"/data_labelled_"*NN*".json"
#folder_path  = overall_path*"mean_of_prior/"*NN

num_videos = 100
left_off = 1
num_frames = 20
office_subset = ["chair", "bowl", "umbrella", "potted plant", "tv"]
threshold = 0.0 #threshold for MetaGen inputs
num_particles = 100
params = Video_Params(n_possible_objects = 5)


for i = 0:3
    path = overall_path*"/shuffle_" *string(i)* "_" *NN* "/"
    accuracy_just_retro(json_file_path, path)
end

################################################################################
# NN = "detr"
# overall_path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/04_14_22/iclr_empty_room/"
# json_file_path = overall_path*"data_labelled_"*NN*"_0.json"
#
# include("3D_accuracy.jl")
#
# num_videos
# num_training_videos = 50
#
# for i = 0:3
#     path = overall_path *"200_MCMC_top_5_" *NN* "/shuffle_" *string(i)* "_" *NN* "/"
#     write_3D_accuracy_csvs(path, num_videos, num_training_videos)
# end
