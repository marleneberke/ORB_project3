# #For some reason, have to add CSV with "using CSV" before activating the MetaGen environment
include("write_accuracy_csvs.jl")
include("bb_accuracy_helper.jl")
include("../scripts/useful_functions.jl")

NN = "faster_rcnn"
#writes results to a csv in full_path
full_path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/05_09_22/"*NN*"/Final_Results/Full_for_trained_NN/"
lesion_path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/05_09_22/"*NN*"/Final_Results/Lesioned/"

#file_path = overall_path*"../data_labelled_"*NN*".json"
#file_path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/05_09_22/"*NN*"/data_labelled_"*NN*".json"

num_training_videos = 50
num_frames = 20
office_subset = ["chair", "bowl", "umbrella", "potted plant", "tv"]

num_videos = 100
top_n = 5
threshold = 0.0 #threshold for MetaGen inputs
num_particles = 100
params = Video_Params(n_possible_objects = 5)

for i = 0:3
    full_path_shuffle_n = full_path * "/shuffle_" *string(i)* "_" *NN* "/"
    lesion_path_shuffle_n = lesion_path * "/shuffle_" *string(i)* "_" *NN* "/"
    write_accuracy_csvs(full_path_shuffle_n, lesion_path_shuffle_n, top_n, num_videos, num_training_videos, num_frames)
end

################################################################################
#file_path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/05_09_22/faster_rcnn/Final_Results/Trained_NN/data_labelled_faster_rcnn_shuffle_3-fine-tuned-1.pt.json"
#write_accuracy_NN_only(file_path, top_n, num_videos, num_frames)
