#This is for calcuting the groud-truth V, then getting learning curve.

include("ideal_V_using_bbs.jl")
include("bb_accuracy_helper.jl") #for the hungarian stuff
include("../bootstrap_V.jl")
include("../scripts/useful_functions.jl")

NN = "retinanet"
path0 = overall_path = "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/05_09_22/"*NN*"/Final_Results/Full/"

office_subset = ["chair", "bowl", "umbrella", "potted plant", "tv"]
num_videos = 100
num_frames = 20

gt_V(path0*"/shuffle_0_"*NN*"/")
################################################################################

include("../bootstrap_V.jl")
num_videos = 50 #these are just the training videos
params = Video_Params(n_possible_objects = length(office_subset))

for i = 0:3
    path = path0*"/shuffle_"*string(i)*"_"*NN*"/"
    bootstrap_V(path)
end