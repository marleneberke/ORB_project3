#using PyCall

using Gen
using Distributions
using FreqTables
using Distances

#push!(pyimport("sys")["path"], pwd());
#pyimport("something.py")[:hello_world]()
#pythonFile = pyimport("something")

#pythonFile.hello_world()

##############################################################################################
#Setting up helper functions

struct TruncatedPoisson <: Gen.Distribution{Int} end

const trunc_poisson = TruncatedPoisson()

function Gen.logpdf(::TruncatedPoisson, x::Int, lambda::U, low::U, high::U) where {U <: Real}
	d = Distributions.Poisson(lambda)
	td = Distributions.Truncated(d, low, high)
	Distributions.logpdf(td, x)
end

function Gen.logpdf_grad(::TruncatedPoisson, x::Int, lambda::U, low::U, high::U)  where {U <: Real}
	gerror("Not implemented")
	(nothing, nothing)
end

function Gen.random(::TruncatedPoisson, lambda::U, low::U, high::U)  where {U <: Real}
	d = Distributions.Poisson(lambda)
	rand(Distributions.Truncated(d, low, high))
end

(::TruncatedPoisson)(lambda, low, high) = random(TruncatedPoisson(), lambda, low, high)
is_discrete(::TruncatedPoisson) = true

has_output_grad(::TruncatedPoisson) = false
has_argument_grads(::TruncatedPoisson) = (false,)

@gen function sample_wo_repl(A,n)
	A_immutable = copy(A)

	println("A is ", A)
	println("n is ", n)

    sample = Array{String}(undef,n)
    for i in 1:n
    	println("i is ", i)
    	
    	idx = @trace(Gen.uniform_discrete(1, length(A)), (:idx, i))
        sample[i] = splice!(A, idx)
        println("A is ", A)
    end
    #want to rearrange so that the order of items in the sample matches the order of items that we're sampling from
    sampleIdx = names_to_IDs(sample, A_immutable)
    sorted = sort(sampleIdx)
    ordered_sample = A_immutable[sorted]
    return ordered_sample
end

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ["BG", "person", "bicycle", "car", "motorcycle", "airplane",
               "bus", "train", "truck", "boat", "traffic light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird",
               "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
               "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
               "suitcase", "frisbee", "skis", "snowboard", "sports ball",
               "kite", "baseball bat", "baseball glove", "skateboard",
               "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
               "donut", "cake", "chair", "couch", "potted plant", "bed",
               "dining table", "toilet", "tv", "laptop", "mouse", "remote",
               "keyboard", "cell phone", "microwave", "oven", "toaster",
               "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"]

#This function converts a list of category names to a list of category IDs. Specific to the COCO
#categories. Must have access to class_names.
function names_to_IDs(names::Vector{String}, possible_objects::Vector{String})
	IDs = Vector{Int}(undef, length(names))
	for i=1:length(names)
		#should only be one location of a given object
		IDs[i] = findfirst(isequal(names[i]),possible_objects)
	end
	return IDs
end

#Define generative model gm. gm takes as input the possible objects and the number of frames.
@gen function gm(possible_objects::Vector{String}, n_frames::Int)

	#need to make one possible_objects to change when replaced, another to not change?
	possible_objects_immutable = copy(possible_objects)
	possible_objects_mutable = copy(possible_objects)

	#Determining visual system V
	V = Matrix{Float64}(undef, length(possible_objects_immutable), 2)

	for j = 1:length(possible_objects_immutable)
		#set false alarm rate
		V[j,1] = @trace(Gen.beta(1.909091, 107.99999999999999), (:fa, j)) #leads to false alarm rate of 0.01
		#set miss rate
		V[j,2] = @trace(Gen.beta(1.909091, 36.272727), (:m, j)) #leads to miss rate of 0.05
	end

	#Determining frame of reality R
	lambda = 1 #must be <= length of possible_objects
	low = 1
	high = length(possible_objects_immutable)

	println("possible_objects_mutable ", possible_objects_mutable)

	numObjects = @trace(trunc_poisson(lambda, low, high), :numObjects)
    R = @trace(sample_wo_repl(possible_objects_mutable,numObjects), :R) #order gets mixed up

	#Determing the percept based on the visual system V and the reality frame R
    #A percept is a matrix where each row is the percept for a frame.
	percept = Matrix{Bool}(undef, n_frames, length(possible_objects_immutable))
	for f = 1:n_frames
		for j = 1:length(possible_objects_immutable)
			#if the object is in the reality R, it is detected according to 1 - its miss rate
			if possible_objects_immutable[j] in R
				M =  V[j,2]
				percept[f,j] = @trace(bernoulli(1-M), (:percept, f, j))
			else
				FA =  V[j,1]
				percept[f,j] = @trace(bernoulli(FA), (:percept, f, j))
			end
		end
	end


	return (R,V,percept); #returning reality R, (optional)
end;


##############################################################################################
#Defining observations / constraints

possible_objects = ["person", "bicycle", "car","motorcycle", "airplane"]
#Later, possible_objects will equal class_names

# define some observations
gt = Gen.choicemap()

#initializing the generative model. It will create the ground truth V and R
#generates the data and the model
n_frames = 10
gt_trace,_ = Gen.generate(gm, (possible_objects, n_frames))
gt_reality,gt_V,gt_percept = Gen.get_retval(gt_trace)

println("gt_reality is ",gt_reality)
println("gt_percept is ",gt_percept) #could translate back into names

# #Translating gt_percept back into names
# percept = Matrix{String}(undef, n_frames, length(possible_objects))
# for f = 1:n_frames
# 	percept[f,:] = possible_objects[gt_percept[f,:]]
# end
# println("percept is ",percept)

gt_choices = Gen.get_choices(gt_trace)
println(gt_choices)


#get the percepts
#obs = Gen.get_submap(gt_choices, :percept)
observations = Gen.choicemap()
nrows,ncols = size(gt_percept)
for i = 1:nrows
	for j = 1:ncols
			observations[(:percept,i,j)] = gt_percept[i,j]
	end
end


##################################################################################################################

#initialize a new trace
#trace, _ = Gen.generate(gm, (possible_objects, n_frames), observations)

#Inference procedure 1: Importance resampling

num_samples = 100

#add log_norm_weights as middle return arguement for importance_sampling
#(trace, lml_est) = Gen.importance_resampling(gm, (possible_objects, n_frames), observations, num_samples)
#(traces, log_norm_weights, lml_est) = Gen.importance_sampling(gm, (possible_objects, n_frames), observations, num_samples)
#for some reason importance_sampling seems not to account for observations

#trying repeated importance_resampling
# amount_of_computation_per_resample = 100
# traces = []
# log_probs = Array{Float64}(undef, num_samples)
# for i = 1:num_samples
#      (tr, lml_est) = Gen.importance_resampling(gm, (possible_objects, n_frames), observations, amount_of_computation_per_resample)
#      push!(traces,tr)
#      log_probs[i] = Gen.get_score(tr)
# end


#################################################################################################################################

#trying Metropolis Hastings


# trace, _ = Gen.generate(gm, (possible_objects, n_frames), observations)

# # Perform a single block resimulation update of a trace.
# function block_resimulation_update(tr)

#     # Block 1: Update the reality
#     reality = select(:R)
#     (tr, _) = mh(tr, reality)
    
#     # Block 2: Update the visual system
#     (possible_objects, n_frames) = get_args(tr)
#     n = length(possible_objects)
#     for i = 1:n
#     	row_V = select((:(fa,i)),(:(m,i)))
#     	(tr, _) = mh(tr, row_V)
#         #(tr, _) = mh(tr, select(:(fa,i)))
#         #(tr, _) = mh(tr, select(:(m,i)))
#     end
    
#     # Return the updated trace
#     tr
# end;

# function block_resimulation_inference((possible_objects, n_frames), observations)
#     (tr, _) = generate(gm, (possible_objects, n_frames), observations)
#     for iter=1:amount_of_computation_per_resample
#         tr = block_resimulation_update(tr)
#     end
#     tr
# end;


# traces = []
# for i=1:num_samples
#     tr = block_resimulation_inference((possible_objects, n_frames), observations)
#     push!(traces,tr)
# end



#################################################################################################################################


#trying Hamiltonian MC


trace, _ = Gen.generate(gm, (possible_objects, n_frames), observations)

#Perform a single block resimulation update of a trace.
function block_resimulation_update(tr)

    # Block 1: Update the reality
    reality = select(:R)
    (tr, _) = hmc(tr, reality)
    
    # Block 2: Update the visual system
    (possible_objects, n_frames) = get_args(tr)
    n = length(possible_objects)
    for i = 1:n
    	row_V = select((:(fa,i)),(:(m,i)))
    	(tr, _) = hmc(tr, row_V)
    end
    
    # Return the updated trace
    tr
end;

function block_resimulation_inference((possible_objects, n_frames), observations)
    (tr, _) = generate(gm, (possible_objects, n_frames), observations)
    for iter=1:amount_of_computation_per_resample
        tr = block_resimulation_update(tr)
    end
    tr
end;


traces = []
for i=1:num_samples
    tr = block_resimulation_inference((possible_objects, n_frames), observations)
    push!(traces,tr)
end


#################################################################################################################################


burnin = 50 #how many samples to ditch

realities = Array{String}[]
Vs = Array{Float64}[]
for i = burnin+1:num_samples
	reality,V,_ = Gen.get_retval(traces[i])
	push!(realities,reality)
	push!(Vs,V)
end

###################################################################################################################
#Analysis

#want to make a frequency table of the realities sampled
ft = freqtable(realities)

#compare means of Vs to gt_V
#for false alarms
euclidean(gt_V[1], mean(Vs)[1])
#for hit rates
euclidean(gt_V[2], mean(Vs)[2])


#want, for each reality, to bin Vs
unique_realities = unique(realities)
avg_Vs_binned = Array{Float64}[]
freq = Array{Float64}(undef, length(unique_realities))

for j = 1:length(unique_realities)
	index = findall(isequal(unique_realities[j]),realities)
	#freq keeps track of how many there are
	freq[j] = length(index)
	push!(avg_Vs_binned, mean(Vs[index]))
end

#find avg_Vs_binned at most common realities and compute euclidean distances
#index of most frequent reality
idx = findfirst(isequal(maximum(freq)),freq)
unique_realities[idx]

#compare mean V of most frequent reality to gt_V
#for false alarms
euclidean(gt_V[1], Vs[idx][1])
#for hit rates
euclidean(gt_V[2], Vs[idx][2])

#compare to least frequent reatlity
#index of most frequent reality
idx2 = findfirst(isequal(minimum(freq)),freq)
unique_realities[idx2]

#compare mean V of most frequent reality to gt_V
#for false alarms
euclidean(gt_V[1], Vs[idx2][1])
#for hit rates
euclidean(gt_V[2], Vs[idx2][2])





#going to explode quickly as possible_objects gets larger and samples get larger



#want to compare Vs sampled to ground V

# for i=1:numIters
#     (tr, _) = Gen.importance_resampling(gm, (possible_objects, n_frames), obs, 2000)
#     putTrace!(viz, "t$(i)", serialize_trace(tr))
#     log_probs[i] = Gen.get_score(tr)
# end





# inference_history = Vector{typeof(trace)}(undef, N)
# for i = 1:N
# 	#selection = Gen.select(:fa, :m, :numObjects, :obj_selection)
# 	trace,_ = Gen.hmc(trace)
# 	inference_history[i] = trace
# end