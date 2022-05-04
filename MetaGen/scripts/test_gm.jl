using Revise
using MetaGen
using Gen
using Profile
using StatProfilerHTML
using GenRFS

#Profile.init(; n = 10^4, delay = 1e-5)

#GenRFS.modify_partition_ctx!(1000)

#call it
#@profilehtml gt_trace,_ = Gen.generate(metacog, (possible_objects,))
#@profilehtml gt_trace,_ = Gen.generate(metacog, (possible_objects,))
num_frames = 20
num_videos = 10
params = Video_Params(n_possible_objects = 5)

#@time gt_trace,_ = Gen.generate(main, (false, num_videos, num_frames, params));

#@profilehtml Gen.generate(main, (false, num_videos, num_frames, params));


Profile.init(; n = 10^7, delay = 1e-6)
Profile.clear()
#@profilehtml Gen.generate(main, (false, num_videos, num_frames, params));
#println(gt_trace)
#gt_choices = get_choices(gt_trace)

objects_observed = Matrix{Array{Array{Detection2D}}}(undef, num_videos, num_frames)
camera_trajectories = Matrix{Camera_Params}(undef, num_videos, num_frames)

obs = Gen.choicemap()
for v=1:num_videos
    for f=1:num_frames
        obs[:videos => v => :frame_chain => f => :observations_2D] = convert(Array{Any, 1}, [(10., 20., 1), (100., 100., 1), (200., 200., 2)])
	    #objects_observed[v, f] = temp_sorted_into_rfs
        obs[:videos => v => :frame_chain => f => :camera => :camera_location_x] = 0.1
        obs[:videos => v => :frame_chain => f => :camera => :camera_location_y] = 0.001
        obs[:videos => v => :frame_chain => f => :camera => :camera_location_z] = -0.01
        obs[:videos => v => :frame_chain => f => :camera => :camera_focus_x] = 1.0001
        obs[:videos => v => :frame_chain => f => :camera => :camera_focus_y] = 0.00001
        obs[:videos => v => :frame_chain => f => :camera => :camera_focus_z] = -0.001

    end
end

@profilehtml Gen.generate(main, (false, num_videos, num_frames, params), constraints);
