
function make_observations(dict::Array{Any,1}, receptive_fields::Vector{Receptive_Field})

    num_videos = 10
    num_frames = 6

    objects_observed = Matrix{Array{Array{Detection2D}}}(undef, num_videos, num_frames)
    #getting undefined reference when I change to Array{Array{}} instead of matrix

    camera_trajectories = Matrix{Camera_Params}(undef, num_videos, num_frames)

    for v=1:num_videos
        for f=1:num_frames
            #indices is where confidence was > 0.5
            indices = dict[v]["views"][f]["detections"]["scores"] .> 0.5
            labels = dict[v]["views"][f]["detections"]["labels"][indices]
            center = dict[v]["views"][f]["detections"]["center"][indices]
            temp = Array{Detection2D}(undef, length(labels))
            for i = 1:length(labels)
                label = labels[i]
                x = center[i][1]
                y = center[i][2]
                temp[i] = (x, y, label) #so now I have an array of detections
            end
            #turn that array of detections into an array of an array of detections sorted by receptive_field
            temp_sorted_into_rfs = map(rf -> filter(p -> within(p, rf), temp), receptive_fields)
            objects_observed[v, f] = temp_sorted_into_rfs

            #camera trajectory
            x = dict[v]["views"][f]["camera"][1]
            y = dict[v]["views"][f]["camera"][2]
            z = dict[v]["views"][f]["camera"][3]
            #focus
            f_x = dict[v]["views"][f]["lookat"][1]
            f_y = dict[v]["views"][f]["lookat"][2]
            f_z = dict[v]["views"][f]["lookat"][3]
            c = Camera_Params(camera_location = Coordinate(x,y,z), camera_focus = Coordinate(f_x,f_y,f_z))
            camera_trajectories[v, f] = c
        end
    end

    return objects_observed, camera_trajectories
end
