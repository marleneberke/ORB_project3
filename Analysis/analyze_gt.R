library(tidyverse)
#setwd("~/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/Data/20particles_threshold64_18945877")
#setwd("~/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/08_30/")
#setwd("~/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/08_31/with_retro_and_lesioned/100particles_threshold07_18992814/")
setwd("~/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/09_30/dataset_1_obj_model2/detr/shuffle_1_detr/")

data = read_delim("similarity2D.csv", delim=',')

num_training_videos = 50
total_num_videos = 100

#mean(data$sim_lesioned[1:num_training_videos])
mean(data$sim_online[1:num_training_videos])
mean(data$sim_retrospective[1:num_training_videos])
mean(data$sim_NN_fitted[1:num_training_videos])
mean(data$sim_NN_input[1:num_training_videos])


#mean(data$sim_lesioned[(num_training_videos+1):total_num_videos])
mean(data$sim_retrospective[(num_training_videos+1):total_num_videos])
mean(data$sim_NN_fitted[(num_training_videos+1):total_num_videos])
mean(data$sim_NN_input[(num_training_videos+1):total_num_videos])


data1 = data %>% select(order_run, sim_online, sim_retrospective, sim_NN_input, sim_NN_fitted)
# data_lower = data %>% select(run_order, sim_online_lower_ci, sim_lesioned_lower_ci, sim_retrospective_lower_ci)
# data_lower$sim_NN_lower_ci <- rep(NA, length(nrow(data_lower)))
# data_upper = data %>% select(run_order, sim_online_upper_ci, sim_lesioned_upper_ci, sim_retrospective_upper_ci)
# data_upper$sim_NN_upper_ci <- rep(NA, length(nrow(data_upper)))

temp <- gather(data1, "model", "value", c(sim_online, sim_retrospective, sim_NN_input, sim_NN_fitted))
#temp_lower <- gather(data_lower, "model", "lower", c(sim_online_lower_ci, sim_retrospective_lower_ci, sim_NN_lower_ci))
#temp_upper <- gather(data_upper, "model", "upper", c(sim_online_upper_ci, sim_retrospective_upper_ci, sim_NN_upper_ci))
#temp$lower <- temp_lower$lower
#temp$upper <- temp_upper$upper

#df = gather(data, "model", "value", c(sim_online, sim_lesioned, sim_retrospective, sim_NN))
#df = gather(data, "model", "value", c(sim_online, sim_NN))


#jitter <- position_jitter(width = 5, height = )
p <- ggplot(
  temp,
  aes(
    x = order_run,
    y = value,
    color = model#,
    #ymin = lower,
    #ymax = upper
  )
) + #geom_point() +
 # geom_ribbon(alpha = 0.5) #+ 
  geom_point() +
  geom_smooth(method='loess', formula= y~x) + 
  xlim = c(0,50) + theme(aspect.ratio=1)
p

p <- ggplot(
  temp,
  aes(
    x = order_run,
    y = value,
    color = model
  )
) + geom_point() +
  geom_smooth(method='loess', formula= y~x) + 
  xlim = c(0,50) + theme(aspect.ratio=1)
p



temp <- temp %>% filter(order_run <= num_training_videos)
p <- ggplot(
  temp,
  aes(
    x = order_run,
    y = value,
    color = model
  )
) + geom_point() +
  geom_smooth(method='loess', formula= y~x) + theme(aspect.ratio=1)
p
