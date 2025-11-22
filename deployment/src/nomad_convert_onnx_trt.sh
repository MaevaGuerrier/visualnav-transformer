








trtexec --onnx=nomad_vision_encoder.onnx --saveEngine=nomad_vision_encoder.trt --minShapes=obs_img:1x12x96x96,goal_img:1x3x96x96,input_goal_mask:1 --optShapes=obs_img:4x12x96x96,goal_img:4x3x96x96,input_goal_mask:4 --maxShapes=obs_img:8x12x96x96,goal_img:8x3x96x96,input_goal_mask:8 --memPoolSize=workspace:4096  --verbose



trtexec --onnx=nomad_dist_pred_net.onnx --saveEngine=nomad_dist_pred_net.trt --fp16 --minShapes=obsgoal_cond:1x256 --optShapes=obsgoal_cond:4x256 --maxShapes=obsgoal_cond:8x256 --memPoolSize=workspace:4096  --verbose


# trtexec --onnx=nomad_noise_pred_net.onnx --saveEngine=nomad_noise_pred_net.trt 
# --minShapes=sample:1x8x2,timestep:,global_cond:1x256 
# --optShapes=sample:4x8x2,timestep:,global_cond:4x256 
# --maxShapes=sample:8x8x2,timestep:,global_cond:8x256 
# --memPoolSize=workspace:4096  --verbose





# trtexec --onnx=nomad_noise_pred_net.onnx --saveEngine=nomad_noise_pred_net.trt --minShapes=sample:1x8x2,global_cond:1x256 --optShapes=sample:4x8x2,global_cond:4x256 --maxShapes=sample:8x8x2,global_cond:8x256 --memPoolSize=workspace:4096  --verbose
