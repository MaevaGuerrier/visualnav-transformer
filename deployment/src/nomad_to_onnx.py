import torch
import yaml
import os
from utils import load_model
import onnxruntime as ort
import onnx
import einops
import torch.nn as nn

MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"

model_name = "nomad"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

with open(MODEL_CONFIG_PATH, "r") as f:
    model_paths = yaml.safe_load(f)

model_config_path = model_paths[model_name]["config_path"]
with open(model_config_path, "r") as f:
    model_params = yaml.safe_load(f)


context_size = model_params["context_size"]
assert context_size != None

# load model weights
ckpth_path = model_paths[model_name]["ckpt_path"]
if os.path.exists(ckpth_path):
    print(f"Loading model from {ckpth_path}")
else:
    raise FileNotFoundError(f"Model weights not found at {ckpth_path}")

model = load_model(
    ckpth_path,
    model_params,
    device,
)
model = model.to(device)

model.eval()

print("loading model")


print("------------------------ Vision Encoder --------------------------------")

vision_encoder = model.vision_encoder
vision_encoder.eval()


dummy_goal = torch.randn(4, 3, 96, 96, device=device)
# Nomad vision encoder takes in 4 past obs, each with 3 channels, img dim 96x96
# Not consistent with paper https://arxiv.org/pdf/2310.07896, past 5 times obs Figure.2
dummy_obs = torch.randn(4, 12, 96, 96, device=device) 

# Issue in paper codebase 'goal_mask' referenced before assignment
# Always set in code because input_goal mask is passed see line 80 and 114 in vint_train/models/nomad/nomad_vint.py
dummy_mask = torch.zeros(1).long().to(device)  
dummy_input_goal_mask=dummy_mask.repeat(len(dummy_goal))

print("Testing forward pass for nomad vision encoder ...")
with torch.no_grad():

    test_embedding = vision_encoder(dummy_obs, dummy_goal, dummy_input_goal_mask)

    print(
        f"Success forward pass for nomad vision encoder with shapes for model {test_embedding}"
    )

# print("\nExporting to vision encoder ONNX...")
# torch.onnx.export(
#     vision_encoder,
#     (dummy_obs, dummy_goal),
#     "nomad_vision_encoder.onnx",
#     export_params=True,
#     opset_version=17,
#     do_constant_folding=True,
#     input_names=["obs", "goal"],
#     output_names=["embedding"],
#     dynamic_axes={"obs": {0: "batch"}, 
#                   "goal": {0: "batch"}, 
#                   "embedding": {0: "batch"}
#     },
# )


# onnx_model = onnx.load("nomad_vision_encoder.onnx")
# onnx.checker.check_model(onnx_model)
# print("ONNX model of vision encoder is valid!")


# print("\nTesting vision encoder ONNX Runtime...")
# providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# ort_session = ort.InferenceSession("nomad_vision_encoder.onnx", providers=providers)
# ort_inputs = {
#     "obs_img": dummy_obs.cpu().numpy(),
#     "goal_img": dummy_goal.cpu().numpy(),
# }

# ort_outputs = ort_session.run(None, ort_inputs)
# print(ort_outputs)
# print(f"ONNX Runtime output shape: {ort_outputs[0].shape} and {ort_outputs[1].shape}")

# # Verify outputs match
# print(f"\nVerifying outputs match...")
# test_cpu_embedding_output = test_embedding.cpu().numpy()
# max_diff_embedding = abs(test_cpu_embedding_output - ort_outputs[0]).max()
# print(f"Maximum difference between for distance PyTorch and ONNX: {max_diff_embedding}")

print("---------------------- End of Vision Encoder ---------------------------------- \n")







# print("---------------------- Noise pred network -----------------------------")

# noise_pred = model.noise_pred_net
# noise_pred.eval()

# dummy_sample      = torch.randn(1, 2, 16, device=device)
# dummy_timestep    = torch.randint(0, 1000, (1,), device=device)
# dummy_globalcond  = torch.randn(1, model_params["encoding_size"], device=device)


# print("Testing forward pass for nomad noise pred network ...")
# with torch.no_grad():

#     test_noise_pred = noise_pred(dummy_sample, dummy_timestep, dummy_globalcond)

#     print(
#         f"Success forward pass for nomad noise pred network with shapes for model {test_noise_pred}"
#     )

# torch.onnx.export(
#     noise_pred,
#     (dummy_sample, dummy_timestep, dummy_globalcond),
#     "nomad_noise_pred_net.onnx",
#     opset_version=17,
#     input_names=["sample", "timestep", "global_cond"],
#     output_names=["noise"],
#     dynamic_axes={"sample": {0: "batch"}, 
#                   "global_cond": {0: "batch"}, 
#                   "noise": {0: "batch"}},
# )



# print("---------------------- Dist pred network -----------------------------")


# dist_pred = model.dist_pred_net
# dist_pred.eval()

# dummy_cond = torch.randn(1, model_params["encoding_size"], device=device)


# print("Testing forward pass for nomad dist pred network ...")
# with torch.no_grad():

#     test_dist_pred = dist_pred(dummy_cond)

#     print(
#         f"Success forward pass for nomad dist pred network with shapes for model {test_dist_pred}"
#     )


# torch.onnx.export(
#     dist_pred,
#     dummy_cond,
#     "nomad_dist_pred_net.onnx",
#     opset_version=17,
#     input_names=["obsgoal_cond"],
#     output_names=["distance_pred"],
#     dynamic_axes={"obsgoal_cond": {0: "batch"}, 
#                   "distance_pred": {0: "batch"}}
# )




# ## ORIG CODE BELOW


# print("------------------------ Distance Pred Network --------------------------------")
# print("converting dist pred network to onnx")

# dummy_obs  = torch.randn(4, 18, 64, 85, device=device)
# dummy_goal = torch.randn(4, 3,  64, 85, device=device)



# output_path = "nomad.onnx"

# print("Testing forward pass for nomad ...")
# with torch.no_grad():

#     test_distance_output, test_action_output = model(dummy_obs, dummy_goal)

#     print(
#         f"Success forward pass for distance enocder with shapes for model {test_distance_output} and { test_action_output}"
#     )

# print("\nExporting to dist enocder ONNX...")
# torch.onnx.export(
#     model,
#     (dummy_obs, dummy_goal),
#     output_path,
#     export_params=True,
#     opset_version=17,
#     do_constant_folding=True,
#     input_names=["obs_img", "goal_img"],
#     output_names=["distances", "waypoints"],
#     dynamic_axes={
#         "obs_img": {0: "batch_size"},
#         "goal_img": {0: "batch_size"},
#         "distances": {0: "batch_size"},      # Add this
#         "waypoints": {0: "batch_size"},      # Add this
#     },
# )


# onnx_model = onnx.load(output_path)
# onnx.checker.check_model(onnx_model)
# print("ONNX model of dist enocder is valid!")


# print("\nTesting dist enocder ONNX Runtime...")
# providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# ort_session = ort.InferenceSession(output_path, providers=providers)
# ort_inputs = {
#     "obs_img": dummy_obs.cpu().numpy(),
#     "goal_img": dummy_goal.cpu().numpy(),
# }
# ort_outputs = ort_session.run(None, ort_inputs)
# print(ort_outputs)
# print(f"ONNX Runtime output shape: {ort_outputs[0].shape} and {ort_outputs[1].shape}")

# # Verify outputs match
# print(f"\nVerifying outputs match...")
# # wrapper_cpu_output = wrapper_dist_output.cpu().numpy()
# test_cpu_distnace_output = test_distance_output.cpu().numpy()
# test_cpu_waypoint_output = test_action_output.cpu().numpy()
# # max_diff_wrapper = abs(wrapper_cpu_output - ort_outputs[0]).max()
# max_diff_model_distance = abs(test_cpu_distnace_output - ort_outputs[0]).max()
# max_diff_action_distance = abs(test_cpu_waypoint_output - ort_outputs[1]).max()
# print(f"Maximum difference between for distance PyTorch and ONNX: {max_diff_model_distance}")
# print(f"Maximum difference between for action PyTorch and ONNX: {max_diff_action_distance}")

# print("---------------------- End of Dist Encoder ----------------------------------")



