import inspect
from inspect import signature

import random
import torch
import yaml
import os
from utils import load_model
import onnxruntime as ort
import onnx
import einops
import torch.nn as nn


class NoisePredNetWrapper(nn.Module):
    def __init__(self, nomad_model):
        super().__init__()
        self.noise_pred_net = nomad_model.noise_pred_net

    def forward(self, sample, timestep, global_cond):
        # Call the underlying noise_pred_net directly with named arguments
        return self.noise_pred_net(
            sample=sample, timestep=timestep, global_cond=global_cond
        )




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


# print("------------------------ Vision Encoder --------------------------------")

# vision_encoder = model.vision_encoder
# vision_encoder.eval()


# dummy_goal = torch.randn(4, 3, 96, 96, device=device)
# # Nomad vision encoder takes in 4 past obs, each with 3 channels, img dim 96x96
# # Not consistent with paper https://arxiv.org/pdf/2310.07896, past 5 times obs Figure.2
# dummy_obs = torch.randn(4, 12, 96, 96, device=device) 

# # Issue in paper codebase 'goal_mask' referenced before assignment
# # Always set in code because input_goal mask is passed see line 80 and 114 in vint_train/models/nomad/nomad_vint.py
# dummy_mask = torch.zeros(1).long().to(device)  
# dummy_input_goal_mask=dummy_mask.repeat(len(dummy_goal))

# print("Testing forward pass for nomad vision encoder ...")
# with torch.no_grad():

#     test_obs_encoding_tokens = vision_encoder(dummy_obs, dummy_goal, dummy_input_goal_mask)

#     print(
#         f"Success forward pass for nomad vision encoder with shapes for model {test_obs_encoding_tokens}"
#     )

# print("\nExporting to vision encoder ONNX...")
# torch.onnx.export(
#     vision_encoder,
#     (dummy_obs, dummy_goal, dummy_input_goal_mask),
#     "nomad_vision_encoder.onnx",
#     export_params=True,
#     opset_version=17,
#     do_constant_folding=True,
#     input_names=["obs_img", "goal_img", "input_goal_mask"], # This has to be the same as forward inputs (e.g., forward(self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor = None))
#     output_names=["obs_encoding_tokens"], # This has to be the same as forward outputs (e.g., return output)
#     dynamic_axes={"obs_img": {0: "batch"}, 
#                   "goal_img": {0: "batch"}, 
#                   "input_goal_mask": {0: "batch"},
#                   "obs_encoding_tokens": {0: "batch"}
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
#     "input_goal_mask": dummy_input_goal_mask.cpu().numpy(),
# }

# ort_outputs = ort_session.run(None, ort_inputs)
# print(ort_outputs)
# print(f"ONNX Runtime output shape: {ort_outputs[0].shape}")

# # Verify outputs match
# print(f"\nVerifying outputs match...")
# test_cpu_obs_encoding_tokens_output = test_obs_encoding_tokens.cpu().numpy()
# max_diff_obs_encoding_tokens = abs(test_cpu_obs_encoding_tokens_output - ort_outputs[0]).max()
# print(f"Maximum difference for distance between PyTorch and ONNX: {max_diff_obs_encoding_tokens}")

# print("---------------------- End of Vision Encoder ---------------------------------- \n")



# print("---------------------- Dist pred network -----------------------------")


# dist_pred_net = model.dist_pred_net
# dist_pred_net.eval()

# # obsgoal_cond = model('vision_encoder', ...
# # dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond) --> dist takes inputs of obsgoal_cond
# # test_obs_encoding_tokens.shape torch.Size([4, 256])
# dummy_obsgoal_cond = torch.randn(test_obs_encoding_tokens.shape[0], test_obs_encoding_tokens.shape[1], device=device)
# # VERY IMPORTANT the first input can be changed (see --radius in navigate.py)

# print("Testing forward pass for nomad dist pred network ...")
# with torch.no_grad():

#     test_dist_pred = dist_pred_net(dummy_obsgoal_cond)

#     print(
#         f"Success forward pass for nomad dist pred network with shapes for model {test_dist_pred}"
#     )

# onnx_dist_pred = "nomad_dist_pred_net.onnx"

# torch.onnx.export(
#     dist_pred_net,
#     dummy_obsgoal_cond,
#     onnx_dist_pred,
#     opset_version=17,
#     input_names=["obsgoal_cond"],
#     output_names=["distances_pred"],
#     dynamic_axes={"obsgoal_cond": {0: "batch"}, 
#                   "distances_pred": {0: "batch"}}
# )


# onnx_model = onnx.load(onnx_dist_pred)
# onnx.checker.check_model(onnx_model)
# print("ONNX model of distance predictor is valid!")


# print("\nTesting distance predictor ONNX Runtime...")
# providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# ort_session = ort.InferenceSession(onnx_dist_pred, providers=providers)
# ort_inputs = {
#     "obsgoal_cond": dummy_obsgoal_cond.cpu().numpy(),
# }

# ort_outputs = ort_session.run(None, ort_inputs)
# print(ort_outputs)
# print(f"ONNX Runtime output shape: {ort_outputs[0].shape}") # One output only, distances

# # Verify outputs match
# print(f"\nVerifying outputs match...")
# test_cpu_dist_pred_output = test_dist_pred.cpu().numpy()
# max_diff_dist_pred = abs(test_cpu_dist_pred_output - ort_outputs[0]).max()
# print(f"Maximum difference for distance between PyTorch and ONNX: {max_diff_dist_pred}")

# print("---------------------- End of Distance Predictor ---------------------------------- \n")
























#############################################################################

## DEBUG CODE BELOW - 

print("---------------------- Noise pred network -----------------------------")

noise_pred = model.noise_pred_net
noise_pred.eval()

# print(signature(noise_pred.forward))
# (sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], local_cond=None, global_cond=None, **kwargs)
# SHAPES e.g.,: sample/naction torch.Size([8, 8, 2]), noise_pred torch.Size([8, 8, 2]), timestep 9, obs_cond torch.Size([8, 256])
# This can be changes by user actually - see navigate.py 
dummy_sample = torch.randn(8, 8, 2, device=device)
# noise_scheduler.timesteps: tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
# high is included
# Diffusion timestep: val e.g., 9, type: <class 'torch.Tensor'> torch type: torch.int64, shape torch.Size([])
dummy_timestep = torch.randint(low=0, high=model_params["num_diffusion_iters"], size=(), device=device, dtype=torch.int64)

# same as obsgoal_cond from vision encoder output
dummy_global_cond = torch.randn(8, model_params["encoding_size"], device=device) 
dummy_local_cond = None  # not used in nomad codebase

print("Testing forward pass for nomad noise pred network ...")
with torch.no_grad():

    # IMPORTANT: local_cond is never used in nomad codebase, thus we have to precise parameters by name
    test_noise_pred = noise_pred(sample=dummy_sample, timestep=dummy_timestep, local_cond=dummy_local_cond, global_cond=dummy_global_cond)

    print(
        f"Success forward pass for nomad noise pred network with shapes for model {test_noise_pred.shape}"
    )


onnx_noise_pred = "nomad_noise_pred_net.onnx"

# ONNX gaph does not keep track of optional parameters in forward method
# Hence why we are using the wrapper
# Create wrapper
wrapper_noise_pred = NoisePredNetWrapper(model)
wrapper_noise_pred = wrapper_noise_pred.to(device)
wrapper_noise_pred.eval()

torch.onnx.export(
    wrapper_noise_pred,
    (dummy_sample, dummy_timestep, dummy_global_cond),
    onnx_noise_pred,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["sample", "timestep", "global_cond"],
    output_names=["output"],
    dynamic_axes=None,  # Fixed batch size of 8
)

print("converting noise_pred_net to onnx")


onnx_model = onnx.load(onnx_noise_pred)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# # Optional: Test with ONNX Runtime


# print("\nTesting ONNX Runtime...")
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
ort_session = ort.InferenceSession(onnx_noise_pred, providers=providers)
ort_inputs = {
    "sample": dummy_sample.cpu().numpy(),
    "timestep": dummy_timestep.cpu().numpy(),
    "global_cond": dummy_global_cond.cpu().numpy(),
}
ort_outputs = ort_session.run(None, ort_inputs)
print(f"ONNX Runtime output shape: {ort_outputs[0].shape}")

# # Verify outputs match
print(f"\nVerifying outputs match...")
torch_output = test_noise_pred.cpu().numpy()
max_diff = abs(torch_output - ort_outputs[0]).max()
print(f"Maximum difference between PyTorch and ONNX: {max_diff}")

print("---------------------- End of Noise Predictor ---------------------------------- \n")


