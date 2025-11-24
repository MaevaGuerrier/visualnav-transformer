import torch
import yaml
import os
from utils import load_model
import onnxruntime as ort
import onnx
import einops
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"

model_name = "vint"

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

# model.eval()

print("loading model")


class NoisePredNetWrapper(nn.Module):
    def __init__(self, nomad_model):
        super().__init__()
        self.noise_pred_net = nomad_model.noise_pred_net

    def forward(self, sample, timestep, global_cond):
        # Call the underlying noise_pred_net directly with named arguments
        return self.noise_pred_net(
            sample=sample, timestep=timestep, global_cond=global_cond
        )


class VisionEncoderWrapper(nn.Module):
    def __init__(self, nomad_model):
        super().__init__()
        self.vision_encoder = nomad_model.vision_encoder

    def forward(self, obs_img, goal_img, input_goal_mask):
        # Call the underlying noise_pred_net directly with named arguments
        return self.vision_encoder(
            obs_img=obs_img, goal_img=goal_img, input_goal_mask=input_goal_mask
        )

class VisionEncoderWrapper_vint(nn.Module):
    def __init__(self, vint_model):
        super().__init__()
        self.obs_encoder = vint_model.obs_encoder
        self.goal_encoder = vint_model.goal_encoder
        self.vision_encoder = vint_model.decoder

    def forward(self, obs_img, goal_img):
        obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
        goal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        return self.vision_encoder(
            obs_img=obs_img, goal_img=goal_img
        )


class DistPredWrapper(nn.Module):
    def __init__(self, nomad_model):
        super().__init__()
        self.dist_pred_net = nomad_model.dist_pred_net

    def forward(self, obsgoal_cond):
        return self.dist_pred_net(obsgoal_cond)

class DistPredWrapper_vint(nn.Module):
    def __init__(self, vint_model):
        super().__init__()
        self.dist_pred_net = vint_model

    def forward(self, obs, goal):
        distance, _ = self.dist_pred_net(obs, goal)
        return distance


# print("---------------------- Start Vision Encoder ----------------------------------")

# dummy_obs = torch.randn(4, 12, 96, 96, device=device)
# dummy_goal = torch.randn(4, 3, 96, 96, device=device)
# dummy_mask = torch.randn(4, device=device)

# vision_wrapper = VisionEncoderWrapper(model)

#vision_wrapper = vision_wrapper.to(device)
#vision_wrapper.eval()
# model.eval()

# output_path = "vint.onnx"
# print("Testing forward pass for vision_encoder ...")
# with torch.no_grad():
#     test_vision_output = model(
#         "vision_encoder",
#         obs_img=dummy_obs,
#         goal_img=dummy_goal,
#         input_goal_mask=dummy_mask,
#     )

#     wrapper_vision_output = vision_wrapper(
#         obs_img=dummy_obs,
#         goal_img=dummy_goal,
#         input_goal_mask=dummy_mask,
#     )

#     print(
#         f"Success forward pass for vision enocder with shapes for model {test_vision_output.shape} and {wrapper_vision_output.shape}"
#     )


# print("\nExporting to vision enocder ONNX...")
# torch.onnx.export(
#     vision_wrapper,
#     (dummy_obs, dummy_goal, dummy_mask),
#     output_path,
#     export_params=True,
#     opset_version=17,
#     do_constant_folding=True,
#     input_names=["obs_img", "goal_img", "input_goal_mask"],
#     output_names=["vision_output"],
#     dynamic_axes=None,  # Fixed batch size of 8
# )


# onnx_model = onnx.load(output_path)
# onnx.checker.check_model(onnx_model)
# print("ONNX model of vision enocder is valid!")

# # Optional: Test with ONNX Runtime


# print("\nTesting vision enocder ONNX Runtime...")
# providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# ort_session = ort.InferenceSession(output_path, providers=providers)
# ort_inputs = {
#     "obs_img": dummy_obs.cpu().numpy(),
#     "goal_img": dummy_goal.cpu().numpy(),
# }
# ort_outputs = ort_session.run(None, ort_inputs)
# print(f"ONNX Runtime output shape: {ort_outputs[0].shape}")


# # Verify outputs match
# print(f"\nVerifying outputs match...")
# wrapper_cpu_output = wrapper_vision_output.cpu().numpy()
# test_vision_cpu_output = test_vision_output.cpu().numpy()
# max_diff_wrapper = abs(wrapper_cpu_output - ort_outputs[0]).max()
# max_diff_model = abs(test_vision_cpu_output - ort_outputs[0]).max()
# print(
#     f"Maximum difference between PyTorch and ONNX: {max_diff_wrapper} & {max_diff_model}"
# )


# print("---------------------- End of Vision Encoder ----------------------------------")


print("------------------------ Distance Pred Network --------------------------------")
print("converting dist pred network to onnx")

dummy_goal = torch.randn(4, 3, 64, 85, device=device)
#dist_wrapper = DistPredWrapper(model)
dist_wrapper = DistPredWrapper_vint(model)
dist_wrapper = dist_wrapper.to(device)
dist_wrapper.eval()
dummy_obs = torch.randn(4, 18, 64, 85, device=device)

print("obs_img shape:", dummy_obs.shape)
print("goal_img shape:", dummy_goal.shape)
obsgoal_img = torch.cat([dummy_obs[:, 3*5:, :, :], dummy_goal], dim=1)
print("obsgoal_img shape:", obsgoal_img.shape)

output_path = "dist_pred_net.onnx"

print("Testing forward pass for dist pred encoder ...")
with torch.no_grad():
    # test_distance_output = model(
    #     "dist_pred_net",
    #     obsgoal_cond=dummy_goalcond,
    # )
    test_distance_output, _ = model(dummy_obs, dummy_goal)

    # wrapper_dist_output = dist_wrapper(
    #     dummy_goalcond,
    # )
    wrapper_dist_output = dist_wrapper(dummy_obs, dummy_goal)

    print(
        f"Success forward pass for distance enocder with shapes for model {test_distance_output.shape} and {wrapper_dist_output.shape}"
    )

print("\nExporting to dist enocder ONNX...")
torch.onnx.export(
    dist_wrapper,
    (dummy_obs, dummy_goal),
    output_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["obs", "goal"],
    output_names=["dists"],
    dynamic_axes={
        'obs' : {0 : 'batch_size'},
        'goal' : {0 : 'batch_size'},
    },  # Dynamic batch size
)


onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("ONNX model of dist enocder is valid!")


print("\nTesting dist enocder ONNX Runtime...")
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
ort_session = ort.InferenceSession(output_path, providers=providers)
ort_inputs = {
    "obs": dummy_obs.cpu().numpy(),
    "goal": dummy_goal.cpu().numpy(),
}
ort_outputs = ort_session.run(None, ort_inputs)
print(f"ONNX Runtime output shape: {ort_outputs[0].shape}")

# Verify outputs match
print(f"\nVerifying outputs match...")
# wrapper_cpu_output = wrapper_dist_output.cpu().numpy()
test_cpu_output = test_distance_output.cpu().numpy()
# max_diff_wrapper = abs(wrapper_cpu_output - ort_outputs[0]).max()
max_diff_model = abs(test_cpu_output - ort_outputs[0]).max()
print(
    f"Maximum difference between PyTorch and ONNX: {max_diff_model}"
)

print("---------------------- End of Dist Encoder ----------------------------------")

# print("------------------------------- noise pred net --------------------------- ")
# batch_size = 8

# # Create dummy inputs matching your model's expected input format
# sequence_length = 8
# input_dim = 2
# encoding_size = 256

# # Create dummy input tensor (batch_size, input_dim, sequence_length) on CUDA
# dummy_input = torch.randn(batch_size, sequence_length, input_dim).to(device)

# # Create dummy global condition (batch_size, encoding_size) on CUDA
# dummy_global_cond = torch.randn(batch_size, encoding_size).to(device)

# # Create dummy timestep (batch_size,) on CUDA
# dummy_timestep = torch.randint(0, 1000, (batch_size,)).float().to(device)

# # Test forward pass first to make sure it works
# print("Testing forward pass...")
# with torch.no_grad():
#     test_output = model(
#         "noise_pred_net",
#         sample=dummy_input,
#         timestep=dummy_timestep,
#         global_cond=dummy_global_cond,
#     )
#     print(f"Forward pass successful! Output shape: {test_output.shape}")


# # Export to ONNX
# output_path = "noise_pred_net.onnx"

# # Create wrapper
# wrapper = NoisePredNetWrapper(model)
# wrapper = wrapper.to(device)
# wrapper.eval()

# print("\nExporting to ONNX...")
# torch.onnx.export(
#     wrapper,
#     (dummy_input, dummy_timestep, dummy_global_cond),
#     output_path,
#     export_params=True,
#     opset_version=17,
#     do_constant_folding=True,
#     input_names=["sample", "timestep", "global_cond"],
#     output_names=["output"],
#     dynamic_axes=None,  # Fixed batch size of 8
# )


# print("converting noise_pred_net to onnx")


# onnx_model = onnx.load(output_path)
# onnx.checker.check_model(onnx_model)
# print("ONNX model is valid!")

# # Optional: Test with ONNX Runtime


# print("\nTesting ONNX Runtime...")
# providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# ort_session = ort.InferenceSession(output_path, providers=providers)
# ort_inputs = {
#     "sample": dummy_input.cpu().numpy(),
#     "timestep": dummy_timestep.cpu().numpy(),
#     "global_cond": dummy_global_cond.cpu().numpy(),
# }
# ort_outputs = ort_session.run(None, ort_inputs)
# print(f"ONNX Runtime output shape: {ort_outputs[0].shape}")


# # Verify outputs match
# print(f"\nVerifying outputs match...")
# torch_output = test_output.cpu().numpy()
# max_diff = abs(torch_output - ort_outputs[0]).max()
# print(f"Maximum difference between PyTorch and ONNX: {max_diff}")
