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

model.eval()

print("loading model")


# This can be done more easily than Nomad because forward returns both distance and action predictions
# Whereas in Nomad its calling different sub-networks separately, with their own forward methods

print("------------------------ Distance Pred Network --------------------------------")
print("converting dist pred network to onnx")

dummy_goal = torch.randn(4, 3, 64, 85, device=device)
dummy_obs = torch.randn(4, 18, 64, 85, device=device)


output_path = "vint.onnx"

print("Testing forward pass for vint ...")
with torch.no_grad():

    test_distance_output, test_action_output = model(dummy_obs, dummy_goal)

    print(
        f"Success forward pass for distance enocder with shapes for model {test_distance_output} and { test_action_output}"
    )

print("\nExporting to dist enocder ONNX...")
torch.onnx.export(
    model,
    (dummy_obs, dummy_goal),
    output_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["obs_img", "goal_img"],
    output_names=["distances", "waypoints"],
    dynamic_axes={
        "obs_img": {0: "batch_size"},
        "goal_img": {0: "batch_size"},
        "distances": {0: "batch_size"},      # Add this
        "waypoints": {0: "batch_size"},      # Add this
    },
)


onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("ONNX model of dist enocder is valid!")


print("\nTesting dist enocder ONNX Runtime...")
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
ort_session = ort.InferenceSession(output_path, providers=providers)
ort_inputs = {
    "obs_img": dummy_obs.cpu().numpy(),
    "goal_img": dummy_goal.cpu().numpy(),
}
ort_outputs = ort_session.run(None, ort_inputs)
print(ort_outputs)
print(f"ONNX Runtime output shape: {ort_outputs[0].shape} and {ort_outputs[1].shape}")

# Verify outputs match
print(f"\nVerifying outputs match...")
# wrapper_cpu_output = wrapper_dist_output.cpu().numpy()
test_cpu_distnace_output = test_distance_output.cpu().numpy()
test_cpu_waypoint_output = test_action_output.cpu().numpy()
# max_diff_wrapper = abs(wrapper_cpu_output - ort_outputs[0]).max()
max_diff_model_distance = abs(test_cpu_distnace_output - ort_outputs[0]).max()
max_diff_action_distance = abs(test_cpu_waypoint_output - ort_outputs[1]).max()
print(f"Maximum difference between for distance PyTorch and ONNX: {max_diff_model_distance}")
print(f"Maximum difference between for action PyTorch and ONNX: {max_diff_action_distance}")

print("---------------------- End of Dist Encoder ----------------------------------")
