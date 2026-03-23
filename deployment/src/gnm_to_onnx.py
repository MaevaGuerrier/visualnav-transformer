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

model_name = "gnm"


"""
NOTE: You might encounter issues with laoding gnm for python version 3.10 or higher.
To solve the issue do not load the model on cuda device.
ERROR TRACE:
Loading model from ../model_weights/gnm.pth
/usr/local/lib/python3.10/dist-packages/torchvision/ops/misc.py:120: UserWarning: Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead.
  warnings.warn(
corrupted size vs. prev_size
Fatal Python error: Aborted

Current thread 0x0000ffffa3279860 (most recent call first):
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 134 in __init__
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 447 in __init__
  File "/usr/local/lib/python3.10/dist-packages/torchvision/ops/misc.py", line 97 in __init__
  File "/usr/local/lib/python3.10/dist-packages/torchvision/ops/misc.py", line 159 in __init__
  File "/usr/local/lib/python3.10/dist-packages/torchvision/models/mobilenetv2.py", line 38 in __init__
  File "/workspace/.packages_nomad_ros2/vint_train/models/gnm/modified_mobilenetv2.py", line 90 in __init__
  File "/workspace/.packages_nomad_ros2/vint_train/models/gnm/gnm.py", line 29 in __init__
  File "/workspace/src/visualnav-transformer/deployment/src/utils.py", line 71 in load_model
  File "/workspace/src/visualnav-transformer/deployment/src/gnm_to_onnx.py", line 37 in <module>

Extension modules: numpy.core._multiarray_umath, numpy.core._multiarray_tests, numpy.linalg._umath_linalg, numpy.fft._pocketfft_internal, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._mt19937, numpy.random.mtrand, numpy.random._philox, numpy.random._pcg64, numpy.random._sfc64, numpy.random._generator, torch._C, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special, yaml._yaml, PIL._imaging, PIL._imagingft, google.protobuf.pyext._message (total: 24)
Aborted (core dumped)
"""
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")




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


output_path = "/workspace/src/visualnav-transformer/deployment/model_weights/gnm.onnx"

print("Testing forward pass for gnm ...")
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
"""
NOTE: A similar issue with onnxruntime and python 3.10 or higher might occur.
To solve the issue, use CPUExecutionProvider only.
ERROR TRACE:
Current thread 0x0000ffff80e8b860 (most recent call first):
  File "/usr/local/lib/python3.10/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 491 in _create_inference_session
  File "/usr/local/lib/python3.10/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 419 in __init__
  File "/workspace/src/visualnav-transformer/deployment/src/gnm_to_onnx.py", line 98 in <module>

Extension modules: numpy.core._multiarray_umath, numpy.core._multiarray_tests, numpy.linalg._umath_linalg, numpy.fft._pocketfft_internal, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._mt19937, numpy.random.mtrand, numpy.random._philox, numpy.random._pcg64, numpy.random._sfc64, numpy.random._generator, torch._C, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special, yaml._yaml, PIL._imaging, PIL._imagingft, google.protobuf.pyext._message (total: 24)
Aborted (core dumped)
"""
# providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
providers = ["CPUExecutionProvider"]
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
