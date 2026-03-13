import os
import pickle
import math
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-path",
    "-d",
    type=str,
    help="path of the dataset",
    required=True,
)
parser.add_argument(
    "--output-path",
    "-o",
    type=str,
    help="path to save the histogram (default: waypoint_spacing_histogram.png)",
    default=None,
)

data_path = parser.parse_args().data_path
output_path = parser.parse_args().output_path

# Store all individual waypoint spacings
all_spacings = []
# Store average spacing per trajectory for overall statistics
total_distance = []

for d in os.listdir(data_path):
    traj_path = os.path.join(data_path, d, "traj_data.pkl")
    if not os.path.exists(traj_path):
        continue
    with open(traj_path, "rb") as f:
        traj_data = pickle.load(f)
    if len(traj_data['position']) < 2:
        continue
    waypoints = [tuple(i) for i in list(traj_data['position'])]
    distance_all = 0
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i+1]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        all_spacings.append(distance)
        distance_all += distance
    average_spacing = distance_all / (len(waypoints))
    if average_spacing > 0.02:
        total_distance.append(average_spacing)

print("Average spacing", sum(total_distance)/len(total_distance), "m")
print(f"Total waypoints: {len(all_spacings)}")

if output_path is not None:
    # Save histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_spacings, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('Waypoint Spacing (m)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Waypoint Spacings')
    plt.grid(True, alpha=0.3)
    plt.axvline(sum(total_distance)/len(total_distance), color='r', linestyle='--', 
                label=f'Mean: {sum(total_distance)/len(total_distance):.4f} m')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Histogram saved to {output_path}")