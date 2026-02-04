import os
import pickle
import math

total_distance = []

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-path",
    "-d",
    type=str,
    help="path of the dataset",
    required=True,
)

data_path = parser.parse_args().data_path

for d in os.listdir(data_path):
    with open(os.path.join(data_path, d, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
    waypoints = [tuple(i) for i in list(traj_data['position'])]  # 航路点坐标列表
    distance_all = 0
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i+1]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distance_all += distance
    average_spacing = distance_all / (len(waypoints))
    # print(average_spacing)
    if average_spacing > 0.02:
        total_distance.append(average_spacing)
    
print("Average spacing", sum(total_distance)/len(total_distance), "m")