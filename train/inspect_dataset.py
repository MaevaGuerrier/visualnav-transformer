import os
import pickle
import argparse


def main(args: argparse.Namespace):
    # Get the names of the folders in the data directory that contain the file 'traj_data.pkl'
    for dataset_name in os.listdir(args.data_dir):
        if not os.path.isdir(os.path.join(args.data_dir, dataset_name)):
            continue
        print(f"Inspecting dataset {dataset_name}")
        folder_names = [
            f
            for f in os.listdir(os.path.join(args.data_dir, dataset_name))
            if os.path.isdir(os.path.join(args.data_dir, dataset_name, f))
        ]

        # Print some information about the dataset
        print(f"Total number of folders: {len(folder_names)}")
        with_traj = sum(os.path.exists(os.path.join(args.data_dir, dataset_name, f, 'traj_data.pkl')) for f in folder_names)
        print(f"Folders with traj_data.pkl: {with_traj}")
        # no *.jpg
        print(f"Folders without images: {sum(not any(fname.endswith('.jpg') for fname in os.listdir(os.path.join(args.data_dir, dataset_name, f))) for f in folder_names)}")
        # show missing traj_data.pkl folders
        print("Folders missing traj_data.pkl:")
        for f in folder_names:
            if not os.path.exists(os.path.join(args.data_dir, dataset_name, f, 'traj_data.pkl')):
                print(f" - {f}")
        print("Folders without images:")
        for f in folder_names:
            if not any(fname.endswith('.jpg') for fname in os.listdir(os.path.join(args.data_dir, dataset_name, f))):
                print(f" - {f}")
        

        # calculate traj_data stats
        if with_traj == 0:
            continue
        avg_vels = []
        folder_names = sorted(folder_names)
        durations = []
        for folder_name in folder_names:
            traj_data_path = os.path.join(args.data_dir, dataset_name, folder_name, 'traj_data.pkl')
            if not os.path.exists(traj_data_path):
                continue

            with open(traj_data_path, 'rb') as f:
                traj_data = pickle.load(f)
            positions, yaws = traj_data['position'], traj_data['yaw']
            if len(positions.shape) != 2 or positions.shape[1] != 2:
                print(f"Folder {folder_name} has invalid position shape: {positions.shape}")
                continue
            if positions.shape[0] != yaws.shape[0]:
                print(f"Folder {folder_name} has {positions.shape[0]} positions but {yaws.shape[0]} yaws")
            if positions.shape[0] != (len(os.listdir(os.path.join(args.data_dir, dataset_name, folder_name)))-1):
                print(f"Folder {folder_name} has {positions.shape[0]} positions but {len(os.listdir(os.path.join(args.data_dir, dataset_name, folder_name)))-1} images")
            durations.append(positions.shape[0] / 4.0)  # assuming 4Hz sampling rate
            displacements = positions[1:] - positions[:-1]
            distances = (displacements**2).sum(axis=1)**0.5
            total_distance = distances.sum()
            total_time = (positions.shape[0] - 1) / 4  # assuming 4Hz sampling rate
            avg_vel = total_distance / total_time if total_time > 0 else 0.0
            #print(f"Folder {folder_name}: total distance = {total_distance:.4f}, total time = {total_time:.4f}, average velocity = {avg_vel:.4f}")
            avg_vels.append(avg_vel)
        macro_avg_vel = sum(avg_vels) / len(avg_vels)
        print(f"Average velocity across trajectories: {macro_avg_vel:.4f}")
        print(f"Total duration of all trajectories: {sum(durations) / 3600:.2f} hours")
        print()

if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", "-i", type=str, help="Path to the dataset directory", required=True
    )
    args = parser.parse_args()
    main(args)