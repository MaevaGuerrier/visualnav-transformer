import os
import argparse

import rosbag

def main(args):
    bag_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".bag"):
                bag_files.append(os.path.join(root, file))
    
    lengths = []
    for bag_path in bag_files:
        try:
            b = rosbag.Bag(bag_path)
            length = b.get_end_time() - b.get_start_time()
            lengths.append(length)
        except Exception as e:
            print(f"Error processing {bag_path}: {e}")
        break

    print("Total duration of dataset :", sum(lengths), "seconds",
          "or", sum(lengths)/3600, "hours")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the input directory containing bag files",
        required=True,
    )
    args = parser.parse_args()
    main(args)