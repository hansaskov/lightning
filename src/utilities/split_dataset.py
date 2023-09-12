
import splitfolders

def split_dataset(input_folder, output_folder, ratios=(0.6, 0.20, 0.20), seed=None):
    splitfolders.ratio(input_folder, output=output_folder, seed=seed, ratio=ratios)

# Usage example
input_folder = 'ball_detection/'
output_folder = 'ball_detection2/'
split_dataset(input_folder, output_folder)
