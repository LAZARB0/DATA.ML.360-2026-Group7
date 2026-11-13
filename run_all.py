import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="ml-1m")
args = parser.parse_args()

PART_1 = ["src/load_data.py", "src/most_popular.py", "src/KNN_model.py"]
PART_2 = ["src/markov_model.py"]

for script in PART_2:
    print(f"\n===== {script} =====", flush=True)
    subprocess.run([sys.executable, script, "--data-dir", args.data_dir])
