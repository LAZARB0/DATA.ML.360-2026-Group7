import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="ml-1m")
args = parser.parse_args()

for script in ["src/dataset.py", "src/most_popular.py", "src/KNN_model.py"]:
    print(f"\n===== {script} =====", flush=True)
    subprocess.run([sys.executable, script, "--data-dir", args.data_dir])
