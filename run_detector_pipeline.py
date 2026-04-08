import subprocess
import sys
import os

# ============================================================
# CONFIGURE HERE
# ============================================================
N_RUNS = 3                      # How many times DetectorPipeline.py is run
RUN_PREFIX = "DonutDetector"    # Prefix used for model/results folders

NUM_SIGNAL = 50000               # Number of signal samples to generate
NUM_BACKGROUND = 50000           # Number of background samples to generate

TRAIN_ADV_SIZE = 5000            # Number of adversarial samples for training
VAL_ADV_SIZE = 2500              # Number of adversarial samples for validation
TEST_ADV_SIZE = 2500             # Number of adversarial samples for testing

# Arguments for MakeDonutDummyData.py
MAKE_DATA_ARGS = [
    "--num_signal", str(NUM_SIGNAL),
    "--num_background", str(NUM_BACKGROUND),
]

# Arguments for CrossDetector.py + DetectorPipeline.py
CROSS_DETECTOR_ARGS = [
    "--run_prefix", RUN_PREFIX,
    "--run_num", str(N_RUNS),
]

# Arguments for DetectorPipeline.py
DETECTOR_ARGS = [
    "--run_prefix", RUN_PREFIX,
    "--train_adv_size", str(TRAIN_ADV_SIZE),
    "--val_adv_size", str(VAL_ADV_SIZE),
    "--test_adv_size", str(TEST_ADV_SIZE)
]


# ============================================================

python = sys.executable
scripts_dir = os.path.dirname(os.path.abspath(__file__))

def run(script, args):
    cmd = [python, os.path.join(scripts_dir, script)] + args
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: {script} exited with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)

run("Data/MakeDonutDummyData.py", MAKE_DATA_ARGS)

for i in range(N_RUNS):
    run("Detector/DetectorPipeline.py", ["--runcounter", str(i)] + DETECTOR_ARGS)

run("Detector/CrossDetector.py", CROSS_DETECTOR_ARGS)

print("\nPipeline complete.")