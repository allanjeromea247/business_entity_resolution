import subprocess
import sys
from src.config import CONFIG_PATH
import json

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    print("========================================")
    print("ENTITY RESOLUTION PIPELINE START")
    print("========================================")
    
    run_cmd("python train.py")
    run_cmd("python predict.py")
    run_cmd("python validate.py")
    
    print("\n========================================")
    print("ENTITY RESOLUTION PIPELINE COMPLETE")
    print("========================================")
    
    try:
        with open(CONFIG_PATH, 'r') as f:
            th_config = json.load(f)
        print(f"\nBest validation threshold: {th_config['threshold']}")
        print(f"Validation macro F0.5: {th_config['f05']:.4f}")
    except Exception as e:
        pass
        
    print("\nOutput:")
    print("output/matching_results.tsv")
    print("output/candidate_pairs.tsv")
    print("========================================")
