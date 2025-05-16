"""Thise module just runs the analyze_stochasic_run function from analytics.py for the folder specified."""

import os
import yaml
import tqdm

from src.analytics import analyze_run_stochastic

RUNS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs"))
SINGLE_RUNS_FOLDER = os.path.join(RUNS_FOLDER, "single_runs")
BATCH_RUNS_FOLDER = os.path.join(RUNS_FOLDER, "batch_runs")

if __name__ == "__main__":
    print(RUNS_FOLDER)
    print(SINGLE_RUNS_FOLDER)
    print(BATCH_RUNS_FOLDER)

    """ There are two types of runs: single runs and batch runs.
    If you want to analyze on or more single runs, just specify the folder names in the list below.
    If you want to analyze all the runs in a batch, specificy the batch folder name below.
    
    Finally, set the folders to run to either be the single or batch runs.
    """
    single_runs_folder_names = [
        # "XXX4weeks_1core___",
        "XXX4weeks_2core___",
        # "XXX8weeks_1core",
        # "XXX8weeks_2core",
    ]
    single_runs_folder_paths = [
        os.path.join(SINGLE_RUNS_FOLDER, folder_name)
        for folder_name in single_runs_folder_names
    ]
    batch_runs_folder_name = "test_vss_evpi"
    batch_runs_folder_path = os.path.join(BATCH_RUNS_FOLDER, batch_runs_folder_name)

    ### select either single or batch runs
    run_batch = False
    if run_batch:
        print("Running batch runs")
        runs_folder_paths = [
            os.path.join(batch_runs_folder_path, folder_name)
            for folder_name in os.listdir(batch_runs_folder_path)
            if os.path.isdir(os.path.join(batch_runs_folder_path, folder_name))
        ]
    else:
        print("Running single runs")
        runs_folder_paths = single_runs_folder_paths
    print(f"Paths: {runs_folder_paths}")
    for path in tqdm.tqdm(runs_folder_paths, desc="Running analytics"):
        model_config_path = os.path.join(path, "model_info", "config.yaml")
        print(f"model_config_path: {model_config_path}")
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        print(f"model config save folder: {model_config["save_folder"]}")
        old_save_folder = model_config["save_folder"]
        old_save_folder_name = os.path.basename(old_save_folder)
        if old_save_folder_name != os.path.basename(path):
            print(
                f"Warning: save folder name {old_save_folder_name} does not match run folder name {os.path.basename(path)}. Check for error. Saving results to {os.path.basename(path)}"
            )
        new_save_folder = os.path.join(path)
        print(f"Overwriting save folder from {old_save_folder} to {new_save_folder}")
        model_config["save_folder"] = new_save_folder
        analyze_run_stochastic(model_config=model_config)
