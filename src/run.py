import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from time import time
import os
import yaml
import argparse
from src.analytics import analyze_run
from src.utils import load_model_config
from src.preprocessing import run_preprocessing
from src.models import get_model

RUNS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs"))


def save_model(model, model_config):
    save_folder = model_config["save_folder"]
    model_info_save_folder = os.path.join(save_folder, "model_info")
    model_save_path = os.path.join(model_info_save_folder, "model" + ".mps")
    solution_save_path = os.path.join(model_info_save_folder, "solution" + ".sol")
    config_dump_save_path = os.path.join(model_info_save_folder, "config" + ".yaml")
    model.write(model_save_path)
    model.write(solution_save_path)
    with open(config_dump_save_path, "w") as file:
        yaml.dump(model_config, file)


def run(
    model_config_name: str = "",
    preprocessing_config_name: str = "",
    batch_number: bool = False, batch_folder_name = None
):
    if preprocessing_config_name:
        preprocessing_config = run_preprocessing(preprocessing_config_name)
        print(preprocessing_config)
    else:
        print(f"No preprocessing config provided. Skipping preprocessing...")

    # Load model configuration
    model_config = load_model_config(model_config_name)
    print(model_config)

    if not batch_number:
        base_folder = os.path.join(RUNS_FOLDER, "single_runs")
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
    else:
        base_folder = os.path.join(RUNS_FOLDER, "batch_runs")
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        base_folder = os.path.join(base_folder, batch_folder_name)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
    run_id = create_run_id(model_config)
    if batch_number:
        run_id = f"{batch_number}_{run_id}"
    save_folder = os.path.join(base_folder, run_id)
    model_config["run_id"] = run_id
    model_config["save_folder"] = save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model, model_build_time, model_run_time = get_model(model_config)

    model_info_save_folder = os.path.join(save_folder, "model_info")
    if not os.path.exists(model_info_save_folder):
        os.makedirs(model_info_save_folder)
    # Save high level model information
    # Save basic model information
    model_info = {
        "Objective Value": model.ObjVal if model.Status == GRB.OPTIMAL else None,
        "Optimality Gap (%)": model.MIPGap * 100 if model.IsMIP else None,
        "Runtime (s)": model.Runtime,
        "Build Time (s)": model_build_time,
        "Optimize Time (s)": model_run_time,
        "Total Time (s)": model_build_time + model_run_time,
        "Number of Variables": model.NumVars,
        "Number of Constraints": model.NumConstrs,
        "Number of Nonzeros": model.NumNZs,
        "Number of Integer Variables": model.NumIntVars,  # integer variables (includes both integer and binary)
        "Number of Binary Variables": model.NumBinVars,    # binary variables specifically
        "Number of Quadratic Constraints": model.NumQConstrs,
        "Model Status": model.Status,
    }

    # Convert to DataFrame for easier export
    model_info_df = pd.DataFrame([model_info])
    model_info_df.to_csv(
        os.path.join(model_info_save_folder, "model_info.csv"), index=False
    )

    # Save model
    save_model(model, model_config)

    print("end")
    analyze_run(model_config)


def create_run_id(model_config: dict) -> str:
    from datetime import datetime

    now = datetime.now()
    formatted = now.strftime("%b%d:%a:h%H")
    model_name = model_config["model_name"]
    model_id = model_config["model_id"]
    now = datetime.now()
    formatted = now.strftime("%b%d_%a_h%H")
    if not model_id:
        run_id = f"{formatted}-{model_name}"
    else:
        run_id = f"{formatted}-{model_name}-{model_id}"
    return run_id


# if __name__ == "__main__":
#     model_config_name = "small"
#     run(model_config_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model with inputs model_config_name and preprocessing_config_name."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="The model config name.",
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        default="",
        help="The preprocessing config name.",
    )
    args = parser.parse_args()
    run(args.name)