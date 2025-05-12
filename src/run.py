import re
import shutil
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, tupledict
from time import time
import os
import tqdm
import yaml
import argparse
from src import utils
from src.analytics import analyze_run_stochastic
from src.utils import (
    load_csv_files_from_folder_multi_weeks,
    load_model_config,
    load_multi_year_csv_files_with_week_from_folder,
)
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
    print(f"Model saved to: {save_folder}")


def rerun_co2(model, model_config):
    # calculate CO2 emissions from previous run
    save_folder = model_config["save_folder"]
    data_folder_name = model_config["data_folder_name"]
    data_folder = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", data_folder_name
    )
    print(f"Data folder: {data_folder}")
    years = model_config["years"]
    weeks = model_config["representative_periods"]
    week_weights = model_config["week_weights"]
    decision_variables_folder = os.path.join(save_folder, "decision_variables")

    input_data = load_multi_year_csv_files_with_week_from_folder(
        years, weeks, data_folder
    )
    generators = input_data["generators"]
    decision_variables = load_csv_files_from_folder_multi_weeks(
        decision_variables_folder
    )
    generation = decision_variables["generation"]
    # print(f"Generation head: {generation.head()}")
    # # print(f"Model.getVars(): {model.getVars()}")

    # print("--------------------")
    # print("----------------")

    g_vars = {}
    g_new_vars = {}

    for var in model.getVars():
        match = re.match(r"g\[(.+)\]", var.VarName)
        match_new = re.match(r"g_new\[(.+)\]", var.VarName)
        if match:
            indices = tuple(s.strip() for s in match.group(1).split(","))
            g_vars[indices] = var
        if match_new:
            indices = tuple(s.strip() for s in match_new.group(1).split(","))
            g_new_vars[indices] = var
    # print(g_vars)

    # print("@@@@@@@@@")
    # print("@@@@@@@@@")
    # print(f"g_vars index: {g_vars.keys()}")
    # print(f"Type g_vars: {type(g_vars)}")

    co2_emissions = 0
    for i, y, w, h in g_vars.keys():
        y = int(y)
        w = int(w)
        h = int(h)
        produced = generation.loc[(y, w, h, i), "value"]
        emission_rate = generators.loc[(y, i), "co2_emissions"]
        co2_emission = produced * emission_rate
        weighted_co2_emission = co2_emission * week_weights[w]
        co2_emissions += weighted_co2_emission
    print(f"CO2 emissions: {co2_emissions / 1e6} Million tons")
    print(f"CO2 cost: {co2_emissions * model_config['CO2_price'] / 1e6} Million $")

    new_co2_limit = co2_emissions - 100
    print(f"Setting new co2 limit: {new_co2_limit}")

    def extract_vars(model, base_name):
        var_dict = {}
        for var in model.getVars():
            match = re.match(rf"{base_name}\[(.+)\]", var.VarName)
            if match:
                indices = tuple(
                    int(x) if x.strip().isdigit() else x.strip()
                    for x in match.group(1).split(",")
                )
                var_dict[indices] = var
        return tupledict(var_dict)

    # Usage
    g = extract_vars(model, "g")
    g_new = extract_vars(model, "g_new")

    # print(f"Type g: {type(g)}")
    # print(f"g index: {g.keys()}")

    print(f"Old objective: {model.ObjVal / 1e9} Billion $")
    # Overwrite old emission constraints with new one
    constr = model.getConstrByName("C_emission_limit")
    if constr is not None:
        model.remove(constr)
        model.update()
        print("Removed old emission constraint")
        model.addConstr(
            gp.quicksum(
                week_weights[int(w)]
                * (g[i, int(y), int(w), int(t)] + g_new[i, int(y), int(w), int(t)])
                * generators.loc[(int(y), i), "co2_emissions"]
                for (i, y, w, t) in g.keys()
            )
            <= new_co2_limit,
            name="C_emission_limit",
        )

    else:
        print("No emission constraint found")

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        print("Model is not optimal. Exiting...")
        return
    else:

        print(f"New objective: {model.ObjVal / 1e9} Billion $")

        total_emissions = sum(
            week_weights[int(w)]
            * (g[i, int(y), int(w), int(t)].X + g_new[i, int(y), int(w), int(t)].X)
            * generators.loc[(int(y), i), "co2_emissions"]
            for (i, y, w, t) in g.keys()
        )
        print(f"Total CO2 emissions: {total_emissions / 1e6:.2f} million tons")
        emission_dual = model.getConstrByName("C_emission_limit").Pi
        print(f"Emission dual: {emission_dual}")

    # Overwrite dual for emissions constraint
    dual_variables_folder = os.path.join(save_folder, "dual_variables")
    # 5. Emissions constraint dual (single value)
    try:
        emissions_dual = model.getConstrByName("C_emission_limit").Pi
        emissions_dual_df = pd.DataFrame(
            [("all", "all", "all", "total", emissions_dual)],
            columns=["year", "week", "hour", "scope", "dual_value"],
        )
        emissions_dual_df.to_csv(
            os.path.join(dual_variables_folder, "emissions_dual.csv"), index=False
        )
    except gp.GurobiError:
        print("Warning: Emission limit constraint not active or missing.")


def run(
    model_config_name: str = "",
    preprocessing_config_name: str = "",
    batch_number: bool = False,
    batch_folder_name=None,
    no_run_analysis: bool = False,
    co2_rerun: bool = False,
    is_sub_run: bool = False,
    is_EV_problem: bool = False,
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
    if not model_config.get("save_folder", ""):
        run_id = create_run_id(model_config)
        if batch_number:
            run_id = f"{batch_number}_{run_id}"
        save_folder = os.path.join(base_folder, run_id)
        model_config["run_id"] = run_id
        model_config["save_folder"] = save_folder
    else:
        save_folder = model_config["save_folder"]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_output = get_model(model_config)
    if len(model_output) == 3:
        model, model_build_time, model_run_time = model_output
    elif len(model_output) == 4:
        model, model_build_time, model_run_time, week_weights = model_output
    else:
        raise ValueError(
            "Model output is not in the expected format. Model output is {}.".format(
                model_output
            )
        )

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
        "Number of Binary Variables": model.NumBinVars,  # binary variables specifically
        "Number of Quadratic Constraints": model.NumQConstrs,
        "Model Status": model.Status,
    }

    # Convert to DataFrame for easier export
    model_info_df = pd.DataFrame([model_info])

    if "week_weights" in locals():
        model_config["week_weights"] = week_weights
    # Save model
    if not is_sub_run:
        model_info_df.to_csv(
            os.path.join(model_info_save_folder, "model_info.csv"), index=False
        )
        save_model(model, model_config)
    else:  # this is a sub run, so we just return the model_info_df
        save_model(model, model_config)
        return model_info_df

    if co2_rerun:
        rerun_co2(model, model_config)
    model.dispose()

    if not no_run_analysis:
        analyze_run_stochastic(model_config)

    EVPI = model_config.get("EVPI", False)
    VSS = model_config.get("VSS", False)
    if EVPI:
        results_folder = os.path.join(save_folder, "results")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        # Save model info
        EVPI_runs_folder = os.path.join(results_folder, "EVPI_runs")
        if not os.path.exists(EVPI_runs_folder):
            os.makedirs(EVPI_runs_folder)
        for i in tqdm.tqdm(range(1, 7), desc="Running EVPI analysis"):
            print(f"Running EVPI analysis for scenario {i}...")
            # Make a folder for the new run
            EVPI_save_folder = model_config["save_folder"] + f"_EVPI_{i}"
            scenario_file = f"EVPI_{i}"
            overwrite_dict = {
                "scenario_file": scenario_file,
                "save_folder": EVPI_save_folder,
                "EVPI": False,
                "VSS": False,
            }
            temp_file_name = f"temp_EVPI_{i}"
            utils.copy_and_modify_config(
                model_config_name,
                overwrite_dict=overwrite_dict,
                new_filename=temp_file_name,
            )
            no_run_analysis = True
            is_sub_run = True

            model_info_df = run(
                model_config_name=temp_file_name,
                no_run_analysis=no_run_analysis,
                is_sub_run=is_sub_run,
            )
            model_info_df.to_csv(
                os.path.join(EVPI_runs_folder, f"model_info_EVPI_{i}.csv"), index=False
            )
            utils.delete_config_file(temp_file_name)
            # Delet results folder
            # Delete the EVPI folder
            if os.path.exists(EVPI_save_folder):
                print(f"Deleting folder: {EVPI_save_folder}")
                shutil.rmtree(EVPI_save_folder)

    if VSS:
        # Calculate EV solution
        scenario_file = "mean"
        old_save_folder = model_config["save_folder"]
        VSS_runs_folder = os.path.join(old_save_folder, "VSS_runs")
        if not os.path.exists(VSS_runs_folder):
            os.makedirs(VSS_runs_folder)
        EV_save_folder = os.path.join(VSS_runs_folder, "EV_run")
        if not os.path.exists(EV_save_folder):
            os.makedirs(EV_save_folder)
        print(f"EV save folder: {EV_save_folder}")
        overwrite_dict = {
            "scenario_file": scenario_file,
            "save_folder": EV_save_folder,
            "EVPI": False,
            "VSS": False,
        }
        temp_file_name = f"temp_EV"
        utils.copy_and_modify_config(
            model_config_name,
            overwrite_dict=overwrite_dict,
            new_filename=temp_file_name,
        )
        no_run_analysis = True
        is_sub_run = True
        print("Running EV model...")
        model_info_df = run(
            model_config_name=temp_file_name,
            no_run_analysis=no_run_analysis,
            is_sub_run=is_sub_run,
        )

        results_folder = os.path.join(old_save_folder, "results")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        # Save model info
        EV_run_folder = os.path.join(results_folder, "EV_run")
        if not os.path.exists(EV_run_folder):
            os.makedirs(EV_run_folder)

        model_info_df.to_csv(
            os.path.join(EV_run_folder, f"model_info_EV.csv"), index=False
        )
        utils.delete_config_file(temp_file_name)

        # Calculate EEV
        investment_decsion_variables = [
            "battery_capacity",
            "branch_capacity",
            "generator_capacity",
        ]
        investment_decision_variables_paths = {
            decision_variable
            + "_path": os.path.join(
                EV_save_folder, "decision_variables", decision_variable + ".csv"
            )
            for decision_variable in investment_decsion_variables
        }
        print(
            f"Investment decision variables paths: {investment_decision_variables_paths}"
        )
        EEV_save_folder = os.path.join(VSS_runs_folder, "EEV")
        overwrite_dict.update(investment_decision_variables_paths)
        overwrite_dict["scenario_file"] = ""
        overwrite_dict["save_folder"] = EEV_save_folder
        print(f"Overwrite dict for EEV: {overwrite_dict}")
        temp_file_name = f"temp_EEV"
        utils.copy_and_modify_config(
            model_config_name,
            overwrite_dict=overwrite_dict,
            new_filename=temp_file_name,
        )
        print("Running EEV model...")
        model_info_df = run(
            model_config_name=temp_file_name,
            no_run_analysis=no_run_analysis,
            is_sub_run=is_sub_run,
        )
        # Save model info
        EEV_run_folder = os.path.join(results_folder, "EEV_run")
        if not os.path.exists(EEV_run_folder):
            os.makedirs(EEV_run_folder)

        model_info_df.to_csv(
            os.path.join(EEV_run_folder, f"model_info_EEV.csv"), index=False
        )
        utils.delete_config_file(temp_file_name)

        # Delete the EV folder and the EEV folder
        # if os.path.exists(EV_save_folder):
        #     print(f"Deleting folder: {EV_save_folder}")
        #     shutil.rmtree(EV_save_folder)
        # if os.path.exists(EEV_save_folder):
        #     print(f"Deleting folder: {EEV_save_folder}")
        #     shutil.rmtree(EEV_save_folder)

    if VSS or EVPI:
        SP_path = os.path.join(
            os.path.dirname(results_folder), "model_info", "model_info.csv"
        )
        SP_df = pd.read_csv(SP_path)
        SP_df["Label"] = "SP"
        rows = []
        rows.append(SP_df)
    if VSS:
        # Read EV
        ev_path = os.path.join(results_folder, "EV_run", "model_info_EV.csv")
        ev_df = pd.read_csv(ev_path)
        ev_df["Label"] = "EV"
        rows.append(ev_df)

        # Read EEV
        eev_path = os.path.join(results_folder, "EEV_run", "model_info_EEV.csv")
        eev_df = pd.read_csv(eev_path)
        eev_df["Label"] = "EEV"
        rows.append(eev_df)

    if EVPI:
        # Read all EVPI runs
        evpi_dir = os.path.join(results_folder, "EVPI_runs")
        evpi_files = [
            f
            for f in os.listdir(evpi_dir)
            if f.startswith("model_info_EVPI_") and f.endswith(".csv")
        ]
        print(len(evpi_files), "EVPI files found")
        evpi_dfs = [pd.read_csv(os.path.join(evpi_dir, f)) for f in evpi_files]
        evpi_all = pd.concat(evpi_dfs, ignore_index=True)

        # Calculate mean and std
        evpi_mean = evpi_all.mean(numeric_only=True)
        evpi_std = evpi_all.std(numeric_only=True)

        evpi_mean["Label"] = "EVPI"
        evpi_std["Label"] = "EVPI std"

        rows.append(evpi_mean.to_frame().T)
        rows.append(evpi_std.to_frame().T)

    # Combine all rows
    result_df = pd.concat(rows, ignore_index=True)
    result_df.set_index("Label", inplace=True)
    result_df
    save_path = os.path.join(
        os.path.dirname(results_folder), "model_info", "model_info_combined.csv"
    )
    save_path
    result_df.to_csv(save_path)

    # Delete old folders
    if VSS:
        shutil.rmtree(os.path.join(results_folder, "EV_run"))
        shutil.rmtree(os.path.join(results_folder, "EEV_run"))
    if EVPI:
        shutil.rmtree(os.path.join(results_folder, "EVPI_runs"))


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
    parser.add_argument(
        "--no_run_analysis",
        action="store_true",
        help="Keep empty to run analysis, use the tag to not run the analysis.",
    )

    # This flag will be False if specified, True otherwise
    parser.add_argument(
        "--co2-rerun",
        dest="co2_rerun",
        action="store_true",
        help="Enable the feature (default is disabled)",
    )

    args = parser.parse_args()
    run(
        model_config_name=args.name,
        preprocessing_config_name=args.preprocessing,
        no_run_analysis=args.no_run_analysis,
        co2_rerun=args.co2_rerun,
    )
