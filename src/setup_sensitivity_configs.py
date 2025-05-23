# %%
# Import python modules
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Now this works because Python sees `src` as a top-level module
from src import analytics, plotting, utils

# %%
import yaml

# %%
# TODO: Read both no_bat and bat results. Make a copy of the config with co2-prices

# %%
BASE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_FOLDER = os.path.join(BASE_FOLDER, "runs")
BATCH_RUNS_FOLDER = os.path.join(RUNS_FOLDER, "batch_runs")

DATA_FOLDER = os.path.join(BASE_FOLDER, "data")
FOR_THE_REPORT_FOLDER = os.path.join(DATA_FOLDER, "for_the_report")
METHODOLOGY_FOLDER = os.path.join(FOR_THE_REPORT_FOLDER, "methodology")
BENCHMARKING_FOLDER = os.path.join(METHODOLOGY_FOLDER, "benchmarking")
os.makedirs(BENCHMARKING_FOLDER, exist_ok=True)

# %%
co2_prices = [80, 100, 120, 140, 160]
emissions_restriction_reduction = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]

# %%
folder_names = [
    "1_May23_Fri_h18_m29_s11-GTSEP_stochastic_v2-128_ES_PT_sv2",
    "2_May23_Fri_h19_m46_s36-GTSEP_stochastic_v2-128_ES_PT_sv2_no_bat",
]
config_names = ["128_sv2_8_weeks_ef05_rf025", "128_sv2_8_weeks_ef05_rf025_no_bat"]

# %%
SINGLE_RUNS_FOLDER = os.path.join(RUNS_FOLDER, "single_runs")

# %% [markdown]
# # Switch to batch runs folder when doing the 128 4week model run

# %%
BATCH_RUNS_FOLDER_BASE = os.path.join(
    BATCH_RUNS_FOLDER, "batch_128_sv2_8_weeks_ef05_rf025_bnb"
)
folders = [
    os.path.join(BATCH_RUNS_FOLDER_BASE, folder_name) for folder_name in folder_names
]
print(folders)

# %% [markdown]
# ## Read data

# %%
dummy_co2_limit = 100

# %%
configs = [utils.load_model_config(config_name) for config_name in config_names]
for config in configs:
    print(config)

# %%
years_list = [config["years"] for config in configs]
data_folder_names = [config["data_folder_name"] for config in configs]

# %%
input_data_folders = [
    os.path.join(DATA_FOLDER, "processed", folder_name)
    for folder_name in data_folder_names
]
input_data_sets = [
    utils.load_multi_year_csv_files_with_week_from_folder(
        years=years, data_folder_path=input_data_folder
    )
    for years, input_data_folder in zip(years_list, input_data_folders)
]
# for i in range(len(input_data_sets)):
#     print(f"config: {i}, {config_names[i]}")
#     print(f"Input data folder: {input_data_folders[i]}")
#     print(f"Input data set: {input_data_sets[i]}")

# %%
decision_variables_folders = [
    os.path.join(folder, "decision_variables") for folder in folders
]
decision_variables_sets = [
    utils.load_csv_files_from_folder_with_scenarios(decision_variables_folder)
    for decision_variables_folder in decision_variables_folders
]

# %%
decision_variables_sets[0].keys()

# %%
# Finally, calculate the CO2 emissions for each scenario, but first I need the scenarios
# scenario_file_path = r"C:\Users\tinus\OneDrive\Dokumenter\0 Master\code\master_project\data\scenario_multipliers\base_scenarios.csv"
scenario_file_path = os.path.join(
    DATA_FOLDER, "scenario_multipliers", "base_scenarios.csv"
)
import pandas as pd

# %%
scenarios_set = []
Omegas = []
week_weights_set = []
configs_after_running = []
for i, config in enumerate(configs):
    scenario_file = config["scenario_file"]
    print(f"Scenario file: {scenario_file}")
    scenario_multiplier = utils.load_scenario_multiplier(scenario_file)

    # Check that years in scenario_multiplier match the years in the data
    scenario_years = scenario_multiplier.index.values.tolist()
    scenarios_list = scenario_multiplier.columns.tolist()
    scenarios = {
        year: [name for name in scenario_multiplier.loc[year].dropna().index]
        for year in scenario_multiplier.index
    }
    scenarios_set.append(scenarios)

    Omegas.append(scenarios_list)
    config_after_running = yaml.safe_load(
        open(os.path.join(folders[i], "model_info", "config.yaml"))
    )
    print(f"Config after running: {config_after_running}")
    configs_after_running.append(config_after_running)
    week_weights = config_after_running["week_weights"]
    week_weights_set.append(week_weights)

    Omega = scenarios_list  # e.g. ['NT','GA','DE']
    Omega_y = scenarios  # e.g. {2040:['NT','GA','DE'],...}

    # Still need Ys, Gs, Ws, and Ts
# for i, scenario in enumerate(scenarios_set):
#     print(f"Scenario: {scenario}")
#     print(f"Omega: {Omegas[i]}")
#     print(week_weights_set[i])

# %%
decision_variables_sets[0].keys()

# %%
import pandas as pd
import os


def compute_total_co2_emissions(
    decision_variables: dict[str, pd.DataFrame],
    generators: pd.DataFrame,
    week_weights: dict[str, float],
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Compute total CO₂ emissions per year and scenario, preserving NaNs
    where no data exists.

    Emissions are calculated as:
      sum_over(i,w,t) [ value(i,ω,y,w,t) * co2_emissions(i,y) * week_weights[w] ]

    Parameters
    ----------
    decision_variables : dict of pd.DataFrame
        Contains 'generation' DataFrame with MultiIndex
        ['generator','scenario','year','week','hour'] and column 'value'.
    generators : pd.DataFrame
        MultiIndexed by ['year','generator'], with column 'co2_emissions'.
    week_weights : dict of str->float or int->float
        Mapping week -> weight.
    savefolder : str or None
        Directory to save CSV. If None, not saved.

    Returns
    -------
    pd.DataFrame
        Indexed by year, columns=scenario, values=emissions (tonnes),
        with NaN for missing year+scenario combinations.
    """
    # Convert week_weights keys to int if needed
    ww_int = {int(k): v for k, v in week_weights.items()}
    # Flatten generation
    gen = (
        decision_variables["generation"]
        .reset_index()
        .rename(columns={"value": "gen_mwh"})
    )
    # Merge CO₂ factors
    meta = generators.reset_index()[["year", "generator", "co2_emissions"]]
    df = gen.merge(meta, on=["year", "generator"], how="left")
    # Map week → weight
    df["weight"] = df["week"].map(ww_int)
    # Compute emissions
    df["emissions_tonnes"] = df["gen_mwh"] * df["co2_emissions"] * df["weight"]
    # Aggregate and pivot without filling NaNs
    table = (
        df.groupby(["year", "scenario"])["emissions_tonnes"]
        .sum()
        .reset_index()
        .pivot(index="year", columns="scenario", values="emissions_tonnes")
    )
    if savefolder:
        table.to_csv(os.path.join(savefolder, "annual_co2_emissions.csv"))
    return table


# %%
co2_emissions_dataframes = []
for i, decision_variables_set in enumerate(decision_variables_sets):
    print(f"Folder name: {folder_names[i]}")
    # print(f"Decision variables: {decision_variables_set}")
    co2_emissions = compute_total_co2_emissions(
        decision_variables=decision_variables_set,
        generators=input_data_sets[i]["generators"],
        week_weights=week_weights_set[i],
        savefolder=BENCHMARKING_FOLDER,
    )
    # Compute row-wise mean across all scenario columns
    co2_emissions["mean"] = co2_emissions.mean(axis=1, skipna=True)
    co2_emissions_dataframes.append(co2_emissions)
    print(co2_emissions)

# %%
input_data_sets[0]["generators"].head()

# %%
co2_emissions_folder = os.path.join(DATA_FOLDER, "co2_emissions_files")
os.makedirs(co2_emissions_folder, exist_ok=True)
for i, co2_emissions in enumerate(co2_emissions_dataframes):
    co2_emissions_path = os.path.join(
        co2_emissions_folder, f"co2_emissions_{folder_names[i]}.csv"
    )
    co2_emissions.to_csv(co2_emissions_path)

# %%
co2_emissions_folder = os.path.join(DATA_FOLDER, "co2_emissions_files")
test_read_dfs = []
co2_emissions_paths = []
os.makedirs(co2_emissions_folder, exist_ok=True)
for i, co2_emissions in enumerate(co2_emissions_dataframes):
    co2_emissions_path = os.path.join(
        co2_emissions_folder, f"co2_emissions_{folder_names[i]}.csv"
    )
    test_read_dfs.append(pd.read_csv(co2_emissions_path, index_col=0))
    co2_emissions_paths.append(co2_emissions_path)

# %%
test1 = test_read_dfs[0]
test1

# %%
years = [2025, 2030, 2040, 2050]
scenarios = ["DE", "GA", "NT"]
# for year in years:
#     for scenario in scenarios:

#         print(
#             f"For year {year} scenario {scenario} the value is {test1.loc[year, scenario]}"
#         )
#         print(type(test1.loc[year, scenario]))

# %% [markdown]
# # Time to make copies of the config files

# %%
for i, name in enumerate(folder_names):
    for num, co2_price in enumerate(co2_prices):
        suffix = f"_co2price_{co2_price}"
        overwrite_dict = {
            "CO2_price": co2_price,
        }
        utils.copy_and_modify_config(
            config_names[i], overwrite_dict=overwrite_dict, suffix=suffix
        )


# %%
for i, name in enumerate(folder_names):
    for num, emission_restriction in enumerate(emissions_restriction_reduction):
        suffix = f"_emissionsreduction_{emission_restriction}"
        co2_emissions_dataframe = co2_emissions_dataframes[i]
        co2_emissions_dataframe_scaled = co2_emissions_dataframe * (
            1 - emission_restriction
        )
        co2_emissions_path = os.path.join(
            co2_emissions_folder, f"co2_emissions_{folder_names[i]}{suffix}.csv"
        )
        co2_emissions_dataframe_scaled.to_csv(co2_emissions_path)
        overwrite_dict = {
            "co2_emissions_path": co2_emissions_path,
        }
        utils.copy_and_modify_config(
            config_names[i], overwrite_dict=overwrite_dict, suffix=suffix
        )

# %%


# %%
