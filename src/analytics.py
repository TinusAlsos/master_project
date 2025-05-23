"""This module contains functions for running automated analytics on the data in different stages of the pipeline."""

import os

import numpy as np
import pandas as pd
import src.plotting as plotting
import src.utils as utils
import matplotlib.pyplot as plt
import gurobipy as gp

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, "processed")
# Color mapping for scenarios (reuse for other plots)
scenario_colors = {
    "NT": "#1f77b4",
    "GA": "#ff7f0e",
    "DE": "#2ca02c",
    "Average": "#7f7f7f",
}

# Define a consistent color palette for cost categories
category_colors = {
    "Invest Gen": "#b20000",  # dark red
    "Invest Bat": "#0072b2",  # blue
    "Invest Tx": "#009e73",  # green
    "Prod Cost": "#f0e442",  # yellow
    "CO₂ Cost": "#d55e00",  # orange
    "Load Shedding Cost": "#cc79a7",  # magenta
    "Curtailment Cost": "#56b4e9",  # light blue
}


def run_analytics_on_input_data(
    data_folder_name: str,
    SAVE_FIGURES: bool = True,
    SAVE_TABLES: bool = True,
    show_plots: bool = False,
):
    """
    Run analytics on the input data.
    """
    print("Running analytics on input data...")
    if not show_plots:
        original_show = plt.show

        # Override plt.show with a no-op lambda.
        plt.show = lambda: None

    print("Processed data folder:", PROCESSED_DATA_FOLDER)
    data_folder = os.path.join(PROCESSED_DATA_FOLDER, data_folder_name)
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")
    output_folder = os.path.join(data_folder, "analyzed_data")
    if not os.path.exists(output_folder):
        if SAVE_FIGURES or SAVE_TABLES:
            os.makedirs(output_folder)
    demand_output_folder = (
        None if not SAVE_FIGURES else os.path.join(output_folder, "demand")
    )
    if not os.path.exists(demand_output_folder):
        if SAVE_FIGURES or SAVE_TABLES:
            os.makedirs(demand_output_folder)
    capacity_factors_output_folder = (
        None if not SAVE_FIGURES else os.path.join(output_folder, "capacity_factors")
    )
    if not os.path.exists(capacity_factors_output_folder):
        if SAVE_FIGURES or SAVE_TABLES:
            os.makedirs(capacity_factors_output_folder)

    print("Data folder:", data_folder)
    # Load data
    (
        batteries,
        branches,
        capacity_factors,
        generators,
        generator_costs,
        hourly_demand,
        nodes,
    ) = utils.load_csv_files_from_folder(data_folder).values()

    ### Grid Overview ###
    savefolder = None if not SAVE_FIGURES else output_folder
    table_savefolder = None if not SAVE_TABLES else output_folder
    plotting.plot_buses_and_lines(nodes, branches, savefolder=savefolder)
    plotting.plot_base_network_with_lineIDs_and_city_text(
        nodes, branches, savefolder=savefolder
    )
    plotting.plot_sized_lines_with_extensions(nodes, branches, savefolder=savefolder)
    plotting.plot_sized_generators_and_lines(
        nodes, branches, generators, savefolder=savefolder
    )
    ### Transmission Lines
    plotting.plot_sized_branches(nodes, branches, savefolder=savefolder)
    plotting.get_branches_overview_table(branches, savefolder=table_savefolder)
    ### Demand
    plotting.plot_demand_network_daily(
        nodes, hourly_demand, savefolder=demand_output_folder
    )
    plotting.plot_demand_network_hourly(
        nodes, hourly_demand, savefolder=demand_output_folder
    )
    plotting.plot_normalized_hourly_load_by_country(
        nodes, hourly_demand, savefolder=demand_output_folder
    )
    plotting.plot_aggregated_average_hourly_demand_with_stds(
        hourly_demand, savefolder=demand_output_folder
    )
    plotting.plot_average_hourly_demand_each_month_aggragated(
        hourly_demand, savefolder=demand_output_folder
    )
    plotting.plot_average_hourly_demand_each_month_at_node(
        hourly_demand,
        nodes.index.values[0],
        savefolder=demand_output_folder,
    )
    plotting.plot_average_hourly_demand_each_season_aggragated(
        hourly_demand, savefolder=demand_output_folder
    )
    ### Generators Tables
    plotting.get_generators_overview_table(generators, savefolder=table_savefolder)
    plotting.get_generators_cost_and_emissions_table(
        generators, savefolder=table_savefolder
    )
    ### Generators Plots
    plotting.plot_sized_generators(nodes, branches, generators, savefolder=savefolder)
    plotting.plot_installed_capacity_bar_chart(nodes, generators, savefolder=savefolder)
    plotting.plot_effective_capacity_generators_bar_chart(
        nodes,
        generators,
        capacity_factors,
        savefolder=savefolder,
    )
    ### Batteries
    plotting.plot_battery_cases(nodes, savefolder=savefolder)
    ### Capacity Factors
    plotting.plot_correlation_matrix_carrier_by_carrier(
        capacity_factors, savefolder=capacity_factors_output_folder
    )
    # Plot combined hourly and monthly averages.
    plotting.plot_avg_hourly_capacity_factors(
        capacity_factors, generators, output_folder=capacity_factors_output_folder
    )
    plotting.plot_avg_monthly_capacity_factors(
        capacity_factors, generators, output_folder=capacity_factors_output_folder
    )
    # Plot separate hourly and monthly figures for each generator type.
    plotting.plot_avg_hourly_capacity_factors_sep(
        capacity_factors, generators, output_folder=capacity_factors_output_folder
    )
    plotting.plot_avg_monthly_capacity_factors_sep(
        capacity_factors, generators, output_folder=capacity_factors_output_folder
    )
    # Plot heatmaps and weekly averages.
    plotting.plot_heatmap_capacity_factors(
        capacity_factors,
        use_global_limits=False,
        output_folder=capacity_factors_output_folder,
    )
    plotting.plot_avg_weekly_capacity_factors(
        capacity_factors,
        use_global_limits=False,
        output_folder=capacity_factors_output_folder,
    )
    # Plot time series (combined and separate).
    plotting.plot_timeseries_capacity_factors(
        capacity_factors,
        generators,
        use_global_limits=False,
        output_folder=capacity_factors_output_folder,
    )
    plotting.plot_timeseries_capacity_factors_sep(
        capacity_factors,
        generators,
        use_global_limits=False,
        output_folder=capacity_factors_output_folder,
    )

    ### Above this line, the code is from the first iteration

    if not show_plots:
        # Restore the original show function
        plt.show = original_show

    print("Input data analytics completed.")


def tables_generators_overview(
    generators: pd.DataFrame,
    generation: pd.DataFrame,
    capacity_factors: pd.DataFrame,
    model_config: dict,
    savefolder: str,
):
    built_generators_by_carrier = (
        generators[(generators["new"] == 1) & (generators["exists"] == 1)]
        .groupby("carrier")
        .size()
    )
    potential_generators_by_carrier_new = (
        generators[generators["new"] == 1].groupby("carrier").size()
    )
    capacity_by_carrier_new = (
        generators[(generators["new"] == 1) & (generators["exists"] == 1)]
        .groupby("carrier")["new_capacity"]
        .sum()
    )

    for carrier in generators["carrier"].unique():
        if carrier not in capacity_by_carrier_new.index:
            capacity_by_carrier_new[carrier] = 0.0

    capacity_by_carrier_old = (
        generators[(generators["new"] == 0) & (generators["exists"] == 1)]
        .groupby("carrier")["p_nom"]
        .sum()
    )

    capacity_by_carrier_total = capacity_by_carrier_old + capacity_by_carrier_new
    generators["potential_capacity"] = (
        generators["p_nom"] * model_config["expansion_factor"]
    )

    # Group by carrier and calculate the sum of potential_capacity
    potential_capacity_by_carrier_new = (
        generators[generators["new"] == 1]
        .groupby("carrier")["potential_capacity"]
        .sum()
    )
    combined_df = pd.concat(
        [
            built_generators_by_carrier.rename("Built Generators"),
            potential_generators_by_carrier_new.rename("Potential Generators"),
            capacity_by_carrier_new.rename("Capacity"),
            potential_capacity_by_carrier_new.rename("Potential Capacity"),
        ],
        axis=1,
    )

    # Filter the generators dataframe to include only existing generators
    existing_generators = generators[generators["exists"] == 1]

    # Calculate total actual production for each carrier type
    total_production_by_carrier = {}

    for carrier in existing_generators["carrier"].unique():
        # Select columns from the generation dataframe that match the carrier
        carrier_columns = [
            col
            for col in generation.columns[:-1]
            if col.endswith(carrier) or col.split(" ")[-2].endswith(carrier)
        ]
        # Sum the production for all generators of the carrier type
        total_production_by_carrier[carrier] = generation[carrier_columns].sum().sum()

    # Convert the production dictionary to a pandas Series
    total_production_by_carrier = pd.Series(
        total_production_by_carrier, name="total_production"
    )

    # Calculate the utilization rate for each carrier
    # Multiply capacity by 8760 (hours in a year) to get the maximum possible production
    utilization_rate = (
        total_production_by_carrier / (capacity_by_carrier_total * 8760)
    ) * 100

    # Display the utilization rate as a percentage
    utilization_rate = utilization_rate.rename("utilization_rate (%)")

    # Calculate theoretical max utilization
    max_utilization_rates = {}
    for carrier in generators["carrier"].unique():
        columns = [gen for gen in capacity_factors.columns if carrier in gen]
        max_production_existing = (
            capacity_factors[columns].mean() * generators.loc[columns, "p_nom"]
        ).sum()
        max_production_wo_capacity_factor_existing = generators.loc[
            columns, "p_nom"
        ].sum()
        new_columns = [col + " new" for col in columns]
        max_production_new = (
            capacity_factors[columns].mean().values
            * generators.loc[new_columns, "new_capacity"].values
        ).sum()
        max_production_wo_capacity_factor_new = generators.loc[
            new_columns, "new_capacity"
        ].sum()
        total_max_production = max_production_existing + max_production_new
        total_max_production_wo_capacity_factor = (
            max_production_wo_capacity_factor_existing
            + max_production_wo_capacity_factor_new
        )
        max_utilization_rates[carrier] = (
            total_max_production / total_max_production_wo_capacity_factor
        )

        # convert to series
    max_utilization_rates = pd.Series(
        max_utilization_rates, name="max_utilization_rate"
    )
    utilization_rate_df = pd.concat([utilization_rate, max_utilization_rates], axis=1)
    utilization_rate_df["max_utilization_rate"] = (
        utilization_rate_df["max_utilization_rate"] * 100
    )
    utilization_rate_df.columns = ["Utilization rate (%)", "Max utilization rate (%)"]
    utilization_rate_df.round(2)
    generators_results = pd.concat([combined_df, utilization_rate_df], axis=1)

    # Display the combined dataframe
    if savefolder:
        generators_results.to_csv(
            os.path.join(savefolder, "table_generators_overview.csv")
        )


def preprocess_branches(
    branches: pd.DataFrame, branch_build: pd.DataFrame, branch_capacity: pd.DataFrame
) -> pd.DataFrame:
    # Create new branches
    # Add a new column 'exists' to the original branches dataframe and set it to 1
    branches.index = branches.index.astype(str)
    branches["exists"] = 1
    # Create a copy of the dataframe for the "new" branches
    branches_new = branches.copy()
    # Update the index by appending " new" to the original index
    branches_new.index = branches_new.index.astype(str) + " new"
    # Set the 'exists' column to 0 for the new branches
    branches_new["exists"] = 0
    # Concatenate the original dataframe and the new dataframe
    branches = pd.concat([branches, branches_new])
    branches["new"] = branches.index.str.endswith("new").astype(int)

    # Make sure a branch is built if its new capacity is 0
    new_branches = pd.merge(
        branch_build, branch_capacity, left_index=True, right_index=True
    )
    # rename columns
    column_names = ["built", "capacity"]
    new_branches.columns = column_names
    # Update 'built' to 0 where 'capacity' is 0
    new_branches.loc[new_branches["capacity"] == 0, "built"] = 0
    branches.loc[branch_build.index.values, "exists"] = new_branches["built"]
    branches["new_capacity"] = 0.0
    branches.loc[branch_capacity.index.values, "new_capacity"] = new_branches[
        "capacity"
    ]
    return branches


def preprocess_generators(
    generators: pd.DataFrame,
    generator_build: pd.DataFrame,
    generator_capacity: pd.DataFrame,
) -> pd.DataFrame:
    # Add a new column 'exists' to the original dataframe and set it to 1
    generators["exists"] = 1
    # Create a copy of the dataframe for the "new" generators
    generators_new = generators.copy()
    # Update the index by appending " new" to the original index
    generators_new.index = generators_new.index + " new"
    # Set the 'exists' column to 0 for the new generators
    generators_new["exists"] = 0
    # Concatenate the original dataframe and the new dataframe
    generators = pd.concat([generators, generators_new])
    generators["new"] = generators.index.str.endswith("new").astype(int)
    # Make sure a generator is built if its new capacity is 0
    new_generators = pd.merge(
        generator_build, generator_capacity, left_index=True, right_index=True
    )
    # rename columns
    column_names = ["built", "capacity"]
    new_generators.columns = column_names
    # Update 'built' to 0 where 'capacity' is 0
    new_generators.loc[new_generators["capacity"] == 0, "built"] = 0
    generators.loc[generator_build.index.values, "exists"] = new_generators["built"]
    generators["new_capacity"] = 0.0
    generators.loc[generator_capacity.index.values, "new_capacity"] = new_generators[
        "capacity"
    ]

    return generators


def preprocess_batteries(
    batteries: pd.DataFrame,
    battery_build: pd.DataFrame,
    battery_charging: pd.DataFrame,
    battery_discharging: pd.DataFrame,
    battery_soc: pd.DataFrame,
) -> None:
    old_battery_index_to_new = {
        battery: plotting.node_to_city[battery[:-4]] for battery in batteries.index
    }
    # 1. Rename the index in `batteries`
    batteries["old_index"] = batteries.index
    batteries.rename(index=old_battery_index_to_new, inplace=True)

    # 2. Rename the index in `battery_build`
    battery_build.rename(index=old_battery_index_to_new, inplace=True)

    # 3. Rename the columns in `battery_charging`, `battery_discharging`, and `battery_soc`
    battery_charging.rename(columns=old_battery_index_to_new, inplace=True)
    battery_discharging.rename(columns=old_battery_index_to_new, inplace=True)
    battery_soc.rename(columns=old_battery_index_to_new, inplace=True)
    battery_charging.index = pd.to_datetime(battery_charging.index)
    battery_discharging.index = pd.to_datetime(battery_discharging.index)
    battery_soc.index = pd.to_datetime(battery_soc.index)
    # Make battery_build consistent, whether it consists of binary decision variables or capacity.
    if battery_build.isin([0, 1]).all().all():
        batteries["new_power_capacity"] = (
            batteries["P_discharge_max"]
            * battery_build.loc[batteries.index.values, "value"]
        )
        batteries["new_energy_capacity"] = (
            batteries["new_power_capacity"] * batteries["hour_capacity"]
        )
    else:
        batteries["new_power_capacity"] = battery_build.loc[
            batteries.index.values, "value"
        ]
        batteries["new_energy_capacity"] = (
            batteries["new_power_capacity"] * batteries["hour_capacity"]
        )
        battery_build["value"] = (battery_build["value"] != 0).astype(int)


def check_line_errors(branches: pd.DataFrame, power_flow: pd.DataFrame) -> None:
    # Make sure power flow is + if branch doesn't exist
    epsilon = 1e-6
    for branch in branches[branches["exists"] == 0].index:
        if (
            power_flow[branch].abs().sum() > epsilon
            or (power_flow[branch].abs().sum()) * (-1) < epsilon
        ):
            print(f"WARNING: Branch {branch} has power flow when it doesn't exist")
    errors = 0
    num_lines_with_errors = 0
    for column in power_flow.columns[: len(power_flow.columns) // 2]:
        this_line_error = 0
        for idx in power_flow.index.values:
            if (
                power_flow.loc[idx, column] >= 0
                and power_flow.loc[idx, column + " new"] >= 0
            ):
                continue
            elif (
                power_flow.loc[idx, column] <= 0
                and power_flow.loc[idx, column + " new"] <= 0
            ):
                continue
            else:
                errors += 1
                # print(f"Branch: {column}, timestep = {idx}")
                # print(power_flow.loc[idx, column])
                # print(power_flow.loc[idx, column + " new"])
                # print("")
                this_line_error += 1
        if this_line_error > 0:
            num_lines_with_errors += 1
    if errors > 0:
        print(
            f"WARNING, CHECK FOR Errors: {errors}, the number of times the same branch has different signs in the old and new branches"
        )
        print(f"Number of lines with errors: {num_lines_with_errors}")


def branches_complete_analysis(
    nodes: pd.DataFrame,
    branches: pd.DataFrame,
    power_flow: pd.DataFrame,
    savefolder: str,
):
    num_new_branches_built = sum(branches[branches["new"] == 1]["exists"])
    total_capacity_built = sum(
        branches[(branches["new"] == 1) & (branches["exists"] == 1)]["new_capacity"]
    )
    branch_capacity_old_branches = branches[branches["new"] == 0]["p_max"].sum()
    branch_capacity_new_branches = branches[branches["new"] == 1]["new_capacity"].sum()
    existing_branch_ids = branches[branches["new"] == 0].index.astype(str)
    new_branch_ids = branches[branches["new"] == 1].index.astype(str)
    average_used_capacity_old_branches = (
        power_flow.loc[:, existing_branch_ids].abs().mean(axis=0)
    )
    average_used_capacity_new_branches = (
        power_flow.loc[:, new_branch_ids].abs().mean(axis=0)
    )
    combined_branches = branches.copy(deep=True)
    new_indexes = branches[
        [
            True if endswith == True else False
            for endswith in branches.index.str.endswith("new")
        ]
    ].index
    # print(new_indexes)
    # Keep only the new branches
    combined_branches = combined_branches.loc[new_indexes]
    combined_branches["p_max"] = (
        combined_branches["p_max"] + combined_branches["new_capacity"]
    )
    combined_branches["add_on"] = np.where(combined_branches["new_capacity"] > 0, 1, 0)
    combined_branches.index = combined_branches.index.str.replace(" new", "")

    # Normalize column names in power_flow by removing " new" to group new and existing flows
    normalized_columns = power_flow.columns.str.replace(" new", "", regex=False)

    # Group columns by their normalized names and sum them
    aggregated_power_flow = power_flow.T.groupby(normalized_columns).sum().T
    aggregated_power_flow = aggregated_power_flow[
        [str(branch) for branch in branches.index if not str(branch).endswith("new")]
    ]
    # Calculate congestion rate for each branch
    p_max_normalized = branches.set_index(branches.index.astype(str))["p_max"].reindex(
        normalized_columns.unique()
    )
    # Calculate the congestion, congestion limit sets the threshold for calling it a congestion
    congestion_limit = 0.98
    congestion = aggregated_power_flow.abs() > (congestion_limit * p_max_normalized)
    # Calculate the congestion rate: number of congested timesteps / total timesteps
    congestion_rate = congestion.sum() / len(aggregated_power_flow)

    # Create a DataFrame for results
    congestion_rate_df = pd.DataFrame({"congestion_rate": congestion_rate})
    absolute_aggregated_power_flow = aggregated_power_flow.abs()
    average_used_capacity_combined_branches = absolute_aggregated_power_flow.mean(
        axis=0
    )
    # Creating DataFrame with lowercase column names and no rounding
    branches_overview = pd.DataFrame(
        [
            {
                "new_branches": num_new_branches_built,
                "new_capacity_mw": total_capacity_built,
                "util_old_%": (
                    sum(average_used_capacity_old_branches)
                    / branch_capacity_old_branches
                )
                * 100,
                "util_new_%": (
                    (
                        sum(average_used_capacity_new_branches)
                        / branch_capacity_new_branches
                    )
                    * 100
                    if branch_capacity_new_branches > 0
                    else None
                ),
                "util_combined_%": (
                    average_used_capacity_combined_branches.sum()
                    / combined_branches["p_max"].sum()
                )
                * 100,
                "congestion_%": congestion_rate.mean() * 100,
            }
        ]
    ).dropna(
        axis=1
    )  # Remove None columns if new branches don't exist

    if savefolder:
        savepath = os.path.join(savefolder, "table_branches_overview.csv")
        branches_overview.to_csv(savepath)

    savefolder = os.path.join(os.path.dirname(savefolder), "figures")
    plotting.plot_congestion_network(
        nodes,
        combined_branches,
        congestion_rate,
        savefolder=savefolder,
        plot_line_numbers=True,
        type="congestion",
    )
    plotting.plot_congestion_network(
        nodes,
        combined_branches,
        average_used_capacity_combined_branches / combined_branches["p_max"],
        savefolder=savefolder,
        plot_line_numbers=True,
        type="utilization",
    )
    return branches_overview


def table_macro_results(
    generators: pd.DataFrame,
    generator_build: pd.DataFrame,
    objective_value: float,
    branches: pd.DataFrame,
    branches_build: pd.DataFrame,
    batteries: pd.DataFrame,
    batteries_build: pd.DataFrame,
    load_shedding: pd.DataFrame,
    curtailment: pd.DataFrame,
    savefolder: str = "",
) -> None:
    num_generators_built = generator_build.sum().values[0]
    total_generators_capacity_built = generators["new_capacity"].sum()
    num_potential_generators = len(generators) / 2
    num_potential_transmission_lines = len(branches) / 2
    num_new_branches_built = branches_build.sum().values[0]
    total_capacity_built = branches["new_capacity"].sum()
    num_batteries_built = batteries_build.sum().values[0]
    num_potential_batteries = len(batteries.loc[batteries["exists"] == 0])
    battery_energy_capacity_built = batteries["new_energy_capacity"].sum()
    battery_power_capacity_built = batteries["new_power_capacity"].sum()

    data_macro = {
        "Objective Value (Billion Euro)": objective_value / 1e9,
        "Generators Built": num_generators_built,
        "Potential Generators": num_potential_generators,
        "Total Generator Capacity (MW)": total_generators_capacity_built,
        "Transmission Lines Expanded": num_new_branches_built,
        "Potential Transmission Lines": num_potential_transmission_lines,
        "Total Transmission Capacity (MW)": total_capacity_built,
        "Batteries Built": num_batteries_built,
        "Potential Batteries": num_potential_batteries,
        "Total Battery Energy Capacity (MWh)": battery_energy_capacity_built.sum(),
        "Total Battery Power Capacity (MW)": battery_power_capacity_built.sum(),
        "Total Load Shedding (MWh)": load_shedding.sum().sum(),
        "Total Curtailment (MWh)": curtailment.sum().sum(),
    }

    macro_df = pd.DataFrame([data_macro])
    if savefolder:
        savepath = os.path.join(savefolder, "table_macro_results.csv")
        macro_df.to_csv(savepath)


def table_generators_production_breakdown(
    generators: pd.DataFrame,
    generation: pd.DataFrame,
    model_config: dict,
    savefolder: str,
) -> None:
    generators["cost_of_buildout"] = np.where(
        (generators["new_capacity"] > 0),
        generators["new_capacity"] * generators["capital_cost"],
        0,
    )
    cost_of_buildout_by_carrier = generators.groupby("carrier")[
        "cost_of_buildout"
    ].sum()
    carriers = generators["carrier"].unique()
    total_production_by_carrier_old = {}
    total_production_by_carrier_new = {}
    total_production_cost_by_carrier = {}
    total_co2_emissions_cost_by_carrier = {}
    for carrier in carriers:
        carrier_columns = [col for col in generation.columns if col.endswith(carrier)]
        total_production_by_carrier_old[carrier] = (
            generation[carrier_columns].sum().sum()
        )
        total_production_cost = (
            (
                generation[carrier_columns]
                * generators.loc[carrier_columns, "marginal_cost"]
            )
            .sum()
            .sum()
        )
        total_co2_emissions_cost = (
            (
                generation[carrier_columns]
                * generators.loc[carrier_columns, "co2_emissions"]
            )
            .sum()
            .sum()
        ) * model_config["CO2_price"]
        new_carrier_columns = [
            col for col in generation.columns if col.endswith(f"{carrier} new")
        ]
        total_production_by_carrier_new[carrier] = (
            generation[new_carrier_columns].sum().sum()
        )
        total_production_cost_by_carrier[carrier] = (
            total_production_cost
            + (
                generation[new_carrier_columns]
                * generators.loc[new_carrier_columns, "marginal_cost"]
            )
            .sum()
            .sum()
        )
        total_co2_emissions_cost_by_carrier[carrier] = (
            total_co2_emissions_cost
            + (
                generation[new_carrier_columns]
                * generators.loc[new_carrier_columns, "co2_emissions"]
            )
            .sum()
            .sum()
            * model_config["CO2_price"]
        )
    total_production_by_carrier = {
        key: total_production_by_carrier_old[key] + total_production_by_carrier_new[key]
        for key in total_production_by_carrier_old.keys()
    }
    # Convert each dictionary to a DataFrame
    df_production = pd.DataFrame.from_dict(
        total_production_by_carrier, orient="index", columns=["Total Production [MW]"]
    )
    df_production_new = pd.DataFrame.from_dict(
        total_production_by_carrier_new, orient="index", columns=["New Production [MW]"]
    )
    df_production_old = pd.DataFrame.from_dict(
        total_production_by_carrier_old, orient="index", columns=["Old Production [MW]"]
    )
    df_production_cost = pd.DataFrame.from_dict(
        total_production_cost_by_carrier,
        orient="index",
        columns=["Total Production Cost [€]"],
    )

    # Concatenate the DataFrames
    generators_df_summary = pd.concat(
        [
            df_production,
            df_production_new,
            df_production_old,
            df_production_cost,
            cost_of_buildout_by_carrier,
        ],
        axis=1,
    )

    # Reset the index and rename the carrier column
    generators_df_summary.reset_index(inplace=True)
    generators_df_summary.rename(columns={"index": "carrier"}, inplace=True)
    marginal_cost_by_carrier = generators.groupby("carrier")["marginal_cost"].mean()
    emission_per_mwh_by_carrier = generators.groupby("carrier")["co2_emissions"].mean()
    generators_df_summary = pd.merge(
        generators_df_summary, marginal_cost_by_carrier, on="carrier"
    )
    generators_df_summary = pd.merge(
        generators_df_summary, emission_per_mwh_by_carrier, on="carrier"
    )
    # Set 'carrier' as the index and rename columns
    generators_df_summary.set_index("carrier", inplace=True)

    generators_df_summary.rename(
        columns={
            "cost_of_buildout": "Cost of Buildout [€]",
            "marginal_cost": "Marginal Cost [€/MWh]",
            "co2_emissions": "CO2 Emission Intensity [ton/MWh]",
        },
        inplace=True,
    )

    generators_df_summary["Total Production Cost [€] old"] = (
        generators_df_summary["Total Production [MW]"]
        * generators_df_summary["Marginal Cost [€/MWh]"]
    )
    generators_df_summary["Total CO2 Emissions [ton]"] = (
        generators_df_summary["Total Production [MW]"]
        * generators_df_summary["CO2 Emission Intensity [ton/MWh]"]
    )
    generators_df_summary["Total CO2 Cost [€]"] = (
        generators_df_summary["Total CO2 Emissions [ton]"] * model_config["CO2_price"]
    )
    generators_df_summary["Total Production Cost With Emissions [€]"] = (
        generators_df_summary["Total Production Cost [€]"]
        + generators_df_summary["Total CO2 Cost [€]"]
    )
    generators_df_summary["Total Cost €"] = (
        generators_df_summary["Cost of Buildout [€]"]
        + generators_df_summary["Total Production Cost With Emissions [€]"]
    )
    total = generators_df_summary.sum(numeric_only=True)
    generators_df_summary.loc["Total"] = total

    if savefolder:
        savepath = os.path.join(savefolder, "table_generators_production_breakdown.csv")
        generators_df_summary.to_csv(savepath)
    return generators_df_summary


def table_cost_breakdown(
    generators_df_summary: pd.DataFrame,
    generators: pd.DataFrame,
    generators_build: pd.DataFrame,
    branches: pd.DataFrame,
    branches_build: pd.DataFrame,
    batteries: pd.DataFrame,
    batteries_build: pd.DataFrame,
    load_shedding: pd.DataFrame,
    curtailment: pd.DataFrame,
    model_config: dict,
    savefolder: str = "",
) -> None:
    battery_build_cost = (
        batteries["new_energy_capacity"] * batteries["capital_cost"]
    ).sum()

    load_shedding_cost = load_shedding.sum().sum() * float(model_config["VOLL"])
    curtailment_cost = curtailment.sum().sum() * model_config["CC"]
    branches_cost = branches.loc[
        branches_build[branches_build["value"] == 1].index, "capital_cost"
    ].sum()

    cost_breakdown = {
        "Building Generators (€)": generators_df_summary.loc[
            "Total", "Cost of Buildout [€]"
        ],
        "Building Transmission Lines (€)": branches_cost,
        "Building Batteries (€)": battery_build_cost,
        "Energy Production Cost (w/o Emissions) (€)": generators_df_summary.loc[
            "Total", "Total Production Cost [€]"
        ],
        "CO2 Emissions Cost (€)": generators_df_summary.loc[
            "Total", "Total CO2 Cost [€]"
        ],
        "Production Cost with Emissions (€)": generators_df_summary.loc[
            "Total", "Total Production Cost With Emissions [€]"
        ],
        "Load Shedding Cost (€)": load_shedding_cost,
        "Curtailment Cost (€)": curtailment_cost,
    }

    cost_breakdown["Total Cost (€)"] = (
        sum(cost_breakdown.values())
        - cost_breakdown["Production Cost with Emissions (€)"]
    )

    cost_breakdown_df = pd.DataFrame([cost_breakdown])

    if savefolder:
        savepath = os.path.join(savefolder, "table_cost_breakdown.csv")
        cost_breakdown_df.to_csv(savepath)


def analyze_run(
    model_config: dict,
    SAVE_FIGURES: bool = True,
    SAVE_TABLES: bool = True,
    show_plots: bool = False,
):
    print(30 * "-")
    print("Analyzing model run...")
    print(30 * "-")
    print(model_config)

    if not show_plots:
        original_show = plt.show

        # Override plt.show with a no-op lambda.
        plt.show = lambda: None
    # Set up folders
    folder = model_config["save_folder"]
    decision_variables_folder = os.path.join(folder, "decision_variables")
    model_info_folder = os.path.join(folder, "model_info")
    if not os.path.exists(decision_variables_folder):
        os.makedirs(decision_variables_folder)
    if not os.path.exists(model_info_folder):
        os.makedirs(model_info_folder)
    results_folder = os.path.join(folder, "results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    if SAVE_TABLES:
        tables_folder = os.path.join(results_folder, "tables")
        if not os.path.exists(tables_folder):
            os.makedirs(tables_folder)
    else:
        tables_folder = None
    if SAVE_FIGURES:
        figures_folder = os.path.join(results_folder, "figures")
        batteries_folder = os.path.join(figures_folder, "batteries")
        if not os.path.exists(figures_folder):
            os.makedirs(figures_folder)
        if not os.path.exists(batteries_folder):
            os.makedirs(batteries_folder)
    else:
        figures_folder = None
        batteries_folder = None

    input_data_folder = os.path.join(
        PROCESSED_DATA_FOLDER, model_config["data_folder_name"]
    )

    # region Data Loading & Processing

    # Load processed input data
    input_data = utils.load_csv_files_from_folder(input_data_folder)
    batteries = input_data["batteries"]
    branches = input_data["branches"]
    generators = input_data["generators"]
    capacity_factors = input_data["capacity_factors"]
    generator_costs = input_data["generator_costs"]
    hourly_demand = input_data["hourly_demand"]
    nodes = input_data["nodes"]
    # Load decision variables
    data = utils.load_csv_files_from_folder(decision_variables_folder)
    battery_build = data["battery_build"]
    battery_charging = data["battery_charging"]
    battery_discharging = data["battery_discharging"]
    battery_soc = data["battery_soc"]
    branch_build = data["branch_build"]
    branch_capacity = data["branch_capacity"]
    curtailment = data["curtailment"]
    generation = data["generation"]
    generator_build = data["generator_build"]
    generator_capacity = data["generator_capacity"]
    load_shedding = data["load_shedding"]
    power_flow = data["power_flow"]

    # Data processing
    branches = preprocess_branches(
        branches, branch_build=branch_build, branch_capacity=branch_capacity
    )
    generators = preprocess_generators(generators, generator_build, generator_capacity)
    batteries["exists"] = 0
    preprocess_batteries(
        batteries, battery_build, battery_charging, battery_discharging, battery_soc
    )
    # endregion

    tables_generators_overview(
        generators, generation, capacity_factors, model_config, tables_folder
    )
    plotting.plot_effective_capacity_generators_bar_chart(
        nodes=nodes,
        generators=generators,
        capacity_factors=capacity_factors,
        savefolder=figures_folder,
    )

    plotting.plot_sized_generators_and_lines(
        nodes,
        branches[branches["exists"] == 1],
        generators[(generators["new"] == 1) & (generators["exists"] == 1)],
        savefolder=figures_folder,
    )
    plotting.plot_monthly_production(generators, generation, savefolder=figures_folder)

    check_line_errors(branches, power_flow)

    plotting.plot_buses_and_lines(
        nodes,
        branches[(branches["exists"] == 1) & (branches["new"] == 1)],
        savefolder=figures_folder,
    )

    branches_complete_analysis(nodes, branches, power_flow, tables_folder)

    plotting.plot_battery_average_hourly_soc_per_battery(
        battery_soc=battery_soc, savefolder=batteries_folder
    )

    plotting.plot_battery_average_hourly_soc_per_month(
        battery_soc=battery_soc, savefolder=batteries_folder
    )

    plotting.plot_battery_average_hourly_soc_by_month_per_battery(
        battery_soc=battery_soc, savefolder=batteries_folder
    )

    plotting.plot_num_cycles_per_month(
        batteries=batteries,
        battery_discharging=battery_discharging,
        savefolder=batteries_folder,
    )
    plotting.cake_battery_usage_per_battery_per_month(
        batteries=batteries,
        battery_discharging=battery_discharging,
        savefolder=batteries_folder,
    )

    ### Macro Results ###
    model_info = pd.read_csv(os.path.join(model_info_folder, "model_info.csv"))
    objective_value = model_info["Objective Value"].values[0]
    table_macro_results(
        generators,
        generator_build,
        objective_value,
        branches,
        branch_build,
        batteries,
        battery_build,
        load_shedding,
        curtailment,
        savefolder=tables_folder,
    )

    generators_df_summary = table_generators_production_breakdown(
        generators=generators,
        generation=generation,
        model_config=model_config,
        savefolder=tables_folder,
    )

    table_cost_breakdown(
        generators_df_summary=generators_df_summary,
        generators=generators,
        generators_build=generator_build,
        branches=branches,
        branches_build=branch_build,
        batteries=batteries,
        batteries_build=battery_build,
        load_shedding=load_shedding,
        curtailment=curtailment,
        model_config=model_config,
        savefolder=tables_folder,
    )

    if not show_plots:
        # Restore the original show function
        plt.show = original_show
    print(30 * "-")
    print(
        f"Post Optimization Analysis completed for model_id: {model_config['model_id']}, model: {model_config["model_name"]}, run_id: {model_config["run_id"]}. \n Results saved in {results_folder}"
    )
    print(30 * "-")


### Below are functions tailored for stochastic runs ###

import yaml


def plot_utilization_time_series_by_carrier(
    generation: pd.DataFrame,
    capacity_factors: pd.DataFrame,
    generators: pd.DataFrame,
    year: int,
    week: int,
    scenario: str,
    savefolder: str | None = None,
) -> None:
    """
    Plot hourly utilization factor time series for each carrier in a given year/week/scenario.

    Utilization factor = sum_actual_gen_by_carrier_hour / sum_total_capacity*cf_by_carrier_hour

    Parameters
    ----------
    generation : pd.DataFrame
        Hourly generation with MultiIndex
        (generator, scenario, year, week, hour) and 'value' column (MWh).
    capacity_factors : pd.DataFrame
        Hourly capacity factors with MultiIndex (year, week, hour),
        columns are generator names, values in [0,1].
    generators : pd.DataFrame
        Generator table indexed by (year, generator) with:
        - total_capacity (MW)
        - carrier
        - color
    year : int
    week : int
    scenario : str
    savefolder : str or None
    """
    # metadata for this year
    meta = generators.reset_index()
    year_meta = meta[meta["year"] == year]
    # carriers to plot
    carriers = year_meta["carrier"].unique()

    # slice generation for this scenario-year-week; pivot to hour×generator
    gen_slice = generation.xs(
        (scenario, year, week), level=("scenario", "year", "week")
    )["value"]
    gen_df = gen_slice.unstack("generator").fillna(0)

    # slice capacity_factors for this year-week
    cf_slice = capacity_factors.xs((year, week), level=("year", "week")).fillna(0)

    # compute potential: cf * total_capacity
    cap_series = year_meta.set_index("generator")["total_capacity"]
    potential_df = cf_slice.multiply(cap_series, axis=1)

    # prepare plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for carrier in carriers:
        # generators of this carrier
        gens = year_meta[year_meta["carrier"] == carrier]["generator"]
        if len(gens) == 0:
            continue
        # sum actual & potential by hour
        actual_hour = gen_df[gens].sum(axis=1)
        potential_hour = potential_df[gens].sum(axis=1)
        # util factor (guard div0)
        util = actual_hour / potential_hour.replace({0: np.nan})
        # plot
        color = year_meta[year_meta["carrier"] == carrier]["color"].iloc[0]
        label = year_meta[year_meta["carrier"] == carrier]["nice_name"].iloc[0]
        ax.plot(util.index, util.values, label=label, color=color)

    ax.set_xlabel("Hour")
    ax.set_ylabel("Utilization Factor")
    fig.subplots_adjust(right=0.75)
    ax.legend(title="Technology", bbox_to_anchor=(1.02, 1), loc="upper left")
    print(
        f"Hourly utilization factor by carrier: {year}, week {week}, scenario {scenario}"
    )
    plt.tight_layout()
    plt.show()

    if savefolder:
        fname = f"util_factors_time_series_{year}_w{week}_{scenario}.png"
        fig.savefig(os.path.join(savefolder, fname), bbox_inches="tight")


def add_capacity_and_cumulative_metrics(
    generators_df, generator_capacity_df, capacity_col="value"
):
    """
    Merge new capacity values, drop 'extended_by', and add cumulative metrics.

    Parameters
    ----------
    generators_df : pd.DataFrame
        DataFrame indexed by (year, generator), containing columns including
        'extension_potential' and 'extended_by'.
    generator_capacity_df : pd.DataFrame
        DataFrame indexed by (generator, year) with a column (named by capacity_col)
        of new capacity values.
    capacity_col : str, optional
        Name of the column in generator_capacity_df holding the new capacity values
        (default 'value').

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by (year, generator) containing:
        - all original columns except 'extended_by'
        - 'new_capacity' (from generator_capacity_df)
        - 'cum_new_capacity' (running total of new_capacity per generator)
        - 'cum_extension_potential' (running total of extension_potential per generator)
    """
    # 1. Reset indices to expose 'year' and 'generator' as columns
    gen = generators_df.reset_index()
    cap = generator_capacity_df.reset_index()

    # 2. Rename the capacity output column to 'new_capacity'
    cap = cap.rename(columns={capacity_col: "new_capacity"})

    # 3. Merge new_capacity into the generators table
    merged = gen.merge(
        cap[["year", "generator", "new_capacity"]], on=["year", "generator"], how="left"
    )

    # 4. Drop the old 'extended_by' column
    merged = merged.drop(columns=["extended_by"], errors="ignore")

    # 5. Compute cumulative metrics per generator in chronological order
    merged = merged.sort_values(["generator", "year"])
    # cumulative new capacity
    merged["cum_new_capacity"] = merged.groupby("generator")["new_capacity"].cumsum()
    # cumulative extension potential
    merged["cum_extension_potential"] = merged.groupby("generator")[
        "extension_potential"
    ].cumsum()

    # 6. Restore the MultiIndex
    merged = merged.set_index(["year", "generator"])

    merged["total_capacity"] = merged["p_nom"] + merged["cum_new_capacity"]

    return merged


def plot_capacity_investment_by_carrier(
    generators: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Plot a stacked bar chart of new capacity investments per year by generation technology.

    Parameters
    ----------
    generators : pd.DataFrame
        DataFrame indexed by (year, generator) with columns:
        - new_capacity
        - carrier
        - nice_name
        - color
    savefolder : str or None, optional
        Directory to save the figure. If None, the figure is not saved. Default is None.
    """
    # Prepare the data: aggregate new_capacity by year & carrier
    temp = generators.reset_index()
    agg = (
        temp.groupby(["year", "carrier"])["new_capacity"]
        .sum()
        .unstack("carrier")
        .fillna(0)
    )

    # Build mappings for labels and colors from the original DataFrame
    meta = (
        temp[["carrier", "nice_name", "color"]]
        .drop_duplicates("carrier")
        .set_index("carrier")
    )
    labels = [meta.loc[c, "nice_name"] for c in agg.columns]
    colors = [meta.loc[c, "color"] for c in agg.columns]

    # Plot
    fig, ax = plt.subplots()
    agg.plot(kind="bar", stacked=True, ax=ax, color=colors)

    ax.set_xlabel("Year")
    ax.set_ylabel("New Capacity (MW)")
    ax.legend(labels, title="Technology")

    plt.tight_layout()

    # Save if requested
    if savefolder:
        savepath = os.path.join(savefolder, "capacity_investment_by_technology.png")
        fig.savefig(savepath, bbox_inches="tight")

    # Print title line instead of setting it on the plot
    print("Annual Capacity Investments by Technology")

    plt.show()


def plot_capacity_spending_by_carrier(
    generators: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Plot a stacked bar chart of annual capital expenditure on new capacity by generation technology,
    with the y-axis in billions of euros.

    Parameters
    ----------
    generators : pd.DataFrame
        DataFrame indexed by (year, generator) with columns:
        - new_capacity
        - capital_cost
        - carrier
        - nice_name
        - color
    savefolder : str or None, optional
        Directory to save the figure. If None, the figure is not saved. Default is None.
    """
    # Prepare the data: compute spending = new_capacity * capital_cost (in billions)
    temp = generators.reset_index()
    temp["investment_cost"] = temp["new_capacity"] * temp["capital_cost"] / 1e9

    # Aggregate by year & carrier
    agg = (
        temp.groupby(["year", "carrier"])["investment_cost"]
        .sum()
        .unstack("carrier")
        .fillna(0)
    )

    # Build mappings for labels and colors
    meta = (
        temp[["carrier", "nice_name", "color"]]
        .drop_duplicates("carrier")
        .set_index("carrier")
    )
    labels = [meta.loc[c, "nice_name"] for c in agg.columns]
    colors = [meta.loc[c, "color"] for c in agg.columns]

    # Plot
    fig, ax = plt.subplots()
    agg.plot(kind="bar", stacked=True, ax=ax, color=colors)

    ax.set_xlabel("Year")
    ax.set_ylabel("Capital Expenditure (billion €)")
    ax.legend(labels, title="Technology")

    plt.tight_layout()

    # Save if requested
    if savefolder:
        savepath = os.path.join(savefolder, "capacity_spending_by_technology.png")
        fig.savefig(savepath, bbox_inches="tight")

    # Print title line
    print("Annual Capital Expenditure by Technology")

    plt.show()


def plot_total_capacity_growth(
    generators: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Plot the growth in total installed capacity over time.

    For each year:
    - 'existing_capacity' is the initial p_nom plus all new_capacity built in prior periods.
    - 'new_capacity' is the capacity added in that year.

    Y-axis is in GW. Legend labels are made reader-friendly. Uses a new color palette.
    """
    temp = generators.reset_index()
    years = sorted(temp["year"].unique())
    first_year = years[0]

    # Initial capacity (MW) at first year
    init = temp[temp["year"] == first_year].set_index("generator")["p_nom"]

    # Compute cumulative new capacity
    temp = temp.sort_values(["generator", "year"])
    temp["cum_new_capacity"] = temp.groupby("generator")["new_capacity"].cumsum()
    temp["existing_capacity"] = init.reindex(temp["generator"]).values + (
        temp["cum_new_capacity"] - temp["new_capacity"]
    )

    # Aggregate and convert to GW
    agg = temp.groupby("year")[["existing_capacity", "new_capacity"]].sum().loc[years]
    agg_gw = agg / 1e3

    fig, ax = plt.subplots()
    agg_gw.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#1b9e77", "#d95f02"],  # new palette: green & orange
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Capacity (GW)")
    ax.legend(["Existing Capacity", "New Capacity"], title="")

    plt.tight_layout()

    if savefolder:
        savepath = os.path.join(savefolder, "total_capacity_growth.png")
        fig.savefig(savepath, bbox_inches="tight")

    print("Growth in Total Installed Capacity")
    plt.show()


def plot_extension_vs_potential_by_carrier(
    generators: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Single-figure bar chart of extension potential vs actual new capacity by technology and year,
    with values in GW.

    For each year, two sets of stacked bars (Actual vs Potential), colored by carrier.
    Potential bars are drawn with alpha=0.5.

    Parameters
    ----------
    generators : pd.DataFrame
        DataFrame indexed by (year, generator) with columns:
        - new_capacity
        - extension_potential
        - carrier
        - nice_name
        - color
    savefolder : str or None, optional
        Directory to save the figure. If None, the figure is not saved. Default is None.
    """
    temp = generators.reset_index()
    agg = (
        temp.groupby(["year", "carrier"])
        .agg(
            extension_potential=("extension_potential", "sum"),
            new_capacity=("new_capacity", "sum"),
        )
        .reset_index()
    )

    # Pivot and convert to GW
    potential_df = (
        agg.pivot(index="year", columns="carrier", values="extension_potential").fillna(
            0
        )
        / 1e3
    )
    actual_df = (
        agg.pivot(index="year", columns="carrier", values="new_capacity").fillna(0)
        / 1e3
    )
    years = potential_df.index.tolist()
    carriers = potential_df.columns.tolist()

    meta = (
        temp[["carrier", "nice_name", "color"]]
        .drop_duplicates("carrier")
        .set_index("carrier")
    )

    positions = np.arange(len(years))
    width = 0.4
    bottom_act = np.zeros(len(years))
    bottom_pot = np.zeros(len(years))

    fig, ax = plt.subplots(figsize=(10, 6))
    for carrier in carriers:
        color = meta.loc[carrier, "color"]
        nice = meta.loc[carrier, "nice_name"]
        act_vals = actual_df[carrier].reindex(years).values
        pot_vals = potential_df[carrier].reindex(years).values

        ax.bar(
            positions - width / 2,
            act_vals,
            width,
            bottom=bottom_act,
            color=color,
            label=nice,
        )
        ax.bar(
            positions + width / 2,
            pot_vals,
            width,
            bottom=bottom_pot,
            color=color,
            alpha=0.5,
        )

        bottom_act += act_vals
        bottom_pot += pot_vals

    ax.set_xticks(positions)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Year")
    ax.set_ylabel("Capacity (GW)")

    # Expand right margin for legends
    fig.subplots_adjust(right=0.85)

    from matplotlib.patches import Patch

    # Technology legend
    tech_handles = [Patch(facecolor=meta.loc[c, "color"]) for c in carriers]
    tech_labels = [meta.loc[c, "nice_name"] for c in carriers]
    leg1 = ax.legend(
        tech_handles,
        tech_labels,
        title="Technology",
        bbox_to_anchor=(0.88, 1),
        loc="upper left",
        borderaxespad=0.0,
        framealpha=1.0,
    )
    ax.add_artist(leg1)

    # Bar type legend
    type_handles = [
        Patch(facecolor="gray", alpha=1),
        Patch(facecolor="gray", alpha=0.5),
    ]
    type_labels = ["Actual", "Potential"]
    ax.legend(
        type_handles,
        type_labels,
        title="Bar Type",
        bbox_to_anchor=(1.02, 0.66),
        loc="upper left",
        borderaxespad=0.0,
        fontsize="small",
    )

    print("Extension Potential vs Actual New Capacity by Technology")
    plt.show()

    if savefolder:
        savepath = os.path.join(savefolder, "extension_vs_actual_by_technology.png")
        fig.savefig(savepath, bbox_inches="tight")


def create_energy_production_by_carrier_table(
    generation: pd.DataFrame,
    generators: pd.DataFrame,
    scenarios: dict,
    scenario_probabilities: dict,
    week_weights: dict,
    savefolder: str = None,
) -> None:
    """
    Create a table of annual energy production by carrier, both unweighted and weighted by scenario probabilities.

    Parameters
    ----------
    generation : pd.DataFrame
        DataFrame indexed by (year, generator, scenario, week) with columns:
        - value (energy produced)
    generators : pd.DataFrame
        DataFrame indexed by (year, generator) with columns:
        - carrier (energy carrier type)
    scenarios : dict
        Dictionary mapping years to lists of scenarios.
    scenario_probabilities : dict
        Dictionary mapping years to lists of scenario probabilities.
    week_weights : dict
        Dictionary mapping weeks to weights for annualization.

    Returns
    -------
    None
        Displays the unweighted and weighted tables.
    """
    # 1) Build a small DataFrame of (year, scenario, probability)
    scenario_data = []
    for year_str, scen_list in scenarios.items():
        year = int(year_str)
        probs = scenario_probabilities[year_str]
        for scen, prob in zip(scen_list, probs):
            scenario_data.append({"year": year, "scenario": scen, "probability": prob})
    prob_df = pd.DataFrame(scenario_data)

    # 2) Compute weekly sums and annualize using week_weights
    weekly = (
        generation.groupby(["generator", "scenario", "year", "week"])["value"]
        .sum()
        .reset_index(name="weekly_gen")
    )
    weekly["weight"] = weekly["week"].astype(str).map(week_weights)
    weekly["annual_gen"] = weekly["weekly_gen"] * weekly["weight"]

    # 3) Sum to get annual generation per (generator, scenario, year)
    annual = (
        weekly.groupby(["generator", "scenario", "year"])["annual_gen"]
        .sum()
        .reset_index()
    )

    # 4) Map each generator to its carrier
    gen2car = (
        generators.reset_index()[["generator", "carrier"]]
        .drop_duplicates("generator")
        .set_index("generator")["carrier"]
    )
    annual["carrier"] = annual["generator"].map(gen2car)

    # 5) Unweighted: mean across scenarios
    unweighted = (
        annual.groupby(["year", "carrier"])["annual_gen"]
        .mean()
        .unstack("carrier")
        .fillna(0)
    )

    # 6) Weighted: merge probabilities and compute expected value
    annual = annual.merge(prob_df, on=["year", "scenario"], how="left")
    annual["weighted_gen"] = annual["annual_gen"] * annual["probability"]
    weighted = (
        annual.groupby(["year", "carrier"])["weighted_gen"]
        .sum()
        .unstack("carrier")
        .fillna(0)
    )

    # # 7) Display
    # print("Annual energy production by carrier (unweighted average across scenarios):")
    # display(unweighted)

    print("\nAnnual expected energy production by carrier (scenario-weighted):")
    # display(weighted)
    if savefolder:
        savepath = os.path.join(savefolder, "energy_production_by_carrier_table.csv")
        weighted.to_csv(savepath, index=True)

    return weighted


def make_annual_production_table(
    generation: pd.DataFrame,
    generators: pd.DataFrame,
    scenarios: dict,
    scenario_probabilities: dict,
    week_weights: dict,
    savefolder: str | None = None,
) -> None:
    """
    Compute and print the annual expected energy production by carrier and scenario,
    and optionally save it to CSV.

    Parameters
    ----------
    generation : pd.DataFrame
        Hourly generation with MultiIndex (generator, scenario, year, week, hour)
        and a 'value' column in MWh.
    generators : pd.DataFrame
        Generator metadata with a 'carrier' column, indexed by (year, generator)
        or containing a 'generator' column to map to carrier.
    savefolder : str or None, optional
        Directory to save the CSV. If None, the table is not saved.
    """
    # 1. Build a DataFrame of scenario probabilities
    scenario_data = []
    for year_str, scen_list in scenarios.items():
        year = int(year_str)
        probs = scenario_probabilities[year_str]
        for scen, prob in zip(scen_list, probs):
            scenario_data.append({"year": year, "scenario": scen, "probability": prob})
    prob_df = pd.DataFrame(scenario_data)

    # 2. Sum hourly → weekly, then annualize
    weekly = (
        generation.groupby(["generator", "scenario", "year", "week"])["value"]
        .sum()
        .reset_index(name="weekly_gen")
    )
    weekly["weight"] = weekly["week"].astype(str).map(week_weights)
    weekly["annual_gen"] = weekly["weekly_gen"] * weekly["weight"]

    # 3. Collapse to annual per (generator, scenario, year)
    annual = (
        weekly.groupby(["generator", "scenario", "year"])["annual_gen"]
        .sum()
        .reset_index()
    )

    # 4. Map generator → carrier
    gen2car = (
        generators.reset_index()[["generator", "carrier"]]
        .drop_duplicates("generator")
        .set_index("generator")["carrier"]
    )
    annual["carrier"] = annual["generator"].map(gen2car)

    # 5. Merge probabilities and compute weighted generation
    annual = annual.merge(prob_df, on=["year", "scenario"], how="left")
    annual["expected_gen"] = annual["annual_gen"] * annual["probability"]

    # 6. Pivot to get carriers as columns, for each (year, scenario)
    table = (
        annual.groupby(["year", "scenario", "carrier"])["expected_gen"]
        .sum()
        .unstack("carrier")
        .fillna(0)
    )

    # 7. Add a Total column
    table["Total"] = table.sum(axis=1)

    # 8. Print the table
    print("Annual expected energy production by carrier and scenario (MWh):")
    # print(table)

    # 9. Save if requested
    if savefolder:
        savepath = os.path.join(
            savefolder, "annual_production_by_carrier_and_scenario.csv"
        )
        table.to_csv(savepath)
    return table


def make_weighted_annual_production_cost_by_year_table(
    generation: pd.DataFrame,
    generators: pd.DataFrame,
    scenarios: dict,
    scenario_probabilities: dict,
    week_weights: dict,
    carbon_price: float,
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Compute and return the scenario-weighted annual production cost (including CO₂ emissions)
    by technology and year.

    Parameters
    ----------
    generation : pd.DataFrame
        Hourly generation with MultiIndex (generator, scenario, year, week, hour)
        and a 'value' column in MWh.
    generators : pd.DataFrame
        Generator metadata with columns:
        - marginal_cost (EUR/MWh)
        - co2_emissions (tonnes CO₂/MWh)
        - carrier
        indexed by (year, generator) or containing 'year' & 'generator' columns.
    week_weights : dict[str, float]
        Multiplier for each week to annualize generation.
    carbon_price : float
        Emission price in EUR per tonne CO₂.
    savefolder : str or None, optional
        Directory to save the CSV. If None, no file is written.

    Returns
    -------
    pd.DataFrame
        Indexed by year, columns are carrier types (plus 'Total'),
        values are the scenario-weighted annual production cost including emissions (EUR).
    """
    # build scenario probability DataFrame
    scenario_data = []
    for year_str, scen_list in scenarios.items():
        year = int(year_str)
        probs = scenario_probabilities[year_str]
        for scen, prob in zip(scen_list, probs):
            scenario_data.append({"year": year, "scenario": scen, "probability": prob})
    prob_df = pd.DataFrame(scenario_data)

    # hourly → weekly sums, then annualize
    weekly = (
        generation.groupby(["generator", "scenario", "year", "week"])["value"]
        .sum()
        .reset_index(name="weekly_gen")
    )
    weekly["weight"] = weekly["week"].astype(str).map(week_weights)
    weekly["annual_gen"] = weekly["weekly_gen"] * weekly["weight"]

    # annual total per generator–scenario–year
    annual = (
        weekly.groupby(["generator", "scenario", "year"])["annual_gen"]
        .sum()
        .reset_index()
    )

    # attach cost & emission rates and carrier
    meta = generators.reset_index()[
        ["year", "generator", "marginal_cost", "co2_emissions", "carrier"]
    ].drop_duplicates(["year", "generator"])
    annual = annual.merge(meta, on=["year", "generator"], how="left")

    # compute cost including emissions
    annual["production_cost"] = annual["annual_gen"] * annual["marginal_cost"]
    annual["co2_emission"] = annual["annual_gen"] * annual["co2_emissions"]
    annual["co2_cost"] = annual["co2_emission"] * carbon_price
    annual["total_cost"] = annual["production_cost"] + annual["co2_cost"]

    # merge probabilities and compute weighted cost
    annual = annual.merge(prob_df, on=["year", "scenario"], how="left")
    annual["weighted_cost"] = annual["total_cost"] * annual["probability"]

    # pivot to carriers by year
    table = (
        annual.groupby(["year", "carrier"])["weighted_cost"]
        .sum()
        .unstack("carrier")
        .fillna(0)
    )
    table["Total"] = table.sum(axis=1)

    print("Annual scenario-weighted production cost (incl. CO₂) by technology (EUR):")

    if savefolder:
        path = os.path.join(savefolder, "annual_weighted_cost_by_carrier.csv")
        table.to_csv(path)

    return table


def make_weighted_annual_production_by_year(
    generation: pd.DataFrame,
    generators: pd.DataFrame,
    scenarios: dict,
    scenario_probabilities: dict,
    week_weights: dict,
    savefolder: str | None = None,
) -> None:
    """
    Compute and print the annual scenario-weighted energy production by carrier (years only),
    and optionally save it to CSV.

    Parameters
    ----------
    generation : pd.DataFrame
        Hourly generation with MultiIndex (generator, scenario, year, week, hour)
        and a 'value' column in MWh.
    generators : pd.DataFrame
        Generator metadata with a 'carrier' column, indexed by (year, generator)
        or containing a 'generator' column to map to carrier.
    savefolder : str or None, optional
        Directory to save the CSV. If None, the table is not saved.
    """
    # Build scenario probability lookup
    scenario_data = []
    for year_str, scen_list in scenarios.items():
        year = int(year_str)
        probs = scenario_probabilities[year_str]
        for scen, prob in zip(scen_list, probs):
            scenario_data.append({"year": year, "scenario": scen, "probability": prob})
    prob_df = pd.DataFrame(scenario_data)

    # Hourly → weekly sums and annualize
    weekly = (
        generation.groupby(["generator", "scenario", "year", "week"])["value"]
        .sum()
        .reset_index(name="weekly_gen")
    )
    weekly["weight"] = weekly["week"].astype(str).map(week_weights)
    weekly["annual_gen"] = weekly["weekly_gen"] * weekly["weight"]

    # Annual total per (generator, scenario, year)
    annual = (
        weekly.groupby(["generator", "scenario", "year"])["annual_gen"]
        .sum()
        .reset_index()
    )

    # Map generator → carrier
    gen2car = (
        generators.reset_index()[["generator", "carrier"]]
        .drop_duplicates("generator")
        .set_index("generator")["carrier"]
    )
    annual["carrier"] = annual["generator"].map(gen2car)

    # Merge probabilities and compute weighted generation
    annual = annual.merge(prob_df, on=["year", "scenario"], how="left")
    annual["expected_gen"] = annual["annual_gen"] * annual["probability"]

    # Pivot to carriers, index by year
    table = (
        annual.groupby(["year", "carrier"])["expected_gen"]
        .sum()
        .unstack("carrier")
        .fillna(0)
    )

    # Add total column
    table["Total"] = table.sum(axis=1)

    # Print
    print("Annual scenario-weighted energy production by carrier (MWh):")
    # print(table)

    # Save if requested
    if savefolder:
        savepath = os.path.join(savefolder, "annual_weighted_production_by_carrier.csv")
        table.to_csv(savepath)
    return table


def plot_weighted_production_evolution(
    weighted_table: pd.DataFrame,
    generators: pd.DataFrame,
    savefolder: str | None = None,
) -> None:
    """
    Plot the evolution of scenario-weighted annual energy production by technology over years.

    Parameters
    ----------
    weighted_table : pd.DataFrame
        Indexed by year, columns are carrier types (and possibly 'Total'), values are expected annual generation in MWh.
    generators : pd.DataFrame
        Generator metadata with 'carrier', 'nice_name', and 'color' columns.
    savefolder : str or None
        Directory to save the figure. If None, the figure is not saved.
    """
    # Identify carrier columns (exclude 'Total' if present)
    carriers = [c for c in weighted_table.columns if c != "Total"]

    # Extract color and label mappings
    meta = (
        generators.reset_index()[["carrier", "nice_name", "color"]]
        .drop_duplicates("carrier")
        .set_index("carrier")
    )

    # Convert MWh → TWh
    data_twh = weighted_table[carriers] / 1e6

    # Plot stacked area
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [meta.loc[c, "color"] for c in carriers]
    labels = [meta.loc[c, "nice_name"] for c in carriers]
    ax.stackplot(
        data_twh.index, [data_twh[c] for c in carriers], labels=labels, colors=colors
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Generation (TWh)")
    plt.tight_layout()

    # Place legend to the right
    fig.subplots_adjust(right=0.75)
    ax.legend(title="Technology", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Print title
    print(
        "Evolution of Annual Energy Production by Technology (averaged across scenarios)"
    )

    plt.show()

    # Save if requested
    if savefolder:
        savepath = os.path.join(savefolder, "production_evolution_by_technology.png")
        fig.savefig(savepath, bbox_inches="tight")


def plot_weighted_production_evolution_no_legend(
    weighted_table: pd.DataFrame,
    generators: pd.DataFrame,
    savefolder: str | None = None,
) -> None:
    """
    Plot the evolution of scenario-weighted annual energy production by technology over years.

    Parameters
    ----------
    weighted_table : pd.DataFrame
        Indexed by year, columns are carrier types (and possibly 'Total'), values are expected annual generation in MWh.
    generators : pd.DataFrame
        Generator metadata with 'carrier', 'nice_name', and 'color' columns.
    savefolder : str or None
        Directory to save the figure. If None, the figure is not saved.
    """
    # Identify carrier columns (exclude 'Total' if present)
    carriers = [c for c in weighted_table.columns if c != "Total"]

    # Extract color and label mappings
    meta = (
        generators.reset_index()[["carrier", "nice_name", "color"]]
        .drop_duplicates("carrier")
        .set_index("carrier")
    )

    # Convert MWh → TWh
    data_twh = weighted_table[carriers] / 1e6

    # Plot stacked area
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [meta.loc[c, "color"] for c in carriers]
    labels = [meta.loc[c, "nice_name"] for c in carriers]
    ax.stackplot(
        data_twh.index, [data_twh[c] for c in carriers], labels=labels, colors=colors
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Generation (TWh)")
    plt.tight_layout()

    # Place legend to the right
    # fig.subplots_adjust(right=0.75)
    # ax.legend(title="Technology", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Print title
    print(
        "Evolution of Annual Energy Production by Technology (averaged across scenarios)"
    )

    plt.show()

    # Save if requested
    if savefolder:
        savepath = os.path.join(
            savefolder, "production_evolution_by_technology_no_legend.png"
        )
        fig.savefig(savepath, bbox_inches="tight")


def plot_weighted_production_cost_evolution(
    weighted_cost_table: pd.DataFrame,
    generators: pd.DataFrame,
    savefolder: str | None = None,
) -> None:
    """
    Plot the evolution of scenario-weighted annual production cost including CO₂ emissions
    by technology over years, as a stacked area chart.

    Parameters
    ----------
    weighted_cost_table : pd.DataFrame
        Indexed by year, columns are carrier types (and possibly 'Total'),
        values are expected annual production cost including emissions in EUR.
    generators : pd.DataFrame
        Generator metadata with 'carrier', 'nice_name', and 'color' columns.
    savefolder : str or None
        Directory to save the figure. If None, the figure is not saved.
    """
    # identify technology columns (drop any 'Total')
    carriers = [c for c in weighted_cost_table.columns if c != "Total"]

    # build mapping of nice_name and color
    meta = (
        generators.reset_index()[["carrier", "nice_name", "color"]]
        .drop_duplicates("carrier")
        .set_index("carrier")
    )

    # convert EUR → billion EUR
    data_beur = weighted_cost_table[carriers] / 1e9

    # plot stacked area
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [meta.loc[c, "color"] for c in carriers]
    labels = [meta.loc[c, "nice_name"] for c in carriers]
    ax.stackplot(
        data_beur.index, [data_beur[c] for c in carriers], labels=labels, colors=colors
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Production Cost including Emissions (billion €)")
    plt.tight_layout()

    # shift legend to right
    fig.subplots_adjust(right=0.75)
    ax.legend(title="Technology", bbox_to_anchor=(1.02, 1), loc="upper left")

    print("Evolution of Annual Production Cost including CO₂ Emissions by Technology")
    plt.show()

    if savefolder:
        savepath = os.path.join(
            savefolder, "production_cost_evolution_by_technology.png"
        )
        fig.savefig(savepath, bbox_inches="tight")


def make_yearly_system_costs_table(
    generation: pd.DataFrame,
    generators: pd.DataFrame,
    week_weights: dict[str, float],
    carbon_price: float,
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Compute annual system-level metrics by year and scenario:
      - yearly production (MWh)
      - yearly CO2 emissions (tonnes)
      - yearly CO2 emission cost (EUR)
      - yearly production cost (EUR)
      - yearly production cost including emission cost (EUR)

    Parameters
    ----------
    generation : pd.DataFrame
        Hourly generation with MultiIndex
        (generator, scenario, year, week, hour) and a 'value' column in MWh.
    generators : pd.DataFrame
        Generator metadata with columns:
        - marginal_cost (EUR/MWh)
        - co2_emissions (tonnes CO2/MWh)
        indexed by (year, generator) or containing 'year' & 'generator'.
    week_weights : dict[str, float]
        Multipliers for each week to annualize generation.
    carbon_price : float
        Emission price in EUR per tonne CO2.
    savefolder : str or None, optional
        Directory to save the CSV. If None, no file is written.

    Returns
    -------
    pd.DataFrame
        Indexed by (year, scenario) with columns:
        'production', 'co2_emission', 'co2_emission_cost',
        'production_cost', 'production_cost_with_emission'.
    """
    # 1. Hourly → weekly sum
    tmp = (
        generation.groupby(["generator", "scenario", "year", "week"])["value"]
        .sum()
        .reset_index(name="weekly_gen")
    )
    tmp["weight"] = tmp["week"].astype(str).map(week_weights)
    tmp["annual_gen"] = tmp["weekly_gen"] * tmp["weight"]

    # 2. Sum to (generator, scenario, year)
    annual = (
        tmp.groupby(["generator", "scenario", "year"])["annual_gen"].sum().reset_index()
    )

    # 3. Attach marginal_cost and co2_emissions
    meta = generators.reset_index()[
        ["year", "generator", "marginal_cost", "co2_emissions"]
    ].drop_duplicates(["year", "generator"])
    annual = annual.merge(meta, on=["year", "generator"], how="left")

    # 4. Compute metrics per generator‐scenario‐year
    annual["production"] = annual["annual_gen"]
    annual["co2_emission"] = annual["annual_gen"] * annual["co2_emissions"]
    annual["production_cost"] = annual["annual_gen"] * annual["marginal_cost"]
    annual["co2_emission_cost"] = annual["co2_emission"] * carbon_price
    annual["production_cost_with_emission"] = (
        annual["production_cost"] + annual["co2_emission_cost"]
    )

    # 5. Aggregate across generators → (year, scenario)
    table = (
        annual.groupby(["year", "scenario"])[
            [
                "production",
                "co2_emission",
                "co2_emission_cost",
                "production_cost",
                "production_cost_with_emission",
            ]
        ]
        .sum()
        .sort_index()
    )

    # 6. Print and save
    print("Yearly system metrics by year and scenario:")
    # print(table)

    if savefolder:
        path = os.path.join(savefolder, "yearly_system_costs_by_scenario.csv")
        table.to_csv(path)

    return table


def plot_co2_emissions_by_scenario(
    yearly_system_costs: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Plot CO₂ emissions by year and scenario (million tonnes) as a grouped bar chart.

    Parameters
    ----------
    yearly_system_costs : pd.DataFrame
        DataFrame indexed by (year, scenario) with a 'co2_emission' column (tonnes).
    savefolder : str or None, optional
        Directory to save the figure. If None, the figure is not saved.
    """
    # Pivot to have scenarios as columns, convert tonnes → million tonnes
    co2_df = yearly_system_costs["co2_emission"].unstack("scenario") / 1e6

    fig, ax = plt.subplots(figsize=(8, 6))
    # Grouped bar chart
    co2_df.plot(kind="bar", ax=ax, color=[scenario_colors[s] for s in co2_df.columns])

    ax.set_xlabel("Year")
    ax.set_ylabel("CO₂ Emissions (million tonnes)")
    plt.tight_layout()

    # Print title line
    print("CO₂ Emissions by Year and Scenario")

    plt.show()

    if savefolder:
        savepath = os.path.join(savefolder, "co2_emissions_by_scenario.png")
        fig.savefig(savepath, bbox_inches="tight")


def plot_co2_emissions_by_scenario_avg(
    yearly_system_costs: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Grouped bar chart of CO₂ emissions by year, scenario, and average (million tonnes).
    """
    # Pivot without filling zeros
    df = yearly_system_costs["co2_emission"].unstack("scenario") / 1e6
    # Compute average across non-NaN values
    df["Average"] = df.mean(axis=1)
    # Prepare for plotting (fill NaN to avoid plot errors)
    df_plot = df.fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    df_plot.plot(kind="bar", ax=ax, color=[scenario_colors[s] for s in df_plot.columns])

    ax.set_xlabel("Year")
    ax.set_ylabel("CO₂ Emissions (million tonnes)")
    plt.tight_layout()
    print("CO₂ Emissions by Year, Scenario, and Average")
    plt.show()

    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "co2_emissions_by_scenario_avg.png"),
            bbox_inches="tight",
        )


def plot_production_by_scenario(
    yearly_system_costs: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Grouped bar chart of annual production by year, scenario, and average (TWh).
    """
    df = yearly_system_costs["production"].unstack("scenario") / 1e6
    df["Average"] = df.mean(axis=1)
    df_plot = df.fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    df_plot.plot(kind="bar", ax=ax, color=[scenario_colors[s] for s in df_plot.columns])

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Production (TWh)")
    plt.tight_layout()
    print("Annual Production by Year, Scenario, and Average")
    plt.show()

    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "production_by_scenario_avg.png"),
            bbox_inches="tight",
        )


def plot_production_cost_by_scenario(
    yearly_system_costs: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Grouped bar chart of production cost by year, scenario, and average (billion €).
    """
    df = yearly_system_costs["production_cost"].unstack("scenario") / 1e9
    df["Average"] = df.mean(axis=1)
    df_plot = df.fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    df_plot.plot(kind="bar", ax=ax, color=[scenario_colors[s] for s in df_plot.columns])

    ax.set_xlabel("Year")
    ax.set_ylabel("Production Cost (billion €)")
    plt.tight_layout()
    print("Production Cost by Year, Scenario, and Average")
    plt.show()

    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "production_cost_by_scenario_avg.png"),
            bbox_inches="tight",
        )


def plot_production_cost_with_emission_by_scenario(
    yearly_system_costs: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Grouped bar chart of production cost including emissions by year, scenario, and average (billion €).
    """
    df = yearly_system_costs["production_cost_with_emission"].unstack("scenario") / 1e9
    df["Average"] = df.mean(axis=1)
    df_plot = df.fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    df_plot.plot(kind="bar", ax=ax, color=[scenario_colors[s] for s in df_plot.columns])

    ax.set_xlabel("Year")
    ax.set_ylabel("Production Cost w/ Emission (billion €)")
    plt.tight_layout()
    print("Production Cost w/ Emissions by Year, Scenario, and Average")
    plt.show()

    if savefolder:
        fig.savefig(
            os.path.join(
                savefolder, "production_cost_with_emission_by_scenario_avg.png"
            ),
            bbox_inches="tight",
        )


def plot_generation_curves_by_carrier(
    generation: pd.DataFrame,
    generators: pd.DataFrame,
    year: int,
    week: int,
    scenario: str,
    carrier_type: str,
    savefolder: str | None = None,
) -> None:
    """
    Plot hourly generation curves for all generators of a given carrier type
    in a specified year, week, and scenario.

    Parameters
    ----------
    generation : pd.DataFrame
        Hourly generation with MultiIndex
        (generator, scenario, year, week, hour) and a 'value' column.
    generators : pd.DataFrame
        Generator metadata with a 'carrier' and 'color' column,
        indexed by (year, generator) or with columns 'year' and 'generator'.
    year : int
        Year to plot.
    week : int
        Week number to plot.
    scenario : str
        Scenario name to plot.
    carrier_type : str
        Carrier (technology) whose generators to include.
    savefolder : str or None
        Directory to save the figure. If None, figure is not saved.
    """
    # find all generators of this carrier in that year
    meta = generators.reset_index()
    gens = meta.loc[
        (meta["year"] == year) & (meta["carrier"] == carrier_type), "generator"
    ].unique()
    if len(gens) == 0:
        print(f"No generators of carrier {carrier_type} in {year}")
        return

    # build a DataFrame of hours × generators
    dfs = []
    for g in gens:
        try:
            ser = generation.xs(
                (g, scenario, year, week),
                level=("generator", "scenario", "year", "week"),
            )["value"]
        except KeyError:
            continue
        dfs.append(ser.rename(g))
    if not dfs:
        print(
            f"No generation data for {carrier_type} in {year}, week {week}, scenario {scenario}"
        )
        return
    df = pd.concat(dfs, axis=1)

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    for gen in df.columns:
        # use the color from the generators table
        color = meta.loc[
            (meta["year"] == year) & (meta["generator"] == gen), "color"
        ].iloc[0]
        ax.plot(df.index, df[gen], label=gen, color=color)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Generation (MW)")
    title = f"Generation curves for {carrier_type}, {year}, week {week}, scenario {scenario}"
    print(title)
    plt.tight_layout()
    ax.legend(title="Generator", bbox_to_anchor=(1.02, 1), loc="upper left")

    if savefolder:
        fname = f"gen_curves_{carrier_type}_{year}_w{week}_{scenario}.png"
        fig.savefig(os.path.join(savefolder, fname), bbox_inches="tight")
    plt.show()


def plot_fraction_of_max_generation_by_carrier(
    generation: pd.DataFrame,
    capacity_factors: pd.DataFrame,
    generators: pd.DataFrame,
    year: int,
    week: int,
    scenario: str,
    carrier_type: str,
    savefolder: str | None = None,
) -> None:
    """
    Plot, for a given carrier type, the hourly fraction:
      actual_generation / (total_capacity * capacity_factor)
    in a specified year, week, and scenario.

    Parameters
    ----------
    generation : pd.DataFrame
        Hourly generation with MultiIndex
        (generator, scenario, year, week, hour) and a 'value' column (MWh).
    capacity_factors : pd.DataFrame
        Hourly capacity factors with MultiIndex (year, week, hour)
        and columns = generator names (values in [0,1]).
    generators : pd.DataFrame
        Generator table indexed by (year, generator) with columns:
        - total_capacity (MW)
        - color
    year : int
    week : int
    scenario : str
    carrier_type : str
    savefolder : str | None
    """
    # 1) Identify all generators of this carrier in that year
    meta = generators.reset_index()
    gens = meta.loc[
        (meta["year"] == year) & (meta["carrier"] == carrier_type), "generator"
    ].unique()
    if len(gens) == 0:
        print(f"No generators of carrier {carrier_type} in {year}")
        return

    # 2) Slice capacity factors for this year & week
    cf_slice = capacity_factors.xs((year, week), level=("year", "week"))

    # 3) Plot each generator’s fraction curve
    fig, ax = plt.subplots(figsize=(8, 4))
    for g in gens:
        # a) actual generation series
        try:
            gen_ser = generation.xs(
                (g, scenario, year, week),
                level=("generator", "scenario", "year", "week"),
            )["value"]
        except KeyError:
            continue

        # b) capacity‐factor series
        if g not in cf_slice.columns:
            continue
        cf_ser = cf_slice[g]

        # c) total installed capacity
        total_cap = generators.loc[(year, g), "total_capacity"]

        # d) theoretical max = total_capacity * cf
        theo = total_cap * cf_ser.values

        # e) fraction (guard divide‐by‐zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = gen_ser.values / theo

        # f) plot
        color = meta.loc[
            (meta["year"] == year) & (meta["generator"] == g), "color"
        ].iloc[0]
        ax.plot(gen_ser.index, frac, label=g, color=color)

    ax.set_xlabel("Hour")
    ax.set_ylabel("Fraction of Theoretical Max")
    print(
        f"Fraction of actual vs theoretical for {carrier_type}, {year}, w{week}, {scenario}"
    )
    plt.tight_layout()
    ax.legend(title="Generator", bbox_to_anchor=(1.02, 1), loc="upper left")

    # 4) Save if requested
    if savefolder:
        fname = f"fraction_gen_{carrier_type}_{year}_w{week}_{scenario}.png"
        fig.savefig(os.path.join(savefolder, fname), bbox_inches="tight")
    plt.show()


def plot_utilization_hierarchy(
    generation: pd.DataFrame,
    capacity_factors: pd.DataFrame,
    generators: pd.DataFrame,
    week_weights: dict[str, float],
    scenarios: dict[str, list[str]],
    savefolder: str | None = None,
) -> None:
    """
    Grouped bar chart of annual utilization factor by technology, with a 3‐level x‐axis:
      Year → Scenario → Carrier.

    Utilization factor = total actual generation / total possible generation
    (i.e. sum(actual_gen) ÷ sum(total_capacity * capacity_factor)).

    Parameters
    ----------
    generation : pd.DataFrame
        Hourly generation with MultiIndex (generator, scenario, year, week, hour)
        and a 'value' column in MWh.
    capacity_factors : pd.DataFrame
        Hourly capacity factors with MultiIndex (year, week, hour)
        and columns = generator names.
    generators : pd.DataFrame
        Generator table indexed by (year, generator) with columns:
        - total_capacity (MW)
        - carrier
        - nice_name
        - color
    week_weights : dict[str, float]
        Multipliers to annualize each sample week.
    scenarios : dict[str, list[str]]
        Mapping year→list of scenarios.
    savefolder : str or None
        Directory to save the figure. If None, figure is not saved.
    """
    # 1) Annual actual generation per generator-scenario-year
    weekly = (
        generation.groupby(["generator", "scenario", "year", "week"])["value"]
        .sum()
        .reset_index(name="weekly_gen")
    )
    weekly["weight"] = weekly["week"].map(lambda w: week_weights[str(w)])
    weekly["annual_actual"] = weekly["weekly_gen"] * weekly["weight"]
    actual = (
        weekly.groupby(["year", "scenario", "generator"])["annual_actual"]
        .sum()
        .unstack("generator")
        .fillna(0)
    )

    # 2) Annual possible generation per generator-year
    # sum capacity_factors over hours per sample week
    cf_weekly = capacity_factors.groupby(level=["year", "week"]).sum()
    # build weight series, index as ints
    weight_s = pd.Series(week_weights).rename_axis("week").astype(float)
    weight_s.index = weight_s.index.astype(int)
    # multiply by week weight via alignment on 'week' level
    weighted_cf = cf_weekly.multiply(weight_s, level="week", axis=0)
    # sum over weeks to annualize
    annual_cf = weighted_cf.groupby(level="year").sum()

    # total installed capacity per year×generator
    cap_df = generators["total_capacity"].unstack(level="generator")

    print("capacity_factors shape:", capacity_factors.shape)
    print("cf_weekly shape:", cf_weekly.shape)
    print("weighted_cf shape:", weighted_cf.shape)
    print("annual_cf shape:", annual_cf.shape)
    print("cap_df shape:", cap_df.shape)
    




    # potential generation = annual_cf * capacity
    potential = annual_cf * cap_df

    # 3) Map generator→carrier and aggregate
    gen2car = (
        generators.reset_index()[["generator", "carrier"]]
        .drop_duplicates("generator")
        .set_index("generator")["carrier"]
    )
    actual_car = actual.groupby(gen2car, axis=1).sum()  # (year,scenario)×carrier
    potential_car = potential.groupby(gen2car, axis=1).sum()  # year×carrier

    # 4) Compute utilization factor per (year,scenario,carrier)
    util = actual_car.div(potential_car, level="year").stack().unstack("carrier")


    print("Sample of potential_car:", potential_car.head())
    print("potential shape:", potential.shape)

    print("Sample of capacity_factors:", capacity_factors.head())
    print("Sample of cf_weekly:", cf_weekly.head())
    print("Sample of annual_cf:", annual_cf.head())
    print("Sample of cap_df:", cap_df.head())
    print("Sample of potential:", potential.head())

    print("actual shape:", actual.shape)
    print("Sample of actual:", actual.head())

    print("actual_car shape:", actual_car.shape)
    print("potential_car shape:", potential_car.shape)
    print("Sample of actual_car:", actual_car.head())

    # 5) Plot
    carriers = util.columns.tolist()
    meta = (
        generators.reset_index()[["carrier", "nice_name", "color"]]
        .drop_duplicates("carrier")
        .set_index("carrier")
    )
    colors = [meta.loc[c, "color"] for c in carriers]
    labels = [meta.loc[c, "nice_name"] for c in carriers]

    idx = list(util.index)  # list of (year,scenario)
    n_groups = len(idx)
    n_car = len(carriers)
    width = 0.8 / n_car
    base = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, carrier in enumerate(carriers):
        x = base + (i - (n_car - 1) / 2) * width
        ax.bar(x, util[carrier].values, width=width, color=colors[i], label=labels[i])

    tick_labels = [f"{yr}\n{sc}" for yr, sc in idx]
    ax.set_xticks(base)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_xlabel("Year / Scenario")
    ax.set_ylabel("Utilization Factor")

    fig.subplots_adjust(right=0.85)
    ax.legend(title="Technology", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    print("Annual Utilization Factor by Year, Scenario, and Carrier")
    plt.show()

    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "utilization_hierarchy.png"), bbox_inches="tight"
        )


def make_system_summary_table(
    generation: pd.DataFrame,
    generators: pd.DataFrame,
    week_weights: dict[str, float],
    scenarios: dict[str, list[str]],
    scenario_probabilities: dict[str, list[float]],
    carbon_price: float,
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Build a system‐level summary table by year with columns:
      - potential_capacity                  (MW)
      - new_capacity                        (MW added)
      - number_possible_buildouts           (count of gens with extension_potential>0)
      - number_actual_buildouts             (count of gens with new_capacity>0)
      - cost_of_buildout                    (EUR)
      - total_production                    (expected MWh)
      - total_co2_emissions                 (expected tonnes)
      - total_co2_emission_cost             (EUR)
      - total_production_cost               (EUR, excl CO₂)
      - total_production_cost_with_emission (EUR)
      - total_cost                          (EUR; prod_cost_with_emission + buildout)
    """
    # --- Buildout metrics (per year) ---
    df_gen = generators.reset_index()
    df_gen["buildout_cost"] = df_gen["new_capacity"] * df_gen["capital_cost"]
    build_metrics = df_gen.groupby("year").agg(
        potential_capacity=("extension_potential", "sum"),
        new_capacity=("new_capacity", "sum"),
        number_possible_buildouts=("extension_potential", lambda x: (x > 0).sum()),
        number_actual_buildouts=("new_capacity", lambda x: (x > 0).sum()),
        cost_of_buildout=("buildout_cost", "sum"),
    )

    # --- Scenario probabilities DataFrame ---
    scen_rows = []
    for y_str, scen_list in scenarios.items():
        y = int(y_str)
        for scen, prob in zip(scen_list, scenario_probabilities[y_str]):
            scen_rows.append({"year": y, "scenario": scen, "probability": prob})
    prob_df = pd.DataFrame(scen_rows)

    # --- Generation metrics (per gen‐scen‐year) ---
    weekly = (
        generation.groupby(["generator", "scenario", "year", "week"])["value"]
        .sum()
        .reset_index(name="weekly_gen")
    )
    weekly["weight"] = weekly["week"].astype(str).map(week_weights)
    weekly["annual_gen"] = weekly["weekly_gen"] * weekly["weight"]
    annual = (
        weekly.groupby(["generator", "scenario", "year"])["annual_gen"]
        .sum()
        .reset_index()
    )

    # Merge in marginal_cost & co2_emissions
    meta = generators.reset_index()[
        ["year", "generator", "marginal_cost", "co2_emissions"]
    ].drop_duplicates(["year", "generator"])
    annual = annual.merge(meta, on=["year", "generator"], how="left")

    # Compute per‐gen metrics
    annual["production"] = annual["annual_gen"]
    annual["co2_emission"] = annual["annual_gen"] * annual["co2_emissions"]
    annual["production_cost"] = annual["annual_gen"] * annual["marginal_cost"]
    annual["co2_emission_cost"] = annual["co2_emission"] * carbon_price
    annual["production_cost_with_emission"] = (
        annual["production_cost"] + annual["co2_emission_cost"]
    )

    # Merge probabilities, weight and collapse to year
    annual = annual.merge(prob_df, on=["year", "scenario"], how="left")
    for col in [
        "production",
        "co2_emission",
        "co2_emission_cost",
        "production_cost",
        "production_cost_with_emission",
    ]:
        annual[f"w_{col}"] = annual[col] * annual["probability"]

    gen_metrics = annual.groupby("year")[
        [
            f"w_{c}"
            for c in [
                "production",
                "co2_emission",
                "co2_emission_cost",
                "production_cost",
                "production_cost_with_emission",
            ]
        ]
    ].sum()
    gen_metrics.columns = [
        "total_production",
        "total_co2_emissions",
        "total_co2_emission_cost",
        "total_production_cost",
        "total_production_cost_with_emission",
    ]

    # --- Combine and compute final ---
    summary = build_metrics.join(gen_metrics)
    summary["total_cost"] = (
        summary["total_production_cost_with_emission"] + summary["cost_of_buildout"]
    )

    # optional save
    if savefolder:
        path = os.path.join(savefolder, "system_summary_by_year.csv")
        summary.to_csv(path)

    return summary


def make_system_summary_table_by_carrier(
    generation: pd.DataFrame,
    generators: pd.DataFrame,
    week_weights: dict[str, float],
    scenarios: dict[str, list[str]],
    scenario_probabilities: dict[str, list[float]],
    carbon_price: float,
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Build a system‐level summary table by year and carrier with columns:
      - potential_capacity                  (MW)
      - new_capacity                        (MW added)
      - number_possible_buildouts           (count of gens with extension_potential>0)
      - number_actual_buildouts             (count of gens with new_capacity>0)
      - cost_of_buildout                    (EUR)
      - total_production                    (expected MWh)
      - total_co2_emissions                 (expected tonnes)
      - total_co2_emission_cost             (EUR)
      - total_production_cost               (EUR, excl CO₂)
      - total_production_cost_with_emission (EUR)
      - total_cost                          (EUR; production_cost_with_emission + buildout)
    """
    # --- Buildout metrics per year & carrier ---
    df_gen = generators.reset_index()
    df_gen["buildout_cost"] = df_gen["new_capacity"] * df_gen["capital_cost"]
    build_metrics = df_gen.groupby(["year", "carrier"]).agg(
        potential_capacity=("extension_potential", "sum"),
        new_capacity=("new_capacity", "sum"),
        number_possible_buildouts=("extension_potential", lambda x: (x > 0).sum()),
        number_actual_buildouts=("new_capacity", lambda x: (x > 0).sum()),
        cost_of_buildout=("buildout_cost", "sum"),
    )

    # --- Scenario probabilities DataFrame ---
    scen_rows = []
    for y_str, scen_list in scenarios.items():
        y = int(y_str)
        for scen, prob in zip(scen_list, scenario_probabilities[y_str]):
            scen_rows.append({"year": y, "scenario": scen, "probability": prob})
    prob_df = pd.DataFrame(scen_rows)

    # --- Generation metrics per generator / scenario / year ---
    weekly = (
        generation.groupby(["generator", "scenario", "year", "week"])["value"]
        .sum()
        .reset_index(name="weekly_gen")
    )
    weekly["weight"] = weekly["week"].astype(str).map(week_weights)
    weekly["annual_gen"] = weekly["weekly_gen"] * weekly["weight"]
    annual = (
        weekly.groupby(["generator", "scenario", "year"])["annual_gen"]
        .sum()
        .reset_index()
    )

    # attach cost & emissions rates and carrier
    meta = generators.reset_index()[
        ["year", "generator", "marginal_cost", "co2_emissions", "carrier"]
    ].drop_duplicates(["year", "generator"])
    annual = annual.merge(meta, on=["year", "generator"], how="left")

    # compute per‐generator metrics
    annual["production"] = annual["annual_gen"]
    annual["co2_emission"] = annual["annual_gen"] * annual["co2_emissions"]
    annual["production_cost"] = annual["annual_gen"] * annual["marginal_cost"]
    annual["co2_emission_cost"] = annual["co2_emission"] * carbon_price
    annual["production_cost_with_emission"] = (
        annual["production_cost"] + annual["co2_emission_cost"]
    )

    # merge probabilities and compute weighted metrics
    annual = annual.merge(prob_df, on=["year", "scenario"], how="left")
    for col in [
        "production",
        "co2_emission",
        "production_cost",
        "co2_emission_cost",
        "production_cost_with_emission",
    ]:
        annual[f"w_{col}"] = annual[col] * annual["probability"]

    gen_metrics = annual.groupby(["year", "carrier"])[
        [
            f"w_{c}"
            for c in [
                "production",
                "co2_emission",
                "production_cost",
                "co2_emission_cost",
                "production_cost_with_emission",
            ]
        ]
    ].sum()
    gen_metrics.columns = [
        "total_production",
        "total_co2_emissions",
        "total_production_cost",
        "total_co2_emission_cost",
        "total_production_cost_with_emission",
    ]

    # --- Combine buildout & generation metrics ---
    summary = build_metrics.join(gen_metrics)
    summary["total_cost"] = (
        summary["total_production_cost_with_emission"] + summary["cost_of_buildout"]
    )

    # optional save
    if savefolder:
        path = os.path.join(savefolder, "system_summary_by_year_and_carrier.csv")
        summary.to_csv(path)

    return summary


def extend_branches_table(
    branches: pd.DataFrame, branch_capacity: pd.DataFrame
) -> pd.DataFrame:
    """
    Add investment and capacity‐growth columns to the branches DataFrame.

    Parameters
    ----------
    branches : pd.DataFrame
        MultiIndexed by (year, line) with columns:
        - p_max
        - capital_cost
        - length
        - loss_factor
        - extendable
        - extension_potential
    branch_capacity : pd.DataFrame
        MultiIndexed by (line, year) with a 'value' column of new MW built.

    Returns
    -------
    pd.DataFrame
        branches with four new columns:
        - new_capacity                (MW added in that year)
        - cum_new_capacity            (cumulative MW added over time)
        - cum_extension_potential     (cumulative extension potential over time)
        - total_capacity              (existing p_max + cum_new_capacity)
    """
    # 1. Flatten indices for merging
    b = branches.reset_index()  # columns: year, line, p_max, …
    cap = branch_capacity.reset_index()  # columns: line, year, value

    # 2. Rename and merge new capacity
    cap = cap.rename(columns={"value": "new_capacity"})
    merged = b.merge(
        cap[["year", "line", "new_capacity"]], on=["year", "line"], how="left"
    )

    # 3. Sort for cumulative calculations
    merged = merged.sort_values(["line", "year"])

    # 4. Compute cumulative sums by line
    merged["cum_new_capacity"] = merged.groupby("line")["new_capacity"].cumsum()
    merged["cum_extension_potential"] = merged.groupby("line")[
        "extension_potential"
    ].cumsum()

    # 5. Total capacity = original + cumulative additions
    merged["total_capacity"] = merged["p_max"] + merged["cum_new_capacity"]

    # 6. Restore MultiIndex
    return merged.set_index(["year", "line"])


def plot_branch_new_capacity(
    branches: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Plot a bar chart of annual new branch capacity additions.

    Parameters
    ----------
    branches : pd.DataFrame
        DataFrame indexed by (year, line) containing a 'new_capacity' column (MW added).
    savefolder : str or None, optional
        Directory to save the figure. If None, the figure is not saved.
    """
    # Aggregate new_capacity by year
    temp = branches.reset_index()
    annual_new = temp.groupby("year")["new_capacity"].sum()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(annual_new.index.astype(str), annual_new.values)
    ax.set_xlabel("Year")
    ax.set_ylabel("New Capacity (MW)")
    plt.tight_layout()

    # Print title line
    print("Annual New Branch Capacity Additions")
    plt.show()

    # Save if requested
    if savefolder:
        savepath = os.path.join(savefolder, "branch_new_capacity_by_year.png")
        fig.savefig(savepath, bbox_inches="tight")


def make_branch_buildout_summary(
    extended_branches: pd.DataFrame, savefolder: str | None = None
) -> pd.DataFrame:
    """
    Compute yearly branch buildout metrics:
      - potential_capacity           (MW of extension_potential)
      - new_capacity                 (MW added)
      - number_possible_buildouts    (count of lines with extension_potential > 0)
      - number_actual_buildouts      (count of lines with new_capacity > 0)
      - cost_of_buildout             (EUR of new_capacity * capital_cost)

    Parameters
    ----------
    extended_branches : pd.DataFrame
        DataFrame indexed by (year, line) containing columns:
        - extension_potential
        - new_capacity
        - capital_cost
    savefolder : str or None, optional
        Directory to save CSV. If None, does not save.

    Returns
    -------
    pd.DataFrame
        Yearly summary table with the above buildout metrics.
    """
    df = extended_branches.reset_index()
    df["buildout_cost"] = df["new_capacity"] * df["capital_cost"]

    summary = df.groupby("year").agg(
        potential_capacity=("extension_potential", "sum"),
        new_capacity=("new_capacity", "sum"),
        number_possible_buildouts=("extension_potential", lambda x: (x > 0).sum()),
        number_actual_buildouts=("new_capacity", lambda x: (x > 0).sum()),
        cost_of_buildout=("buildout_cost", "sum"),
    )

    print("Branch buildout summary by year:")

    if savefolder:
        path = os.path.join(savefolder, "branch_buildout_summary_by_year.csv")
        summary.to_csv(path)

    return summary


def make_branch_flow_metrics(
    power_flow: pd.DataFrame,
    extended_branches: pd.DataFrame,
    week_weights: dict[str, float],
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Compute per‐line, per‐scenario, per‐year utilization and congestion rates.

    Utilization rate = annualized sum(|flow|) / (total_capacity * hours_per_year)
    Congestion rate = annualized hours(|flow| > 95% * total_capacity) / hours_per_year

    Parameters
    ----------
    power_flow : pd.DataFrame
        MultiIndexed by (line, scenario, year, week, hour) with column 'value' (MW flow).
    extended_branches : pd.DataFrame
        Indexed by (year, line) with column 'total_capacity' (MW).
    week_weights : dict[str, float]
        Mapping week → weight to annualize one representative week.
    savefolder : str or None
        Directory to save CSV. If None, does not save.

    Returns
    -------
    pd.DataFrame
        Indexed by (line, scenario, year) with columns:
        - utilization_rate   (fraction of capacity‐hours used)
        - congestion_rate    (fraction of hours > 95% flow)
    """
    # total hours in year from sample weeks
    hours_per_week = 168
    total_hours = sum(week_weights.values()) * hours_per_week

    # flatten power_flow
    pf = power_flow.reset_index().rename(columns={"value": "flow"})
    # absolute flow and week weight
    pf["abs_flow"] = pf["flow"].abs()
    pf["weight"] = pf["week"].astype(str).map(week_weights)

    # merge in total_capacity
    tb = extended_branches.reset_index()[["year", "line", "total_capacity"]]
    pf = pf.merge(tb, on=["year", "line"], how="left")

    # annualized absolute energy (MWh) and congested hours
    pf["weighted_abs_energy"] = pf["abs_flow"] * pf["weight"]
    pf["is_congested"] = pf["abs_flow"] > 0.95 * pf["total_capacity"]
    pf["weighted_cong_hours"] = pf["is_congested"] * pf["weight"]

    # aggregate
    grp = pf.groupby(["line", "scenario", "year"]).agg(
        abs_energy=("weighted_abs_energy", "sum"),
        cong_hours=("weighted_cong_hours", "sum"),
        total_capacity=("total_capacity", "first"),
    )

    # compute rates
    grp["utilization_rate"] = grp["abs_energy"] / (grp["total_capacity"] * total_hours)
    grp["congestion_rate"] = grp["cong_hours"] / total_hours

    # keep only rates
    result = grp[["utilization_rate", "congestion_rate"]]

    # save if requested
    if savefolder:
        path = os.path.join(savefolder, "branch_flow_metrics.csv")
        result.to_csv(path)

    return result


def make_aggregate_branch_flow_summary(
    power_flow: pd.DataFrame,
    extended_branches: pd.DataFrame,
    week_weights: dict[str, float],
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Compute annual, scenario‐specific system‐level flow metrics across all lines
    with year first in the index.

    Returns a DataFrame indexed by (year, scenario) with:
      - utilization_rate
      - congestion_rate
    """
    # total hours represented by sample weeks
    hours_per_year = sum(week_weights.values()) * 168

    # 1) Flatten and compute absolute flow & weights
    pf = power_flow.reset_index().rename(columns={"value": "flow"})
    pf["abs_flow"] = pf["flow"].abs()
    pf["weight"] = pf["week"].astype(str).map(week_weights)

    # 2) Merge in total_capacity
    tb = extended_branches.reset_index()[
        ["year", "line", "total_capacity"]
    ].drop_duplicates()
    pf = pf.merge(tb, on=["year", "line"], how="left")

    # 3) Weighted metrics
    pf["weighted_abs_energy"] = pf["abs_flow"] * pf["weight"]
    pf["is_congested"] = pf["abs_flow"] > 0.95 * pf["total_capacity"]
    pf["weighted_cong_hours"] = pf["is_congested"].astype(float) * pf["weight"]

    # 4) Aggregate by year, scenario
    agg = pf.groupby(["year", "scenario"]).agg(
        abs_energy=("weighted_abs_energy", "sum"),
        cong_hours=("weighted_cong_hours", "sum"),
    )

    # 5) Denominators per year
    cap_sum = extended_branches.reset_index().groupby("year")["total_capacity"].sum()
    line_counts = extended_branches.reset_index().groupby("year")["line"].nunique()

    # 6) Compute rates (leveraging alignment on 'year' level)
    agg["utilization_rate"] = agg["abs_energy"] / (cap_sum * hours_per_year)
    agg["congestion_rate"] = agg["cong_hours"] / (line_counts * hours_per_year)

    # 7) Select only the rates
    result = agg[["utilization_rate", "congestion_rate"]]

    # 8) Save if requested
    if savefolder:
        result.to_csv(os.path.join(savefolder, "aggregate_branch_flow_summary.csv"))

    return result


def plot_new_branches_for_years_with_investments(
    branches: pd.DataFrame,
    branch_capacity: pd.DataFrame,
    nodes: pd.DataFrame,
    savefolder: str | None = None,
) -> None:
    years_with_branch_investments = (
        branch_capacity[branch_capacity["value"] > 0]
        .index.get_level_values("year")
        .unique()
        .tolist()
    )
    if not savefolder:
        new_branch_plots_folder = None
    else:
        new_branch_plots_folder = os.path.join(savefolder, "new_branch_plots")
        os.makedirs(new_branch_plots_folder, exist_ok=True)
    for year in years_with_branch_investments:
        print(f"Transmission line investments in {year}:")
        temp_branches = branches.xs(year, level="year")
        temp_branches = temp_branches[temp_branches["new_capacity"] > 0]
        temp_branches["p_max"] = temp_branches["new_capacity"]
        plotting.plot_sized_branches_with_year(
            nodes, temp_branches, year, savefolder=new_branch_plots_folder
        )


def extend_batteries_table(
    batteries: pd.DataFrame, battery_capacity: pd.DataFrame
) -> pd.DataFrame:
    """
    Add new_capacity and total_capacity (cumulative) to the batteries DataFrame.

    Parameters
    ----------
    batteries : pd.DataFrame
        MultiIndexed by (year, battery) with battery parameters.
    battery_capacity : pd.DataFrame
        MultiIndexed by (battery, year) with a 'value' column of new MWh built.

    Returns
    -------
    pd.DataFrame
        batteries with two new columns:
        - new_capacity      (MWh added in that year)
        - cum_new_capacity  (cumulative MWh added up to that year)
        - total_capacity    (same as cum_new_capacity, since no initial capacity)
    """
    # 1. Reset indices for merging
    b = batteries.reset_index()  # exposes year, battery, params…
    cap = battery_capacity.reset_index()  # exposes battery, year, value

    # 2. Rename & merge the new capacity
    cap = cap.rename(columns={"value": "new_capacity"})
    merged = b.merge(
        cap[["battery", "year", "new_capacity"]], on=["battery", "year"], how="left"
    )

    # 3. Missing years mean zero new capacity
    merged["new_capacity"] = merged["new_capacity"].fillna(0)

    # 4. Sort and compute cumulatives
    merged = merged.sort_values(["battery", "year"])
    merged["cum_new_capacity"] = merged.groupby("battery")["new_capacity"].cumsum()

    # 5. Total capacity equals cumulative new capacity
    merged["total_capacity"] = merged["cum_new_capacity"]

    # 6. Restore the original MultiIndex
    return merged.set_index(["year", "battery"])


def plot_battery_investment_by_year(
    batteries: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Plot a bar chart of annual battery capacity investments.

    Parameters
    ----------
    batteries : pd.DataFrame
        DataFrame indexed by (year, battery) with a 'new_capacity' column (MWh added).
    savefolder : str or None, optional
        Directory to save the figure. If None, the figure is not saved.
    """
    # Aggregate new_capacity by year
    temp = batteries.reset_index()
    annual_new = temp.groupby("year")["new_capacity"].sum()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(annual_new.index.astype(str), annual_new.values)
    ax.set_xlabel("Year")
    ax.set_ylabel("Battery Investment (MWh)")
    plt.tight_layout()

    # Print title line
    print("Annual Battery Capacity Investments")
    plt.show()

    # Save if requested
    if savefolder:
        savepath = os.path.join(savefolder, "battery_investment_by_year.png")
        fig.savefig(savepath, bbox_inches="tight")


def plot_battery_investment_cost_by_year(
    batteries: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Plot a bar chart of annual battery investment costs in million euros.

    Parameters
    ----------
    batteries : pd.DataFrame
        DataFrame indexed by (year, battery) with columns:
        - new_capacity    (MWh added)
        - capital_cost    (EUR/MWh)
    savefolder : str or None, optional
        Directory to save the figure. If None, the figure is not saved.
    """
    temp = batteries.reset_index()
    # Calculate cost in million euros
    temp["investment_cost_meur"] = (temp["new_capacity"] * temp["capital_cost"]) / 1e6
    annual_cost = temp.groupby("year")["investment_cost_meur"].sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(annual_cost.index.astype(str), annual_cost.values)
    ax.set_xlabel("Year")
    ax.set_ylabel("Investment Cost (million €)")
    plt.tight_layout()

    print("Annual Battery Investment Cost (million €)")
    plt.show()

    if savefolder:
        savepath = os.path.join(savefolder, "battery_investment_cost_by_year.png")
        fig.savefig(savepath, bbox_inches="tight")


def make_battery_system_summary(
    extended_batteries: pd.DataFrame,
    battery_discharging: pd.DataFrame,
    week_weights: dict[str, float],
    scenarios: dict[str, list[str]],
    scenario_probabilities: dict[str, list[float]],
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Compute system‐level battery summary metrics by year:
      - num_buildout       (count of batteries with new_capacity>0)
      - capacity           (sum of total_capacity, MWh)
      - investment_cost    (sum of new_capacity*capital_cost, EUR)
      - cycles_per_year    (expected annual discharge / capacity)

    Parameters
    ----------
    extended_batteries : pd.DataFrame
        Indexed by (year, battery) with columns:
        - new_capacity
        - total_capacity
        - capital_cost
    battery_discharging : pd.DataFrame
        MultiIndexed by (battery, scenario, year, week, hour) with 'value' = discharge MW.
    week_weights : dict[str, float]
        Mapping week→weight for annualization.
    scenarios : dict[str, list[str]]
        Mapping year→list of scenarios.
    scenario_probabilities : dict[str, list[float]]
        Mapping year→list of scenario probabilities.
    savefolder : str or None
        Directory to save CSV. If None, does not save.

    Returns
    -------
    pd.DataFrame
        Indexed by year with columns:
        ['num_buildout', 'capacity', 'investment_cost', 'cycles_per_year']
    """
    # 1) Buildout metrics from extended_batteries
    eb = extended_batteries.reset_index()
    eb["build_cost"] = eb["new_capacity"] * eb["capital_cost"]
    build_summary = eb.groupby("year").agg(
        num_buildout=("new_capacity", lambda x: (x > 0).sum()),
        capacity=("total_capacity", "sum"),
        investment_cost=("build_cost", "sum"),
    )

    # 2) Compute expected annual discharge per battery-year
    wd = (
        battery_discharging.groupby(["battery", "scenario", "year", "week"])["value"]
        .sum()
        .reset_index(name="weekly_discharge")
    )
    wd["weight"] = wd["week"].astype(str).map(week_weights)
    wd["annual_discharge"] = wd["weekly_discharge"] * wd["weight"]

    # attach scenario probabilities
    prob_rows = []
    for y_str, scen_list in scenarios.items():
        y = int(y_str)
        for scen, prob in zip(scen_list, scenario_probabilities[y_str]):
            prob_rows.append({"year": y, "scenario": scen, "prob": prob})
    prob_df = pd.DataFrame(prob_rows)

    wd = wd.merge(prob_df, on=["year", "scenario"], how="left")
    wd["w_annual_discharge"] = wd["annual_discharge"] * wd["prob"]

    # sum across scenarios per battery-year
    exp_discharge = (
        wd.groupby(["battery", "year"])["w_annual_discharge"].sum().reset_index()
    )

    # 3) Sum expected discharge across batteries per year
    total_discharge = (
        exp_discharge.groupby("year")["w_annual_discharge"]
        .sum()
        .rename("expected_discharge")
    )

    # 4) Merge into summary to compute cycles
    summary = build_summary.join(total_discharge)
    summary["cycles_per_year"] = summary["expected_discharge"] / summary["capacity"]

    # drop intermediate
    summary = summary.drop(columns=["expected_discharge"])

    # 5) Optional save
    if savefolder:
        summary.to_csv(os.path.join(savefolder, "battery_system_summary_by_year.csv"))

    return summary


def plot_cycles_per_scenario_with_average(
    battery_discharging: pd.DataFrame,
    week_weights: dict[str, float],
    reference_cycle_mwh: float = 400,
    savefolder: str | None = None,
) -> None:
    """
    Grouped bar chart of annual number of reference cycles by year, scenario, and average.

    A “reference cycle” is defined as using `reference_cycle_mwh` MWh of discharge.
    """
    # 1) Flatten and annualize discharge
    df = battery_discharging.reset_index().rename(columns={"value": "discharge"})
    df["weight"] = df["week"].astype(str).map(week_weights)
    df["annual_discharge"] = df["discharge"] * df["weight"]

    # 2) Sum to annual discharge per battery, scenario, year
    annual = (
        df.groupby(["battery", "scenario", "year"])["annual_discharge"]
        .sum()
        .reset_index()
    )
    # 3) Total annual discharge by scenario & year
    total = (
        annual.groupby(["scenario", "year"])["annual_discharge"]
        .sum()
        .reset_index(name="annual_discharge")
    )
    # 4) Compute cycles
    total["cycles"] = total["annual_discharge"] / reference_cycle_mwh

    # 5) Pivot to have scenarios as columns, then compute average
    cycles_df = total.pivot(index="year", columns="scenario", values="cycles")
    cycles_df["Average"] = cycles_df.mean(axis=1, skipna=True)

    # 6) Prepare for plotting (fill NaN with 0)
    df_plot = cycles_df.fillna(0)

    # 7) Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    df_plot.plot(kind="bar", ax=ax, color=[scenario_colors[s] for s in df_plot.columns])

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Number of {int(reference_cycle_mwh)} MWh Cycles")
    plt.tight_layout()
    print(
        f"Number of {int(reference_cycle_mwh)} MWh Cycles by Year, Scenario, and Average"
    )
    plt.show()

    # 8) Save if requested
    if savefolder:
        filename = (
            f"battery_cycles_per_{int(reference_cycle_mwh)}MWh_by_scenario_avg.png"
        )
        fig.savefig(os.path.join(savefolder, filename), bbox_inches="tight")


def plot_curtailment_by_scenario(
    curtailment: pd.DataFrame,
    week_weights: dict[str, float],
    savefolder: str | None = None,
) -> None:
    """
    Grouped bar chart of annual curtailment by year and scenario.

    Parameters
    ----------
    curtailment : pd.DataFrame
        MultiIndexed by (generator, scenario, year, week, hour) with column 'value' (MWh curtailed).
    week_weights : dict[str, float]
        Mapping week → weight for annualization.
    savefolder : str or None
        Directory to save the figure. If None, the figure is not saved.
    """
    df = curtailment.reset_index().rename(columns={"value": "curtail"})
    df["weight"] = df["week"].astype(str).map(week_weights)
    df["annual_curtail"] = df["curtail"] * df["weight"]

    total = df.groupby(["scenario", "year"])["annual_curtail"].sum().reset_index()
    pivot = total.pivot(
        index="year", columns="scenario", values="annual_curtail"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    pivot.plot(kind="bar", ax=ax, color=[scenario_colors[s] for s in pivot.columns])
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Curtailment (MWh)")
    plt.tight_layout()
    print("Annual Curtailment by Year and Scenario")
    plt.show()

    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "curtailment_by_scenario.png"), bbox_inches="tight"
        )


def plot_load_shedding_by_scenario(
    load_shedding: pd.DataFrame,
    week_weights: dict[str, float],
    savefolder: str | None = None,
) -> None:
    """
    Grouped bar chart of annual load shedding by year and scenario.

    Parameters
    ----------
    load_shedding : pd.DataFrame
        MultiIndexed by (node, scenario, year, week, hour) with column 'value' (MWh shed).
    week_weights : dict[str, float]
        Mapping week → weight for annualization.
    savefolder : str or None
        Directory to save the figure. If None, the figure is not saved.
    """
    df = load_shedding.reset_index().rename(columns={"value": "shed"})
    df["weight"] = df["week"].astype(str).map(week_weights)
    df["annual_shed"] = df["shed"] * df["weight"]

    total = df.groupby(["scenario", "year"])["annual_shed"].sum().reset_index()
    pivot = total.pivot(index="year", columns="scenario", values="annual_shed").fillna(
        0
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    pivot.plot(kind="bar", ax=ax, color=[scenario_colors[s] for s in pivot.columns])
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Load Shedding (MWh)")
    plt.tight_layout()
    print("Annual Load Shedding by Year and Scenario")
    plt.show()

    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "load_shedding_by_scenario.png"),
            bbox_inches="tight",
        )


def make_curtailment_load_shedding_cost_table(
    curtailment: pd.DataFrame,
    load_shedding: pd.DataFrame,
    week_weights: dict[str, float],
    CC: float,
    VOLL: float,
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Compute annual curtailment and load‐shedding and their costs by year and scenario.

    Parameters
    ----------
    curtailment : pd.DataFrame
        MultiIndexed by (generator, scenario, year, week, hour) with column 'value' (MWh curtailed).
    load_shedding : pd.DataFrame
        MultiIndexed by (node, scenario, year, week, hour) with column 'value' (MWh shed).
    week_weights : dict[str, float]
        Mapping week → weight for annualization.
    CC : float
        Curtailment cost in EUR/MWh.
    VOLL : float
        Value of lost load in EUR/MWh.
    savefolder : str or None
        Directory to save the CSV. If None, does not save.

    Returns
    -------
    pd.DataFrame
        Indexed by (year, scenario) with columns:
        - load_shedding       (MWh)
        - curtailment         (MWh)
        - load_shedding_cost  (EUR)
        - curtailment_cost    (EUR)
    """
    # 1) Annualize curtailment
    df_c = curtailment.reset_index().rename(columns={"value": "curtailment"})
    df_c["weight"] = df_c["week"].astype(str).map(week_weights)
    df_c["annual_curtail"] = df_c["curtailment"] * df_c["weight"]
    annual_c = df_c.groupby(["year", "scenario"])["annual_curtail"].sum().reset_index()

    # 2) Annualize load shedding
    df_l = load_shedding.reset_index().rename(columns={"value": "load_shed"})
    df_l["weight"] = df_l["week"].astype(str).map(week_weights)
    df_l["annual_shed"] = df_l["load_shed"] * df_l["weight"]
    annual_l = df_l.groupby(["year", "scenario"])["annual_shed"].sum().reset_index()

    # 3) Merge and compute costs
    summary = annual_l.merge(annual_c, on=["year", "scenario"], how="outer").fillna(0)
    summary["load_shedding_cost"] = summary["annual_shed"] * VOLL
    summary["curtailment_cost"] = summary["annual_curtail"] * CC

    # 4) Rename and set index
    summary = summary.rename(
        columns={"annual_shed": "load_shedding", "annual_curtail": "curtailment"}
    )
    summary = summary.set_index(["year", "scenario"])

    # 5) Save if requested
    if savefolder:
        path = os.path.join(savefolder, "curtailment_load_shedding_costs.csv")
        summary.to_csv(path)

    return summary


def plot_yearly_costs_stacked_avg(
    system_summary: pd.DataFrame,
    battery_summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    yearly_system_costs: pd.DataFrame,
    curtail_shed_costs: pd.DataFrame,
    savefolder: str | None = None,
) -> None:
    """
    Stacked‐bar of annual system costs (billion €), averaged across scenarios:
      - investments (gen, bat, tx)
      - production cost
      - CO₂ cost
      - load shedding cost
      - curtailment cost
    """
    # 1) Average scenario‐level costs
    prod_cost_avg = (
        yearly_system_costs["production_cost"].unstack("scenario").mean(axis=1) / 1e9
    )
    co2_cost_avg = (
        yearly_system_costs["co2_emission_cost"].unstack("scenario").mean(axis=1) / 1e9
    )
    ls_cost_avg = (
        curtail_shed_costs["load_shedding_cost"].unstack("scenario").mean(axis=1) / 1e9
    )
    curt_cost_avg = (
        curtail_shed_costs["curtailment_cost"].unstack("scenario").mean(axis=1) / 1e9
    )

    # 2) Investments (already scenario‐independent), scaled to billion €
    inv_gen = system_summary["cost_of_buildout"] / 1e9
    inv_bat = battery_summary["investment_cost"] / 1e9
    inv_tx = branch_summary["cost_of_buildout"] / 1e9

    # 3) Assemble DataFrame
    df = pd.DataFrame(
        {
            "Invest Gen": inv_gen,
            "Invest Bat": inv_bat,
            "Invest Tx": inv_tx,
            "Prod Cost": prod_cost_avg,
            "CO₂ Cost": co2_cost_avg,
            "Load Shedding Cost": ls_cost_avg,
            "Curtailment Cost": curt_cost_avg,
        }
    ).fillna(0)

    # 4) Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(
        kind="bar", stacked=True, ax=ax, color=[category_colors[c] for c in df.columns]
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Cost (billion €)")
    plt.tight_layout()
    print("Annual System Costs (average across scenarios)")
    plt.show()

    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "yearly_system_costs_avg.png"), bbox_inches="tight"
        )


def plot_yearly_costs_stacked_by_scenario(
    system_summary: pd.DataFrame,
    battery_summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    yearly_system_costs: pd.DataFrame,
    curtail_shed_costs: pd.DataFrame,
    savefolder: str | None = None,
) -> None:
    """
    Stacked‐bar of annual system costs (billion €) for each scenario:
      - investments (gen, bat, tx)
      - production cost
      - CO₂ cost
      - load shedding cost
      - curtailment cost
    """
    # 1) Base costs per (year,scenario), scaled to billion €
    df = yearly_system_costs[["production_cost", "co2_emission_cost"]].copy()
    df = df.rename(
        columns={"production_cost": "Prod Cost", "co2_emission_cost": "CO₂ Cost"}
    )
    df["Prod Cost"] = df["Prod Cost"] / 1e9
    df["CO₂ Cost"] = df["CO₂ Cost"] / 1e9

    # 2) Add load shedding & curtailment cost (billion €)
    ls = (curtail_shed_costs["load_shedding_cost"] / 1e9).rename("Load Shedding Cost")
    ct = (curtail_shed_costs["curtailment_cost"] / 1e9).rename("Curtailment Cost")
    df = df.join(ls).join(ct).fillna(0)

    # 3) Add investments by mapping year → billion €
    yrs = df.index.get_level_values("year")
    df["Invest Gen"] = yrs.map(system_summary["cost_of_buildout"] / 1e9)
    df["Invest Bat"] = yrs.map(battery_summary["investment_cost"] / 1e9)
    df["Invest Tx"] = yrs.map(branch_summary["cost_of_buildout"] / 1e9)

    # 4) Reorder columns
    df = df[
        [
            "Invest Gen",
            "Invest Bat",
            "Invest Tx",
            "Prod Cost",
            "CO₂ Cost",
            "Load Shedding Cost",
            "Curtailment Cost",
        ]
    ]

    # 5) Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(
        kind="bar", stacked=True, ax=ax, color=[category_colors[c] for c in df.columns]
    )
    ax.set_xlabel("Year / Scenario")
    ax.set_ylabel("Cost (billion €)")
    plt.tight_layout()
    print("Annual System Costs by Scenario")
    plt.show()

    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "yearly_system_costs_by_scenario.png"),
            bbox_inches="tight",
        )


def compute_total_system_cost(
    system_summary: pd.DataFrame,
    battery_summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    yearly_system_costs: pd.DataFrame,
    curtail_shed_costs: pd.DataFrame,
    scenarios: dict[str, list[str]],
    scenario_probabilities: dict[str, list[float]],
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Compute, for each year, the total cost as in the optimisation objective:
      AIC_y + ∑_{ω} p_{ω,y} · OC_{ω,y},
    where:
      AIC_y = investment cost in generators + batteries + branches
      OC_{ω,y} = production_cost + co2_emission_cost
               + load_shedding_cost + curtailment_cost
      and all are drawn from the supplied summary tables.

    Returns a DataFrame indexed by year with columns:
      - AIC           (EUR)
      - expected_OC   (EUR)
      - total_cost    (EUR)
    """
    # 1) Build annual investment cost AIC_y
    # system_summary and branch_summary each have .cost_of_buildout,
    # battery_summary has .investment_cost
    battery_investment = battery_summary["investment_cost"].fillna(0)
    AIC = (
        system_summary["cost_of_buildout"]
        + battery_investment
        + branch_summary["cost_of_buildout"]
    ).rename("AIC")

    # 2) Compute OC_{ω,y} = prod + co2 + load shedding + curtailment
    # merge the two per-(year,scenario) tables
    oc = (
        yearly_system_costs[["production_cost", "co2_emission_cost"]]
        .join(curtail_shed_costs[["load_shedding_cost", "curtailment_cost"]])
        .fillna(0)
    )
    oc["OC"] = (
        oc["production_cost"]
        + oc["co2_emission_cost"]
        + oc["load_shedding_cost"]
        + oc["curtailment_cost"]
    )

    # 3) Weight by scenario probabilities to get E[OC] per year
    # build a DataFrame of probabilities
    prob_rows = []
    for y_str, scen_list in scenarios.items():
        y = int(y_str)
        for scen, p in zip(scen_list, scenario_probabilities[y_str]):
            prob_rows.append({"year": y, "scenario": scen, "p": p})
    prob_df = pd.DataFrame(prob_rows).set_index(["year", "scenario"])

    oc = oc.join(prob_df, how="left")
    oc["p"] = oc["p"].fillna(0)
    oc["pOC"] = oc["OC"] * oc["p"]

    exp_OC = oc.groupby(level="year")["pOC"].sum().rename("expected_OC")

    # 4) Combine AIC and expected_OC
    result = pd.concat([AIC, exp_OC], axis=1).fillna(0)
    result["total_cost"] = result["AIC"] + result["expected_OC"]
    if savefolder:
        result.to_csv(os.path.join(savefolder, "total_system_cost.csv"))

    # 5) Print and return
    print("Total system cost by year:")

    return result


def make_mega_cost_table(
    system_summary: pd.DataFrame,
    battery_summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    yearly_system_costs: pd.DataFrame,
    curtail_shed_costs: pd.DataFrame,
    scenarios: dict[str, list[str]],
    scenario_probabilities: dict[str, list[float]],
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Build a “mega‐table” of key annual metrics by (year, scenario), plus a final
    ('All','All') row that sums investments and uses probability‐weighted sums
    of operating costs across scenarios and years.

    Columns:
      - Invest Gen
      - Invest Bat
      - Invest Tx
      - Production Cost
      - CO2 Emission Cost
      - Load Shedding Cost
      - Curtailment Cost
      - Prod+CO2 Cost
      - Total CapEx
      - Total OpEx
      - Total Cost
    """
    # --- 1) Scenario‐level data ---
    df = yearly_system_costs[["production_cost", "co2_emission_cost"]].copy()
    df = df.join(curtail_shed_costs[["load_shedding_cost", "curtailment_cost"]])
    # map investments
    yrs = df.index.get_level_values("year")
    df["Invest Gen"] = yrs.map(system_summary["cost_of_buildout"])
    df["Invest Bat"] = yrs.map(battery_summary["investment_cost"])
    df["Invest Tx"] = yrs.map(branch_summary["cost_of_buildout"])
    # rename
    df = df.rename(
        columns={
            "production_cost": "Production Cost",
            "co2_emission_cost": "CO2 Emission Cost",
            "load_shedding_cost": "Load Shedding Cost",
            "curtailment_cost": "Curtailment Cost",
        }
    )
    # derived cols
    df["Prod+CO2 Cost"] = df["Production Cost"] + df["CO2 Emission Cost"]
    df["Total CapEx"] = df[["Invest Gen", "Invest Bat", "Invest Tx"]].sum(axis=1)
    df["Total OpEx"] = df[
        [
            "Production Cost",
            "CO2 Emission Cost",
            "Load Shedding Cost",
            "Curtailment Cost",
        ]
    ].sum(axis=1)
    df["Total Cost"] = df["Total CapEx"] + df["Total OpEx"]
    cols = [
        "Invest Gen",
        "Invest Bat",
        "Invest Tx",
        "Production Cost",
        "CO2 Emission Cost",
        "Load Shedding Cost",
        "Curtailment Cost",
        "Prod+CO2 Cost",
        "Total CapEx",
        "Total OpEx",
        "Total Cost",
    ]
    df = df[cols]

    # --- 2) Attach scenario probabilities ---
    prob_list = []
    for year_str, scen_list in scenarios.items():  # iterate the *names* dict
        year = int(year_str)
        probs = scenario_probabilities[year_str]  # these are the matching probabilities
        for scen, p in zip(scen_list, probs):
            prob_list.append({"year": year, "scenario": scen, "p": p})
    prob_df = pd.DataFrame(prob_list).set_index(["year", "scenario"])

    dfp = df.join(prob_df, how="left")

    # --- 3) Compute 'All' row via weighted sums ---
    # A) total investments (sum over years, scenario-independent)
    total_inv_gen = system_summary["cost_of_buildout"].sum()
    total_inv_bat = battery_summary["investment_cost"].sum()
    total_inv_tx = branch_summary["cost_of_buildout"].sum()
    # B) weighted sums of opex across (year,scenario)
    exp_prod_cost = (dfp["Production Cost"] * dfp["p"]).sum()
    exp_co2_cost = (dfp["CO2 Emission Cost"] * dfp["p"]).sum()
    exp_ls_cost = (dfp["Load Shedding Cost"] * dfp["p"]).sum()
    exp_curt_cost = (dfp["Curtailment Cost"] * dfp["p"]).sum()
    # recompute derived
    exp_prod_co2 = exp_prod_cost + exp_co2_cost
    total_capex = total_inv_gen + total_inv_bat + total_inv_tx
    total_opex = exp_prod_co2 + exp_ls_cost + exp_curt_cost
    total_cost = total_capex + total_opex

    # assemble 'All' row
    all_metrics = {
        "Invest Gen": total_inv_gen,
        "Invest Bat": total_inv_bat,
        "Invest Tx": total_inv_tx,
        "Production Cost": exp_prod_cost,
        "CO2 Emission Cost": exp_co2_cost,
        "Load Shedding Cost": exp_ls_cost,
        "Curtailment Cost": exp_curt_cost,
        "Prod+CO2 Cost": exp_prod_co2,
        "Total CapEx": total_capex,
        "Total OpEx": total_opex,
        "Total Cost": total_cost,
    }
    all_idx = pd.MultiIndex.from_tuples([("All", "All")], names=["year", "scenario"])
    all_row = pd.DataFrame(all_metrics, index=all_idx)

    # --- 4) Concatenate final mega table ---
    mega = pd.concat([df, all_row], sort=False)

    # drop the probability column if present
    if "p" in mega.columns:
        mega = mega.drop(columns=["p"])

    # --- 5) Save if requested ---
    if savefolder:
        mega.to_csv(os.path.join(savefolder, "mega_cost_table.csv"))

    return mega


def make_nodal_price_table(
    power_balance_duals: pd.DataFrame,
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Compute annual average locational marginal prices (LMPs) per node, scenario, and year
    from the power_balance_duals.

    Parameters
    ----------
    power_balance_duals : pd.DataFrame
        MultiIndexed by (node, scenario, year, week, hour) with column 'dual_value'.
    week_weights : dict[str, float]
        Mapping from week (as string) to weight for annualization.

    Returns
    -------
    pd.DataFrame
        Indexed by (year, scenario), columns are node names, values are annual-average LMP (€/MWh).
    """
    df = power_balance_duals.reset_index().rename(columns={"dual_value": "LMP"})
    # map week to weight
    table = df.groupby(["node", "scenario", "year"])["LMP"].mean()
    table = table.reset_index(name="LMP")
    table = table.pivot(
        index=["year", "scenario"], columns="node", values="LMP"
    ).fillna(0)

    print("Annual Average Locational Marginal Prices by Node")
    if savefolder:
        table.to_csv(os.path.join(savefolder, "nodal_price_table.csv"))
    # Print the first few rows of the table

    return table


def plot_nodal_price_evolution_with_markers(
    lmp_table: pd.DataFrame,
    nodes: list[str] | None = None,
    savefolder: str | None = None,
) -> None:
    """
    Plot line chart of annual average LMPs for selected nodes,
    with each data point marked and legend showing city names.

    Parameters
    ----------
    lmp_table : pd.DataFrame
        Indexed by (year, scenario), columns are node names with avg LMP.
    nodes : list of str, optional
        List of node names to plot. If None, plot all.
    savefolder : str or None
        Directory to save figure. If None, not saved.
    """
    if nodes is None:
        nodes = list(lmp_table.columns)

    years = lmp_table.index.get_level_values("year")
    fig, ax = plt.subplots(figsize=(10, 6))

    for node in nodes:
        city = plotting.node_to_city.get(node, node)
        ax.plot(years, lmp_table[node].values, marker="o", linestyle="-", label=city)

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Avg LMP (€/MWh)")
    fig.subplots_adjust(right=0.75)
    ax.legend(title="City", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    print("Annual Average Locational Marginal Prices by City (with markers)")
    plt.show()

    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "nodal_price_evolution_markers.png"),
            bbox_inches="tight",
        )


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_nodal_price_grouped_bar(
    lmp_table: pd.DataFrame,
    nodes: list[str] | None = None,
    node_to_city: dict[str, str] = None,
    savefolder: str | None = None,
) -> None:
    """
    Grouped bar chart of annual average LMPs for selected nodes,
    with one cluster per (year, scenario).

    Parameters
    ----------
    lmp_table : pd.DataFrame
        Indexed by (year, scenario), columns are node names with avg LMP.
    nodes : list of str, optional
        List of node names to plot. If None, plot all.
    node_to_city : dict, optional
        Mapping from node code to city name for legend labels.
    savefolder : str or None
        Directory to save figure. If None, not saved.
    """
    if nodes is None:
        nodes = list(lmp_table.columns)

    # Prepare data
    df = lmp_table[nodes].fillna(0)
    idx = list(df.index)  # list of (year, scenario) tuples
    n_groups = len(idx)
    n_nodes = len(nodes)

    # Bar positions
    base = np.arange(n_groups)
    width = 0.8 / n_nodes

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, node in enumerate(nodes):
        x = base + (i - (n_nodes - 1) / 2) * width
        city_label = node_to_city.get(node, node)
        ax.bar(x, df[node].values, width=width, label=city_label)

    # X-axis labels
    labels = [f"{year}\n{scenario}" for year, scenario in idx]
    ax.set_xticks(base)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("Year / Scenario")
    ax.set_ylabel("Annual Avg LMP (€/MWh)")

    # Legend
    fig.subplots_adjust(right=0.8)
    ax.legend(title="City", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    print("Annual Average Locational Marginal Prices by City (grouped bar)")
    plt.show()

    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "nodal_price_grouped_bar.png"), bbox_inches="tight"
        )


def compute_lmp_bucket_frequencies(
    power_balance_duals: pd.DataFrame,
    generators: pd.DataFrame,
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Bucket hourly LMPs into ±10% intervals around each carrier’s mean production_cost,
    combine onwind/offwind into 'wind', rename ror to 'non-binding',
    merge overlapping coal & CCGT into 'thermal', create gap buckets for positive/negative/curtailment,
    add symmetric negative buckets, and a curtailment bucket at -100 ±10%.
    Returns bucket bounds and frequencies.
    """
    import numpy as np
    import os

    # 1) Mean production cost by carrier
    carrier_cost = generators.reset_index().groupby("carrier")["production_cost"].mean()
    # 2) Combine & rename carriers: merge coal+CCGT into 'thermal'
    cost_map = {
        "non-binding": carrier_cost["ror"],
        "solar": carrier_cost["solar"],
        "wind": (carrier_cost["onwind"] + carrier_cost["offwind-ac"]) / 2,
        "thermal": None,  # placeholder
    }
    coal_c = carrier_cost["coal"]
    ccgt_c = carrier_cost["CCGT"]
    thermal_lower = min(coal_c, ccgt_c) * 0.9
    thermal_upper = max(coal_c, ccgt_c) * 1.1
    thermal_cost = (coal_c + ccgt_c) / 2
    cost_map["thermal"] = thermal_cost

    # 3) Build intervals ±10%
    intervals = []
    for label, cost in cost_map.items():
        if label == "thermal":
            lower, upper = thermal_lower, thermal_upper
        else:
            if cost == 0:
                lower, upper = -0.0001, 0.0001
            else:
                lower, upper = cost * 0.9, cost * 1.1
        intervals.append({"label": label, "lower": lower, "upper": upper})

    # 4) Mirror negative buckets for each positive bucket (except non-binding/gap)
    mirrored = []
    for iv in intervals:
        label = iv["label"]
        if label == "non-binding":
            continue
        mirrored.append(
            {"label": f"neg_{label}", "lower": -iv["upper"], "upper": -iv["lower"]}
        )

    # 5) Add curtailment bucket: -100 ±10%
    curtailment_cost = -100.0
    curtailment_lower = curtailment_cost * 1.1  # -110
    curtailment_upper = curtailment_cost * 0.9  # -90
    curtailment_low, curtailment_up = min(curtailment_lower, curtailment_upper), max(
        curtailment_lower, curtailment_upper
    )
    curtailment_bucket = {
        "label": "curtailment",
        "lower": curtailment_low,
        "upper": curtailment_up,
    }

    # 6) Combine all buckets: positives, negatives, curtailment
    all_buckets = intervals + mirrored + [curtailment_bucket]

    # 7) Sort all buckets by lower bound
    all_buckets = sorted(all_buckets, key=lambda x: x["lower"])

    # 8) Add gap buckets for any space between buckets
    buckets_with_gaps = []
    for i, iv in enumerate(all_buckets):
        buckets_with_gaps.append(iv)
        if i < len(all_buckets) - 1:
            next_iv = all_buckets[i + 1]
            if next_iv["lower"] > iv["upper"]:
                buckets_with_gaps.append(
                    {
                        "label": f"gap_{iv['label']}_{next_iv['label']}",
                        "lower": iv["upper"],
                        "upper": next_iv["lower"],
                    }
                )

    # 9) Final sort (optional, but keeps all in order)
    buckets = sorted(buckets_with_gaps, key=lambda x: x["lower"])

    # 10) Flatten LMPs
    df = power_balance_duals.reset_index()[["dual_value"]].rename(
        columns={"dual_value": "LMP"}
    )

    # 11) Assign each LMP to a bucket
    conditions = [
        (df["LMP"] >= b["lower"]) & (df["LMP"] <= b["upper"]) for b in buckets
    ]
    labels = [b["label"] for b in buckets]
    df["bucket"] = np.select(conditions, labels, default="other")

    # 12) Compute frequencies
    abs_counts = (
        df["bucket"]
        .value_counts()
        .reindex(labels + ["other"], fill_value=0)
        .astype(int)
    )
    total = len(df)
    rel_perc = (abs_counts / total) * 100

    # 13) Prepare bounds table
    bounds = pd.DataFrame(buckets).set_index("label")[["lower", "upper"]]
    bounds.index.name = "bucket"
    bounds.loc["other"] = [np.nan, np.nan]

    # 14) Combine into result
    result = bounds.copy()
    result["absolute"] = abs_counts
    result["relative_percent"] = rel_perc

    print("LMP Bucket Frequencies")
    if savefolder:
        result.to_csv(os.path.join(savefolder, "lmp_bucket_frequencies.csv"))

    return result


def plot_lmp_histogram_70plus(
    power_balance_duals: pd.DataFrame,
    cap: float = 70,
    bin_width: float = 1.0,
    savefolder: str | None = None,
) -> None:
    """
    Plot histogram of LMP frequencies with bin size 1 €/MWh, capping at `cap`:
    - Bins from floor(min LMP) to `cap` in steps of `bin_width`
    - One extra bin labeled 'cap+' for values above `cap`
    Prints the values and count in the `cap+` bin.
    """
    # Extract LMP values
    lmp = power_balance_duals.reset_index()["dual_value"]

    # Split into under/over
    lmp_under = lmp[lmp <= cap]
    lmp_over = lmp[lmp > cap]

    # Bin edges for <= cap
    min_val = np.floor(lmp_under.min() if not lmp_under.empty else cap)
    bins = np.arange(min_val, cap + bin_width, bin_width)

    # Histogram counts for <= cap
    counts, edges = np.histogram(lmp_under, bins=bins)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        edges[:-1],
        counts,
        width=bin_width,
        align="edge",
        edgecolor="black",
        label=f"≤ {cap} €/MWh",
    )
    # Plot the cap+ bin
    ax.bar(
        cap,
        len(lmp_over),
        width=bin_width,
        align="edge",
        edgecolor="black",
        color="gray",
        label=f"> {cap} €/MWh",
    )

    # Labels & ticks
    xticks = list(edges[:-1]) + [cap]
    xticklabels = [f"{int(e)}" for e in edges[:-1]] + [f"{int(cap)}+"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.set_xlabel("LMP (€/MWh)")
    ax.set_ylabel("Frequency (timesteps)")
    ax.legend()
    plt.tight_layout()
    print(f"LMP ≥ {cap} €/MWh: count = {len(lmp_over)} timesteps")
    # print("Values in ≥ cap bin:", np.sort(lmp_over.values))
    plt.show()

    # Save if requested
    if savefolder:
        fig.savefig(
            os.path.join(savefolder, f"lmp_histogram_{int(cap)}plus.png"),
            bbox_inches="tight",
        )


def make_lmp_frequency_table(
    power_balance_duals: pd.DataFrame,
    bin_width: float,
    cap: float | None = None,
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Create a table of LMP frequencies for custom bin width, including
    counts per year in each bin.

    Parameters
    ----------
    power_balance_duals : pd.DataFrame
        MultiIndexed by (node, scenario, year, week, hour) with 'dual_value' column.
    bin_width : float
        Width of each bin in €/MWh.
    cap : float or None, optional
        If provided, values > cap go into a single 'cap+' bin.

    Returns
    -------
    pd.DataFrame
        Columns:
          - lower_bound (float or 'cap+')
          - upper_bound (float or np.inf)
          - count        (int total)
          - percent      (float, percent of total)
          - count_{year} for each year
    """
    # Extract LMP and year series
    df = power_balance_duals.reset_index()[["year", "dual_value"]].rename(
        columns={"dual_value": "LMP"}
    )
    total = len(df)

    # Determine bin edges
    min_val = np.floor(df["LMP"].min())
    max_val = np.ceil(df["LMP"].max()) if cap is None else cap
    edges = np.arange(min_val, max_val + bin_width, bin_width)

    # Prepare list of years
    years = sorted(df["year"].unique())

    rows = []
    if cap is not None:
        # Under-cap bins
        df_under = df[df["LMP"] <= cap]
        cats = pd.cut(df_under["LMP"], bins=edges, include_lowest=True, right=False)
        df_under = df_under.assign(bin=cats)
        for interval in cats.cat.categories:
            mask = df_under["bin"] == interval
            cnt = mask.sum()
            pct = cnt / total * 100
            row = {
                "lower_bound": interval.left,
                "upper_bound": interval.right,
                "count": int(cnt),
                "percent": pct,
            }
            # per-year counts
            for y in years:
                row[f"count_{y}"] = int(
                    df_under[mask & (df_under["year"] == y)].shape[0]
                )
            rows.append(row)
        # cap+ bin
        df_over = df[df["LMP"] > cap]
        cnt_over = len(df_over)
        pct_over = cnt_over / total * 100
        row = {
            "lower_bound": cap,
            "upper_bound": np.inf,
            "count": int(cnt_over),
            "percent": pct_over,
        }
        for y in years:
            row[f"count_{y}"] = int(df_over[df_over["year"] == y].shape[0])
        rows.append(row)
    else:
        cats = pd.cut(df["LMP"], bins=edges, include_lowest=True, right=False)
        df = df.assign(bin=cats)
        for interval in cats.cat.categories:
            mask = df["bin"] == interval
            cnt = mask.sum()
            pct = cnt / total * 100
            row = {
                "lower_bound": interval.left,
                "upper_bound": interval.right,
                "count": int(cnt),
                "percent": pct,
            }
            for y in years:
                row[f"count_{y}"] = int(df[mask & (df["year"] == y)].shape[0])
            rows.append(row)

    # Build DataFrame
    table = pd.DataFrame(rows)
    if savefolder:
        table.to_csv(os.path.join(savefolder, "lmp_frequency_table.csv"))
    table_sorted = table.sort_values(by="count", ascending=False).reset_index(drop=True)
    if savefolder:
        table_sorted.to_csv(os.path.join(savefolder, "lmp_frequency_table_sorted.csv"))
    return table


def analyze_dual_interval(
    duals: pd.DataFrame | pd.Series,
    lower: float,
    upper: float,
    n_bins: int = 10,
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Print absolute and relative counts of dual values in [lower, upper],
    then return a table dividing that interval into n_bins sub‐intervals
    with counts and percentages within this interval.

    Parameters
    ----------
    duals : pd.DataFrame or pd.Series
        If DataFrame, must have column 'dual_value'; or a Series of values.
    lower : float
        Lower bound of interval (inclusive).
    upper : float
        Upper bound of interval (inclusive).
    n_bins : int, optional
        Number of sub‐bins within [lower, upper] to divide.

    Returns
    -------
    pd.DataFrame
        Columns: lower_bound, upper_bound, count, percent (of values in [lower,upper]).
    """
    # Extract series of values
    if isinstance(duals, pd.DataFrame) and "dual_value" in duals.columns:
        vals = duals["dual_value"]
    elif isinstance(duals, pd.Series):
        vals = duals
    else:
        raise ValueError(
            "`duals` must be DataFrame with 'dual_value' column or a Series."
        )

    total = len(vals)
    mask = (vals >= lower) & (vals <= upper)
    sub = vals[mask]
    count = len(sub)
    percent = count / total * 100

    print(f"Values in interval [{lower}, {upper}]: count = {count} ({percent:.2f}%)")

    # Build sub‐bins
    edges = np.linspace(lower, upper, n_bins + 1)
    cats = pd.cut(sub, bins=edges, include_lowest=True, right=True)
    freq = cats.value_counts().sort_index()

    rows = []
    for interval, cnt in freq.items():
        rows.append(
            {
                "lower_bound": interval.left,
                "upper_bound": interval.right,
                "count": int(cnt),
                "percent_of_interval": cnt / count * 100 if count > 0 else 0,
            }
        )

    df = pd.DataFrame(rows)
    if savefolder:
        df.to_csv(
            os.path.join(savefolder, f"table_duals_{lower}-{upper}_{n_bins}.csv"),
        )
    return pd.DataFrame(rows)


def plot_dual_interval_histogram(
    duals: pd.DataFrame | pd.Series,
    lower: float,
    upper: float,
    n_bins: int = 10,
    savefolder: str | None = None,
) -> None:
    """
    Plot histogram of duals within [lower, upper], divided into n_bins.

    Parameters
    ----------
    duals : pd.DataFrame or pd.Series
        If DataFrame, must have 'dual_value' column, otherwise a Series.
    lower : float
        Lower bound of histogram.
    upper : float
        Upper bound of histogram.
    n_bins : int, optional
        Number of bins between lower and upper.
    savefolder : str or None
        Directory to save the figure. If None, not saved.
    """
    # Extract values
    if isinstance(duals, pd.DataFrame) and "dual_value" in duals.columns:
        vals = duals["dual_value"]
    elif isinstance(duals, pd.Series):
        vals = duals
    else:
        raise ValueError(
            "`duals` must be DataFrame with 'dual_value' column or a Series."
        )

    # Filter interval
    sub = vals[(vals >= lower) & (vals <= upper)]

    # Bin edges
    edges = np.linspace(lower, upper, n_bins + 1)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sub, bins=edges, edgecolor="black")
    ax.set_xlabel("Dual value")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    print(f"Histogram of duals in [{lower}, {upper}] with {n_bins} bins")
    plt.show()

    # Save if requested
    if savefolder:
        fname = f"dual_hist_{lower}-{upper}_{n_bins}bins.png"
        fig.savefig(os.path.join(savefolder, fname), bbox_inches="tight")


def plot_top_10_dual_intervals(
    power_balance_duals: pd.DataFrame,
    sorted_freq_table: pd.DataFrame,
    n_bins: int = 10,
    savefolder: str | None = None,
) -> None:
    """
    Plot histograms for the top n dual intervals from the sorted frequency table.

    Parameters
    ----------
    power_balance_duals : pd.DataFrame
        DataFrame with dual values.
    sorted_freq_table : pd.DataFrame
        DataFrame with sorted frequency table.
    n_bins : int, optional
        Number of bins for histogram.
    savefolder : str or None
        Directory to save figures. If None, not saved.
    """
    if savefolder:
        savefolder = os.path.join(savefolder, "dual_interval_histograms")
        os.makedirs(savefolder, exist_ok=True)
    # Iterate over the top n intervals
    for idk, row in sorted_freq_table.head(10).iterrows():
        lb = row["lower_bound"]
        ub = row["upper_bound"]
        print(f"Analyzing dual interval [{lb}, {ub}]")
        plot_dual_interval_histogram(
            power_balance_duals, lb, ub, n_bins=n_bins, savefolder=savefolder
        )


def create_top_10_dual_intervals_tables(
    power_balance_duals: pd.DataFrame,
    sorted_freq_table: pd.DataFrame,
    n_bins: int = 10,
    savefolder: str | None = None,
) -> None:
    """
    Create tables for the top n dual intervals from the sorted frequency table.

    Parameters
    ----------
    power_balance_duals : pd.DataFrame
        DataFrame with dual values.
    sorted_freq_table : pd.DataFrame
        DataFrame with sorted frequency table.
    n_bins : int, optional
        Number of bins for histogram.
    savefolder : str or None
        Directory to save figures. If None, not saved.
    """
    # Iterate over the top n intervals
    if savefolder:
        savefolder = os.path.join(savefolder, "dual_interval_tables")
        os.makedirs(savefolder, exist_ok=True)
    for idk, row in sorted_freq_table.head(10).iterrows():
        lb = row["lower_bound"]
        ub = row["upper_bound"]
        analyze_dual_interval(
            power_balance_duals, lb, ub, n_bins=10, savefolder=savefolder
        )


def summarise_extension_status_by_carrier(
    generators: pd.DataFrame, savefolder=None
) -> pd.DataFrame:
    """
    For each year and carrier, count how many generators are
    'not extended', 'partly extended', or 'fully extended'.

    Parameters
    ----------
    generators : pd.DataFrame
        Indexed by (year, generator) with columns:
        - carrier
        - new_capacity
        - extension_potential

    Returns
    -------
    pd.DataFrame
        MultiIndexed by (year, carrier) with columns:
        ['not extended', 'partly extended', 'fully extended']
        containing the counts for each status.
    """
    df = generators.reset_index()

    def _status(row):
        nc = row["new_capacity"]
        ep = row["extension_potential"]
        if nc == 0:
            return "not extended"
        if np.isclose(nc, ep):
            return "fully extended"
        if nc < ep:
            return "partly extended"
        # cover any rounding issues
        return "fully extended"

    df["extension_status"] = df.apply(_status, axis=1)
    summary = (
        df.groupby(["year", "carrier", "extension_status"])
        .size()
        .unstack("extension_status")
        .fillna(0)
        .astype(int)
    )
    # ensure all three columns exist
    for col in ["not extended", "partly extended", "fully extended"]:
        if col not in summary.columns:
            summary[col] = 0

    # sort columns
    summary = summary[["not extended", "partly extended", "fully extended"]]

    print(f"Extension status by carrier")
    if savefolder:
        summary.to_csv(os.path.join(savefolder, "extension_status_by_carrier.csv"))
    # Print the first few rows of the summary
    return summary


def summarize_dual_nonzero_counts(
    dual_variables: dict[str, pd.DataFrame], savefolder: str | None = None
) -> pd.DataFrame:
    """
    For each dual variable, count total entries and non-zero entries.
    Optionally save the summary as CSV.

    Parameters
    ----------
    dual_variables : dict[str, pd.DataFrame]
        Mapping dual names → DataFrames with a 'dual_value' column.
    savefolder : str or None
        Directory to save the summary CSV. If None, no file is written.

    Returns
    -------
    pd.DataFrame
        Indexed by dual name with columns:
        - total_duals
        - nonzero_duals
        - percent_nonzero
    """
    rows = []
    for name, df in dual_variables.items():
        total = df["dual_value"].size
        nonzero = (df["dual_value"] != 0).sum()
        pct = (nonzero / total * 100) if total > 0 else 0.0
        rows.append(
            {
                "dual": name,
                "total_duals": int(total),
                "nonzero_duals": int(nonzero),
                "percent_nonzero": pct,
            }
        )
    summary = pd.DataFrame(rows).set_index("dual")

    if savefolder:
        path = os.path.join(savefolder, "dual_nonzero_summary.csv")
        summary.to_csv(path)

    return summary


def summarize_duals_by_year_scenario(
    dual_variables: dict[str, pd.DataFrame], savefolder: str | None = None
) -> pd.DataFrame:
    """
    For each dual variable, and for each year and scenario (or 'All' if no scenario),
    count total entries, non-zero entries, compute percent non-zero, and average dual value.

    Returns a DataFrame indexed by (dual, year, scenario) with columns:
      - total_duals
      - nonzero_duals
      - percent_nonzero
      - average_dual

    Optionally saves to “dual_year_scenario_summary.csv” in savefolder.
    """
    records = []
    for name, df in dual_variables.items():
        temp = df.reset_index()
        if "scenario" not in temp.columns:
            temp["scenario"] = "All"
        if "year" not in temp.columns:
            raise KeyError(f"{name}: missing 'year' index level")
        grp = temp.groupby(["year", "scenario"])["dual_value"]
        for (year, scen), series in grp:
            total = series.size
            nonzero = (series != 0).sum()
            pct = nonzero / total * 100 if total > 0 else 0.0
            avg = series.mean() if total > 0 else 0.0
            records.append(
                {
                    "dual": name,
                    "year": year,
                    "scenario": scen,
                    "total_duals": int(total),
                    "nonzero_duals": int(nonzero),
                    "percent_nonzero": pct,
                    "average_dual": avg,
                }
            )
    summary = pd.DataFrame(records).set_index(["dual", "year", "scenario"])
    if savefolder:
        summary.to_csv(f"{savefolder}/dual_year_scenario_summary.csv")
    return summary


def list_nonzero_gen_extension_duals(
    gen_extension_duals: pd.DataFrame, savefolder: str | None = None
) -> pd.DataFrame:
    """
    Identify which generators have non-zero extension duals in each year.

    Parameters
    ----------
    gen_extension_duals : pd.DataFrame
        MultiIndexed by (generator, year) with column 'dual_value'.
    savefolder : str or None
        Directory to save the results as CSV. If None, not saved.

    Returns
    -------
    pd.DataFrame
        Indexed by (year, generator) with column 'dual_value' for all non-zero entries.
    """
    # Flatten to columns
    df = gen_extension_duals.reset_index()[["year", "generator", "dual_value"]]
    # Filter non-zero
    nonzero = df[df["dual_value"] != 0].copy()
    # Sort by year, descending dual magnitude
    nonzero["abs_dual"] = nonzero["dual_value"].abs()
    nonzero = nonzero.sort_values(["year", "abs_dual"], ascending=[True, False])
    nonzero = nonzero.drop(columns=["abs_dual"])
    # Reindex
    result = nonzero.set_index(["year", "generator"])
    # Save if requested
    if savefolder:
        path = f"{savefolder}/gen_extension_nonzero_duals.csv"
        result.to_csv(path)
    return result

def make_lmp_bucket_label(key, low, up, mid):
    # Curtailment
    if key == "curtailment":
        return "Curtailment\n-100"
    # Solar, Wind, Thermal (positive)
    elif key == "solar":
        return f"Solar\n{mid:.3f}"
    elif key == "wind":
        return f"Wind\n{mid:.3f}"
    elif key == "thermal":
        return f"Thermal\n{mid:.2f}"
    # Negative buckets
    elif key.startswith("neg_"):
        pos_type = key.replace("neg_", "")
        label_base = {"solar": "Solar", "wind": "Wind", "thermal": "Thermal"}.get(pos_type, pos_type)
        # parenthesis and minus
        if pos_type in ["solar", "wind"]:
            return f"-{label_base}\n({mid:.3f})"
        else:
            return f"-{label_base}\n({mid:.2f})"
    # Negative gap
    elif key.startswith("gap_neg_"):
        # Use parenthesis around numbers and a single dash
        return f"({low:.2f})-({up:.2f})"
    # Positive gap
    elif key.startswith("gap"):
        return f"{low:.2f}-{up:.2f}"
    elif key.startswith(">"):
        return f"{key} €/MWh"
    else:
        return f"{key}\n{mid:.2f}"

def make_lmp_bucket_label(key, low, up, mid):
    if key == "curtailment":
        return "Curtailment\n-100"
    elif key == "solar":
        return f"Solar\n{mid:.3f}"
    elif key == "wind":
        return f"Wind\n{mid:.3f}"
    elif key == "thermal":
        return f"Thermal\n{mid:.2f}"
    # Negative mirrored buckets
    elif key.startswith("neg_"):
        pos_type = key.replace("neg_", "")
        label_base = {"solar": "Solar", "wind": "Wind", "thermal": "Thermal"}.get(pos_type, pos_type)
        if pos_type in ["solar", "wind"]:
            return f"-{label_base}\n({mid:.3f})"
        else:
            return f"-{label_base}\n({mid:.2f})"
    # Negative gap bucket: gap_neg_type1_type2
    elif key.startswith("gap_neg_"):
        return f"({low:.2f})-({up:.2f})"
    # Positive gap
    elif key.startswith("gap"):
        return f"{low:.2f}-{up:.2f}"
    elif key.startswith(">"):
        return f"{key} €/MWh"
    else:
        return f"{key}\n{mid:.2f}"

def plot_lmp_bucket_frequencies(
    freqs: pd.DataFrame,
    savefolder: str | None = None,
    min_freq_percent: float = 0.2,  # show only buckets >= 0.2% frequency
) -> None:
    """
    Bar chart of LMP bucket absolute frequencies with custom colors and labels,
    plus bold text labels above each bar and extended top margin.

    - Negative buckets: mirrored color of positive
    - Curtailment: hatched bar
    - Only show buckets with frequency >= min_freq_percent
    """
    # Colors
    color_solar = "#f9d002"
    color_on = "#235ebc"
    color_off = "#6895dd"
    wind_rgb = (
        (int(color_on[1:3], 16) + int(color_off[1:3], 16)) // 2,
        (int(color_on[3:5], 16) + int(color_off[3:5], 16)) // 2,
        (int(color_on[5:7], 16) + int(color_off[5:7], 16)) // 2,
    )
    color_wind = "#{:02x}{:02x}{:02x}".format(*wind_rgb)
    color_thermal = "#b20101"
    color_gap = "#888888"
    color_other = "black"

    color_dict = {
        "solar": color_solar,
        "wind": color_wind,
        "thermal": color_thermal,
        "curtailment": color_thermal,
    }

    # Compute min count based on percent threshold
    total = freqs["absolute"].sum()
    min_abs = max(1, int(np.floor(total * min_freq_percent / 100)))

    # Filter non-zero and above-threshold buckets
    df = freqs[(freqs["absolute"] >= min_abs)].copy()
    # Rename 'other' bucket
    non_other = df.drop(index="other", errors="ignore")
    max_upper = non_other["upper"].max() if not non_other.empty else 0
    if "other" in df.index:
        new_label = f">{max_upper:.2f}"
        df = df.rename(index={"other": new_label})
        df.at[new_label, "lower"] = max_upper
        df.at[new_label, "upper"] = max_upper
    # Sort
    df = df.sort_values("lower")
    buckets = list(df.index)
    lowers = df["lower"].values
    uppers = df["upper"].values
    mids = (lowers + uppers) / 2

    # Labels, colors, hatches
    labels = []
    colors = []
    hatches = []
    for key, low, up, mid in zip(buckets, lowers, uppers, mids):
        hatch = None
        # Use the helper function for label formatting
        labels.append(make_lmp_bucket_label(key, low, up, mid))
        # Colors and hatches as before
        if key == "solar":
            colors.append(color_solar)
        elif key == "wind":
            colors.append(color_wind)
        elif key == "thermal":
            colors.append(color_thermal)
        elif key.startswith("neg_"):
            pos_type = key.replace("neg_", "")
            colors.append(color_dict.get(pos_type, color_gap))
        elif key == "curtailment":
            colors.append(color_thermal)
            hatch = "///"
        elif key.startswith("gap") or key.startswith("gap_neg_"):
            colors.append(color_gap)
        elif key.startswith(">"):
            colors.append(color_other)
        else:
            colors.append(color_gap)
        hatches.append(hatch)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(buckets))
    bars = ax.bar(x, df["absolute"], color=colors, edgecolor="black")

    # Apply hatches for curtailment
    for i, (bar, hatch) in enumerate(zip(bars, hatches)):
        if hatch:
            bar.set_hatch(hatch)

    # Extend top margin
    max_count = df["absolute"].max() if not df["absolute"].empty else 1
    ax.set_ylim(0, max_count * 1.10)

    # Text labels
    for bar, cnt in zip(bars, df["absolute"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max_count * 0.02,
            f"{cnt}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Styling
    ax.set_xlabel("LMP Bucket [€/MWh]", fontsize=16)
    ax.set_ylabel("Frequency (timesteps)", fontsize=16)
    ax.set_xticks(x)
    # Only one set_xticklabels, 45 degrees, no ha argument!
    ax.set_xticklabels(labels, fontsize=14, rotation=45)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    print("LMP Bucket Absolute Frequencies")
    plt.show()

    # Save
    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "lmp_bucket_frequencies.png"), bbox_inches="tight"
        )


def plot_lmp_bucket_percentages(
    freqs: pd.DataFrame, savefolder: str | None = None, min_freq_percent: float = 0.2
) -> None:
    """
    Bar chart of LMP bucket relative frequencies (percent) with custom colors and labels,
    plus bold text labels above each bar and extended top margin to accommodate them.

    - Negative buckets: mirrored color of positive
    - Curtailment: hatched bar
    - Only show buckets with frequency >= min_freq_percent
    """
    # Colors
    color_solar = "#f9d002"
    color_on = "#235ebc"
    color_off = "#6895dd"
    wind_rgb = (
        (int(color_on[1:3], 16) + int(color_off[1:3], 16)) // 2,
        (int(color_on[3:5], 16) + int(color_off[3:5], 16)) // 2,
        (int(color_on[5:7], 16) + int(color_off[5:7], 16)) // 2,
    )
    color_wind = "#{:02x}{:02x}{:02x}".format(*wind_rgb)
    color_thermal = "#b20101"
    color_gap = "#888888"
    color_other = "black"

    color_dict = {
        "solar": color_solar,
        "wind": color_wind,
        "thermal": color_thermal,
        "curtailment": color_thermal,
    }

    # 1) Filter buckets above threshold
    df = freqs[freqs["relative_percent"] >= min_freq_percent].copy()
    # 2) Rename 'other' bucket
    non_other = df.drop(index="other", errors="ignore")
    max_upper = non_other["upper"].max() if not non_other.empty else 0
    if "other" in df.index:
        new_label = f">{max_upper:.2f}"
        df = df.rename(index={"other": new_label})
        df.at[new_label, "lower"] = max_upper
        df.at[new_label, "upper"] = max_upper
    # 3) Sort by lower bound
    df = df.sort_values("lower")
    buckets = list(df.index)
    lowers = df["lower"].values
    uppers = df["upper"].values
    mids = (lowers + uppers) / 2

    # 4) Prepare labels, colors, hatches
    labels = []
    colors = []
    hatches = []
    for key, low, up, mid in zip(buckets, lowers, uppers, mids):
        hatch = None
        labels.append(make_lmp_bucket_label(key, low, up, mid))
        # Positive/negative colors and hatches
        if key == "solar":
            colors.append(color_solar)
        elif key == "wind":
            colors.append(color_wind)
        elif key == "thermal":
            colors.append(color_thermal)
        elif key.startswith("neg_"):
            pos_type = key.replace("neg_", "")
            colors.append(color_dict.get(pos_type, color_gap))
        elif key == "curtailment":
            colors.append(color_thermal)
            hatch = "///"
        elif key.startswith("gap") or key.startswith("gap_neg_"):
            colors.append(color_gap)
        elif key.startswith(">"):
            colors.append(color_other)
        else:
            colors.append(color_gap)
        hatches.append(hatch)

    # 5) Plot bars
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(buckets))
    bars = ax.bar(x, df["relative_percent"], color=colors, edgecolor="black")

    # Apply hatches
    for i, (bar, hatch) in enumerate(zip(bars, hatches)):
        if hatch:
            bar.set_hatch(hatch)

    # 6) Extend top margin so labels fit
    max_pct = df["relative_percent"].max() if not df["relative_percent"].empty else 1
    ax.set_ylim(0, max_pct * 1.1)

    # 7) Add bold text labels above bars
    for bar, pct in zip(bars, df["relative_percent"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max_pct * 0.02,  # offset as 2% of max
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Styling
    ax.set_xlabel("LMP Bucket [€/MWh]", fontsize=16)
    ax.set_ylabel("Frequency (%)", fontsize=16)
    ax.set_xticks(x)
    # Only one set_xticklabels, 45 degrees, no ha argument
    ax.set_xticklabels(labels, fontsize=14, rotation=45)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    print("LMP Bucket Relative Frequencies (%)")
    plt.show()

    # Save if requested
    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "lmp_bucket_percentages.png"), bbox_inches="tight"
        )

def plot_lmp_bucket_percentages_by_year_scenario(
    power_balance_duals: pd.DataFrame,
    freqs: pd.DataFrame,
    savefolder: str | None = None,
    min_freq_percent: float = 0.2,
) -> None:
    """
    For each year & scenario, plot LMP bucket % frequencies using global freqs,
    including a correct ">cap" bin with upper=np.inf, and omitting buckets below min_freq_percent.
    Negative buckets: mirrored color; curtailment: hatched.
    """
    # Colors
    color_solar = "#f9d002"
    color_on = "#235ebc"
    color_off = "#6895dd"
    wind_rgb = (
        (int(color_on[1:3], 16) + int(color_off[1:3], 16)) // 2,
        (int(color_on[3:5], 16) + int(color_off[3:5], 16)) // 2,
        (int(color_on[5:7], 16) + int(color_off[5:7], 16)) // 2,
    )
    color_wind = "#{:02x}{:02x}{:02x}".format(*wind_rgb)
    color_thermal = "#b20101"
    color_gap = "#888888"
    color_other = "black"

    color_dict = {
        "solar": color_solar,
        "wind": color_wind,
        "thermal": color_thermal,
        "curtailment": color_thermal,
    }

    # Prepare global buckets
    df_buckets = freqs[["lower", "upper"]].copy()
    if "other" in df_buckets.index:
        cap = df_buckets.drop(index="other")["upper"].max()
        label = f">{cap:.2f}"
        df_buckets = df_buckets.rename(index={"other": label})
        df_buckets.at[label, "lower"] = cap
        df_buckets.at[label, "upper"] = np.inf
    df_buckets = df_buckets.sort_values("lower")

    keys = df_buckets.index.tolist()
    lowers = df_buckets["lower"].values
    uppers = df_buckets["upper"].values

    # --- Helper for label formatting (reuse from previous answers) ---
    def make_lmp_bucket_label(key, low, up, mid):
        if key == "curtailment":
            return "Curtailment\n-100"
        elif key == "solar":
            return f"Solar\n{mid:.3f}"
        elif key == "wind":
            return f"Wind\n{mid:.3f}"
        elif key == "thermal":
            return f"Thermal\n{mid:.2f}"
        elif key.startswith("neg_"):
            pos_type = key.replace("neg_", "")
            label_base = {"solar": "Solar", "wind": "Wind", "thermal": "Thermal"}.get(pos_type, pos_type)
            if pos_type in ["solar", "wind"]:
                return f"-{label_base}\n({mid:.3f})"
            else:
                return f"-{label_base}\n({mid:.2f})"
        elif key.startswith("gap_neg_"):
            return f"({low:.2f})-({up:.2f})"
        elif key.startswith("gap"):
            return f"{low:.2f}-{up:.2f}"
        elif key.startswith(">"):
            return f"{key} €/MWh"
        else:
            return f"{key}\n{mid:.2f}"

    # Build labels, colors, hatches (for all buckets)
    labels_full = []
    colors_full = []
    hatches_full = []
    for key, low, up in zip(keys, lowers, uppers):
        mid = (low + (up if np.isfinite(up) else low * 1.1)) / 2
        hatch = None
        labels_full.append(make_lmp_bucket_label(key, low, up, mid))
        # Color logic
        if key == "solar":
            colors_full.append(color_solar)
        elif key == "wind":
            colors_full.append(color_wind)
        elif key == "thermal":
            colors_full.append(color_thermal)
        elif key.startswith("neg_"):
            pos_type = key.replace("neg_", "")
            colors_full.append(color_dict.get(pos_type, color_gap))
        elif key == "curtailment":
            colors_full.append(color_thermal)
            hatch = "///"
        elif key.startswith("gap") or key.startswith("gap_neg_"):
            colors_full.append(color_gap)
        elif key.startswith(">"):
            colors_full.append(color_other)
        else:
            colors_full.append(color_gap)
        hatches_full.append(hatch)

    # Loop through years & scenarios
    for year in power_balance_duals.index.get_level_values("year").unique():
        scen_list = (
            power_balance_duals.xs(year, level="year")
            .index.get_level_values("scenario")
            .unique()
        )
        for scen in scen_list:
            sub = power_balance_duals.xs((scen, year), level=("scenario", "year"))
            vals = sub["dual_value"].values
            total = vals.size
            if total == 0:
                continue

            # Compute percents for each bucket
            rel = []
            for low, up in zip(lowers, uppers):
                if np.isfinite(up):
                    cnt = ((vals > low) & (vals <= up)).sum()
                else:
                    cnt = (vals > low).sum()
                rel.append(cnt / total * 100)

            # Apply frequency threshold
            mask = np.array(rel) >= min_freq_percent
            if not mask.any():
                continue
            x = np.arange(mask.sum())
            lbls = [labels_full[i] for i, m in enumerate(mask) if m]
            cols = [colors_full[i] for i, m in enumerate(mask) if m]
            hatches = [hatches_full[i] for i, m in enumerate(mask) if m]
            vals_pct = [rel[i] for i, m in enumerate(mask) if m]

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(x, vals_pct, color=cols, edgecolor="black")
            # Apply hatches for curtailment
            for i, (bar, hatch) in enumerate(zip(bars, hatches)):
                if hatch:
                    bar.set_hatch(hatch)
            max_pct = max(vals_pct)
            ax.set_ylim(0, max_pct * 1.10)
            for xi, pct in zip(x, vals_pct):
                ax.text(
                    xi,
                    pct + max_pct * 0.02,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

            # Styling
            ax.set_xlabel("LMP Bucket [€/MWh]", fontsize=16)
            ax.set_ylabel("Frequency (%)", fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(lbls, fontsize=14, rotation=45)
            ax.tick_params(axis="y", labelsize=14)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.set_title(f"{year} – {scen}", fontsize=16)
            plt.tight_layout()
            plt.show()

            # Save
            if savefolder:
                fname = f"lmp_buckets_{year}_{scen}.png"
                fig.savefig(os.path.join(savefolder, fname), bbox_inches="tight")
            plt.close()


def plot_high_lmp_event_counts(
    power_balance_duals: pd.DataFrame, cap: float, savefolder: str | None = None
) -> None:
    """
    Bar chart of the number of LMP events exceeding cap, by year and scenario.

    Parameters
    ----------
    power_balance_duals : pd.DataFrame
        MultiIndexed by (node, scenario, year, week, hour) with 'dual_value'.
    cap : float
        Threshold above which LMPs are considered 'high'.
    savefolder : str or None
        Directory to save the plot. If None, not saved.
    """
    # 1) Flatten and filter
    df = power_balance_duals.reset_index()[["year", "scenario", "dual_value"]]
    print(f"df {df}")
    high = df[df["dual_value"] > cap]
    print(f"=========================")
    print(f"high {high}")

    # 2) Count events per year & scenario
    counts = (
        high.groupby(["year", "scenario"])
        .size()
        .unstack("scenario")
        .fillna(0)
        .astype(int)
    )
    print(counts)

    # 3) Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="bar", ax=ax)

    # 4) Styling
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel(f"Events (LMP > {cap:.2f} €/MWh)", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Scenario", fontsize=16, title_fontsize=16)

    plt.tight_layout()
    print(f"LMP event counts above {cap:.2f} €/MWh by Year & Scenario")
    plt.show()

    # 5) Save if requested
    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "high_lmp_event_counts.png"), bbox_inches="tight"
        )


def plot_high_lmp_distribution(
    power_balance_duals: pd.DataFrame,
    cap: float,
    n_bins: int = 10,
    savefolder: str | None = None,
) -> None:
    """
    Histogram of LMP values exceeding cap, to show their distribution.

    Parameters
    ----------
    power_balance_duals : pd.DataFrame
        MultiIndexed by (node, scenario, year, week, hour) with 'dual_value'.
    cap : float
        Threshold above which LMPs are considered 'high'.
    n_bins : int
        Number of histogram bins between cap and the max value.
    savefolder : str or None
        Directory to save the figure. If None, not saved.
    """
    # 1) Extract high LMPs
    vals = power_balance_duals.reset_index()["dual_value"]
    high_vals = vals[vals > cap]
    if high_vals.empty:
        print(f"No LMP values above {cap:.2f} €/MWh to plot.")
        return

    # 2) Determine bins
    min_edge = cap
    max_edge = high_vals.max()
    bins = np.linspace(min_edge, max_edge, n_bins + 1)

    # 3) Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(high_vals, bins=bins, edgecolor="black")

    # 4) Styling
    ax.set_xlabel("LMP (€/MWh)", fontsize=16)
    ax.set_ylabel("Frequency (timesteps)", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    print(f"Distribution of LMPs > {cap:.2f} €/MWh")
    plt.show()

    # 5) Save if requested
    if savefolder:
        fig.savefig(
            os.path.join(savefolder, f"high_lmp_distribution_{int(cap)}plus.png"),
            bbox_inches="tight",
        )


def make_high_lmp_frequency_table(
    power_balance_duals: pd.DataFrame,
    cap: float,
    n_bins: int = 10,
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Create a table of frequencies for LMP values exceeding cap,
    dividing [cap, max(LMP)] into n_bins.

    Parameters
    ----------
    power_balance_duals : pd.DataFrame
        MultiIndexed by (node, scenario, year, week, hour) with 'dual_value'.
    cap : float
        Lower threshold; only values > cap are considered.
    n_bins : int, optional
        Number of equal-width bins in [cap, max_value].
    savefolder : str or None, optional
        Directory to save CSV. If None, not saved.

    Returns
    -------
    pd.DataFrame
        Columns:
          - lower_bound: left edge of bin
          - upper_bound: right edge of bin
          - count: number of timesteps in this bin
          - percent_of_high: % of high events in this bin
          - percent_of_total: % of all timesteps in this bin
    """
    vals = power_balance_duals.reset_index()["dual_value"]
    total = len(vals)
    high = vals[vals > cap]
    n_high = len(high)
    # If no high values, return empty table
    if n_high == 0:
        cols = [
            "lower_bound",
            "upper_bound",
            "count",
            "percent_of_high",
            "percent_of_total",
        ]
        return pd.DataFrame(columns=cols)
    # Bin edges
    min_edge = cap
    max_edge = high.max()
    edges = np.linspace(min_edge, max_edge, n_bins + 1)
    # Categorize
    cats = pd.cut(high, bins=edges, include_lowest=True, right=True)
    freq = cats.value_counts().sort_index()
    # Build table
    rows = []
    for interval, cnt in freq.items():
        rows.append(
            {
                "lower_bound": interval.left,
                "upper_bound": interval.right,
                "count": int(cnt),
                "percent_of_high": cnt / n_high * 100,
                "percent_of_total": cnt / total * 100,
            }
        )
    table = pd.DataFrame(rows)
    if savefolder:
        table.to_csv(
            os.path.join(savefolder, f"high_lmp_freq_table_{int(cap)}plus.csv")
        )
    return table


def get_top_high_lmp_buckets(
    high_freq_table: pd.DataFrame, top_n: int = 5, savefolder: str | None = None
) -> pd.DataFrame:
    """
    Return the top_n bins with the highest counts from the high-LMP frequency table.
    """
    top_bins = high_freq_table.sort_values("count", ascending=False).head(top_n)
    if savefolder:
        top_bins.to_csv(os.path.join(savefolder, f"top_{top_n}_high_lmp_bins.csv"))
    return top_bins


def plot_top_high_lmp_buckets(
    high_freq_table: pd.DataFrame, top_n: int = 5, savefolder: str | None = None
) -> None:
    """
    Bar chart of the top_n high-LMP bins by count.
    """
    top_bins = high_freq_table.sort_values("count", ascending=False).head(top_n)
    labels = [
        f"{row['lower_bound']:.0f}-{row['upper_bound']:.0f}"
        for _, row in top_bins.iterrows()
    ]
    counts = top_bins["count"].values
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, edgecolor="black")
    # annotate
    max_count = counts.max()
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            cnt + max_count * 0.02,
            f"{cnt}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_xlabel("High-LMP bin [€/MWh]", fontsize=16)
    ax.set_ylabel("Frequency (timesteps)", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Stretch y-axis by 0.05 (5%) above the tallest bar
    ax.set_ylim(0, max_count * 1.10)

    plt.tight_layout()
    print(f"Top {top_n} high-LMP bins by occurrence")
    plt.show()
    if savefolder:
        fig.savefig(
            os.path.join(savefolder, f"top_{top_n}_high_lmp_bins.png"),
            bbox_inches="tight",
        )


def plot_top_high_bucket_detail(
    power_balance_duals: pd.DataFrame,
    high_freq_table: pd.DataFrame,
    width: float = 1.0,
    savefolder: str | None = None,
) -> None:
    """
    For the highest-frequency high-LMP bin, re-bin its LMP values into width-sized bins (default 1 €/MWh)
    and plot their distribution.

    Parameters
    ----------
    power_balance_duals : pd.DataFrame
        MultiIndexed by (node, scenario, year, week, hour) with column 'dual_value'.
    high_freq_table : pd.DataFrame
        Output of make_high_lmp_frequency_table(...), must contain 'lower_bound' and 'upper_bound'.
    width : float
        Width of the new detailed bins (default 1 €/MWh).
    savefolder : str or None
        Directory to save the figure. If None, not saved.
    """
    # 1) Identify top bucket
    top = high_freq_table.sort_values("count", ascending=False).iloc[0]
    low = top["lower_bound"]
    high = top["upper_bound"]

    # 2) Extract values in that interval
    vals = power_balance_duals.reset_index()["dual_value"]
    subset = vals[(vals > low) & (vals <= high)]

    if subset.empty:
        print(f"No LMPs found in top bucket interval ({low:.2f}, {high:.2f}].")
        return

    # 3) Build detailed bins
    start = np.floor(low)
    end = np.ceil(high)
    bins = np.arange(start, end + width, width)

    # 4) Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(subset, bins=bins, edgecolor="black")

    # 5) Styling
    ax.set_xlabel("LMP [€/MWh]", fontsize=16)
    ax.set_ylabel("Frequency (timesteps)", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    print(f"Detailed histogram for top high-LMP bucket ({low:.2f}–{high:.2f} €/MWh)")
    plt.show()

    # 6) Save if requested
    if savefolder:
        filename = f"top_bucket_detail_{int(low)}_{int(high)}_bin{int(width)}.png"
        fig.savefig(os.path.join(savefolder, filename), bbox_inches="tight")


def extract_duals_in_lmp_bucket(
    power_balance_duals: pd.DataFrame, freqs: pd.DataFrame, bucket: str
) -> pd.DataFrame:
    """
    Return all power‐balance duals whose LMP falls in the named bucket.
    Uses freqs[['lower','upper']] rather than lower_bound/upper_bound.
    """
    # 1) Grab the numeric bounds
    try:
        lower = freqs.loc[bucket, "lower"]
        upper = freqs.loc[bucket, "upper"]
    except KeyError:
        raise KeyError(f"Bucket '{bucket}' not found in freqs index.")

    # 2) Filter the duals
    df = power_balance_duals.reset_index()
    mask = (df["dual_value"] > lower) & (df["dual_value"] <= upper)
    subset = df[mask].copy()

    # 3) Restore original MultiIndex
    return subset.set_index(power_balance_duals.index.names)


def summarize_gap_duals_by_year_scenario(
    subset: pd.DataFrame,
    power_balance_duals: pd.DataFrame,
    savefolder: str | None = None,
) -> pd.DataFrame:
    """
    Summarize how many timesteps in each (year, scenario) fall into the extracted bucket.
    """
    # total hours per year/scenario
    flat = power_balance_duals.reset_index()[["year", "scenario"]]
    total = flat.groupby(["year", "scenario"]).size().rename("count_total")
    # hours in the bucket
    bucket = subset.reset_index()[["year", "scenario"]]
    bucket_counts = bucket.groupby(["year", "scenario"]).size().rename("count_bucket")
    # combine
    summary = pd.concat([total, bucket_counts], axis=1).fillna(0).astype(int)
    summary["percent_bucket_of_total"] = (
        summary["count_bucket"] / summary["count_total"] * 100
    )
    if savefolder:
        summary.to_csv(f"{savefolder}/gap_wind_thermal_dual_summary.csv")
    return summary


def plot_gap_lmp_distribution(
    subset: pd.DataFrame, bucket: str, savefolder: str | None = None
) -> None:
    """
    Histogram of LMPs in the given bucket interval.
    """
    vals = subset["dual_value"]
    if vals.empty:
        print(f"No duals found in bucket '{bucket}'.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=20, edgecolor="black")
    ax.set_xlabel("LMP [€/MWh]", fontsize=16)
    ax.set_ylabel("Frequency (timesteps)", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    print(f"Distribution of LMPs in bucket '{bucket}'")
    plt.show()

    if savefolder:
        fig.savefig(f"{savefolder}/gap_{bucket}_distribution.png", bbox_inches="tight")


def make_annual_lmp_table(
    power_balance_duals: pd.DataFrame, savefolder: str | None = None
) -> pd.DataFrame:
    """
    Build a table of average LMP by year and scenario, plus overall average.
    """
    df = power_balance_duals.reset_index()[["year", "scenario", "dual_value"]]
    table = df.groupby(["year", "scenario"])["dual_value"].mean().unstack("scenario")
    table["Average"] = table.mean(axis=1)

    if savefolder:
        # 1) Save the full table
        table.to_csv(f"{savefolder}/annual_lmp_table.csv")

        # 2) Save a one‐row CSV with the overall average across years
        overall_avg = table["Average"].mean()
        avg_df = pd.DataFrame({"Average_LMP_across_years_€/MWh": [overall_avg]})
        avg_df.to_csv(f"{savefolder}/annual_lmp_overall_average.csv")
        print(f"Average LMP across years: {overall_avg:.2f} €/MWh")

    return table


def plot_annual_lmp_by_scenario(
    power_balance_duals: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Grouped bar chart of mean LMP by year and scenario with ±1 std error bars.
    """
    # Flatten
    df = power_balance_duals.reset_index()[["year", "scenario", "dual_value"]]
    # Compute stats
    stats = (
        df.groupby(["year", "scenario"])["dual_value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    years = sorted(stats["year"].unique())
    scenarios = sorted(stats["scenario"].unique())

    # Pivot
    mean_df = stats.pivot(index="year", columns="scenario", values="mean")
    std_df = stats.pivot(index="year", columns="scenario", values="std")

    # Bar plotting
    x = np.arange(len(years))
    n = len(scenarios)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, scen in enumerate(scenarios):
        xs = x - 0.4 + i * width + width / 2
        ax.bar(xs, mean_df[scen], width, yerr=std_df[scen], label=scen, capsize=5)

    # Styling
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("Average LMP (€/MWh)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(
        title="Scenario",
        fontsize=16,
        title_fontsize=16,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    print("Average LMP by Year and Scenario (±1 std)")
    plt.show()

    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "annual_lmp_by_scenario.png"), bbox_inches="tight"
        )


def plot_annual_lmp_by_scenario_simple(
    power_balance_duals: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Grouped bar chart of mean LMP by year and scenario without error bars.
    """
    # Flatten
    df = power_balance_duals.reset_index()[["year", "scenario", "dual_value"]]
    # Compute mean per year & scenario
    mean_df = df.groupby(["year", "scenario"])["dual_value"].mean().unstack("scenario")
    # Plot setup
    years = mean_df.index.tolist()
    scenarios = mean_df.columns.tolist()
    x = np.arange(len(years))
    n = len(scenarios)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, scen in enumerate(scenarios):
        xs = x - 0.4 + i * width + width / 2
        ax.bar(xs, mean_df[scen], width, label=scen)

    # Styling
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("Average LMP (€/MWh)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(
        title="Scenario",
        fontsize=16,
        title_fontsize=16,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    print("Average LMP by Year and Scenario")
    plt.show()

    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "annual_lmp_by_scenario_simple.png"),
            bbox_inches="tight",
        )


def plot_annual_lmp_by_scenario_line(
    power_balance_duals: pd.DataFrame, savefolder: str | None = None
) -> None:
    """
    Line chart of mean LMP by year and scenario.

    - One line per scenario.
    - X-axis: years; Y-axis: average LMP (€/MWh).
    - Grid lines on y-axis (linestyle '--', alpha=0.5).
    - Axis labels fontsize 16; tick labels fontsize 14; legend fontsize 16.
    """
    # 1) Flatten and compute mean
    df = power_balance_duals.reset_index()[["year", "scenario", "dual_value"]]
    mean_df = df.groupby(["year", "scenario"])["dual_value"].mean().unstack("scenario")

    # 2) Prepare plot
    years = mean_df.index.tolist()
    scenarios = mean_df.columns.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    # 3) Plot each scenario as a line
    for scen in scenarios:
        ax.plot(years, mean_df[scen], marker="o", linestyle="-", label=scen)

    # 4) Styling
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("Average LMP (€/MWh)", fontsize=16)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(
        title="Scenario",
        fontsize=16,
        title_fontsize=16,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()
    print("Average LMP by Year and Scenario (line chart)")
    plt.show()

    # 5) Save if requested
    if savefolder:
        fig.savefig(
            os.path.join(savefolder, "annual_lmp_by_scenario_line.png"),
            bbox_inches="tight",
        )





def analyze_run_stochastic(
    model_config: dict,
    SAVE_FIGURES: bool = True,
    SAVE_TABLES: bool = True,
    show_plots: bool = False,
) -> None:
    print(30 * "-")
    print("Analyzing model run...")
    print(30 * "-")
    print(model_config)

    if not show_plots:
        original_show = plt.show

        # Override plt.show with a no-op lambda.
        plt.show = lambda: None

    FOLDER = model_config["save_folder"]
    decision_variables_folder = os.path.join(FOLDER, "decision_variables")
    model_info_folder = os.path.join(FOLDER, "model_info")
    dual_variables_folder = os.path.join(FOLDER, "dual_variables")

    # Create folders if they don't exist

    RESULTS_FOLDER = os.path.join(FOLDER, "results")
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    if SAVE_TABLES:
        tables_folder = os.path.join(RESULTS_FOLDER, "tables")
        if not os.path.exists(tables_folder):
            os.makedirs(tables_folder)
    else:
        tables_folder = None
    if SAVE_FIGURES:
        figures_folder = os.path.join(RESULTS_FOLDER, "figures")
        if not os.path.exists(figures_folder):
            os.makedirs(figures_folder)
    else:
        figures_folder = None

    if SAVE_FIGURES:
        generators_save_folder = os.path.join(figures_folder, "generators")
        if not os.path.exists(generators_save_folder):
            os.makedirs(generators_save_folder)
    else:
        generators_save_folder = None

    if SAVE_TABLES:
        generators_save_table_folder = os.path.join(tables_folder, "generators")
        if not os.path.exists(generators_save_table_folder):
            os.makedirs(generators_save_table_folder)
    else:
        generators_save_table_folder = None

    if SAVE_FIGURES:
        branches_save_folder = os.path.join(figures_folder, "branches")
        if not os.path.exists(branches_save_folder):
            os.makedirs(branches_save_folder)
    else:
        branches_save_folder = None
    if SAVE_TABLES:
        branches_save_table_folder = os.path.join(tables_folder, "branches")
        if not os.path.exists(branches_save_table_folder):
            os.makedirs(branches_save_table_folder)
    else:
        branches_save_table_folder = None

    if SAVE_FIGURES:
        batteries_save_folder = os.path.join(figures_folder, "batteries")
        if not os.path.exists(batteries_save_folder):
            os.makedirs(batteries_save_folder)
    else:
        batteries_save_folder = None
    if SAVE_TABLES:
        batteries_save_table_folder = os.path.join(tables_folder, "batteries")
        if not os.path.exists(batteries_save_table_folder):
            os.makedirs(batteries_save_table_folder)
        else:
            batteries_save_table_folder = None

    if SAVE_FIGURES:
        macro_save_folder = os.path.join(figures_folder, "macro")
        if not os.path.exists(macro_save_folder):
            os.makedirs(macro_save_folder)
    else:
        macro_save_folder = None
    if SAVE_TABLES:
        macro_save_table_folder = os.path.join(tables_folder, "macro")
        if not os.path.exists(macro_save_table_folder):
            os.makedirs(macro_save_table_folder)
    else:
        macro_save_table_folder = None

    if SAVE_FIGURES:
        # Create the figures folder if it doesn't exist
        dual_folder = os.path.join(RESULTS_FOLDER, "duals")
        os.makedirs(dual_folder, exist_ok=True)
    else:
        dual_folder = None
    if SAVE_TABLES:
        # Create the tables folder if it doesn't exist
        dual_save_table_folder = os.path.join(tables_folder, "duals")
        os.makedirs(dual_save_table_folder, exist_ok=True)
    else:
        dual_save_table_folder = None

    ## Read data
    model_info = pd.read_csv(os.path.join(model_info_folder, "model_info.csv"))
    config = yaml.safe_load(open(os.path.join(model_info_folder, "config.yaml")))
    jsons = utils.read_jsons_from_dir(model_info_folder)
    scenarios = jsons["scenarios"]
    scenario_probabilities = jsons["scenario_probabilities"]
    week_weights = jsons["week_weights"]
    data_folder_name = config["data_folder_name"]
    VOLL = config["VOLL"]
    CC = config["CC"]
    CO2_price = config["CO2_price"]
    E_limit = config["E_limit"]
    p_max_new_branch = config["p_max_new_branch"]
    p_min_new_branch = config["p_min_new_branch"]
    expansion_factor = config["expansion_factor"]
    MS = config["MS"]
    model_name = config["model_name"]
    MIPGap = config["MIPGap"]
    years = config["years"]
    r = config["discount_rate"]
    representative_period_unit = config["representative_period_unit"]
    weeks = config["representative_periods"]
    scenario_file = config["scenario_file"]
    yearly_discount = config.get("yearly_discount", 10)
    input_data_folder = os.path.join(PROCESSED_DATA_FOLDER, config["data_folder_name"])
    input_data = utils.load_multi_year_csv_files_with_week_from_folder(
        years=years, data_folder_path=input_data_folder, yearly_discount=yearly_discount
    )
    scenario_multiplier = utils.load_scenario_multiplier(
        scenario_file_name=scenario_file
    )
    # **Note:** The input data is such that all data is the same for all years. All the data that is year-dependent has year as index. The only exception is the demand data, which does not have copies for years.
    dual_variables = utils.load_csv_files_from_folder_with_scenarios(
        dual_variables_folder
    )
    decision_varables = utils.load_csv_files_from_folder_with_scenarios(
        decision_variables_folder
    )

    # Put data into variables for easier access
    # read dataframes
    # input data
    batteries = input_data["batteries"]
    branches = input_data["branches"]
    generators = input_data["generators"]
    capacity_factors = input_data["capacity_factors"]
    generator_costs = input_data["generator_costs"]
    hourly_demand = input_data["hourly_demand"]
    nodes = input_data["nodes"]
    # decision variables
    battery_capacity = decision_varables["battery_capacity"]
    battery_charging = decision_varables["battery_charging"]
    battery_discharging = decision_varables["battery_discharging"]
    battery_soc = decision_varables["battery_soc"]
    branch_capacity = decision_varables["branch_capacity"]
    curtailment = decision_varables["curtailment"]
    generation = decision_varables["generation"]
    generator_capacity = decision_varables["generator_capacity"]
    load_shedding = decision_varables["load_shedding"]
    power_flow = decision_varables["power_flow"]

    # region Generators
    # Preprocess generators
    generators["extension_potential"] = generators["p_nom"] * config["expansion_factor"]

    generators = add_capacity_and_cumulative_metrics(generators, generator_capacity)

    if SAVE_FIGURES:
        generators_save_folder = os.path.join(figures_folder, "generators")
        if not os.path.exists(generators_save_folder):
            os.makedirs(generators_save_folder)
    else:
        generators_save_folder = None

    plot_capacity_investment_by_carrier(generators, generators_save_folder)
    plot_capacity_spending_by_carrier(generators, generators_save_folder)
    plot_total_capacity_growth(generators, generators_save_folder)
    plot_extension_vs_potential_by_carrier(generators, generators_save_folder)
    production_by_carrier_table = create_energy_production_by_carrier_table(
        generation,
        generators,
        scenarios,
        scenario_probabilities,
        week_weights,
        savefolder=generators_save_table_folder,
    )
    make_annual_production_table(
        generation,
        generators,
        scenarios,
        scenario_probabilities,
        week_weights=week_weights,
        savefolder=generators_save_table_folder,
    )
    annual_cost_table = make_weighted_annual_production_cost_by_year_table(
        generation,
        generators,
        scenarios,
        scenario_probabilities,
        week_weights,
        carbon_price=CO2_price,
        savefolder=generators_save_table_folder,
    )
    weighted_annual_production_by_year = make_weighted_annual_production_by_year(
        generation,
        generators,
        scenarios,
        scenario_probabilities,
        week_weights=week_weights,
        savefolder=generators_save_table_folder,
    )
    plot_weighted_production_evolution(
        weighted_annual_production_by_year,
        generators,
        savefolder=generators_save_folder,
    )
    plot_weighted_production_evolution_no_legend(
        weighted_annual_production_by_year,
        generators,
        savefolder=generators_save_folder,
    )
    plot_weighted_production_cost_evolution(
        annual_cost_table, generators, savefolder=generators_save_folder
    )
    yearly_system_cost_table = make_yearly_system_costs_table(
        generation, generators, week_weights, CO2_price, macro_save_table_folder
    )
    plot_co2_emissions_by_scenario(
        yearly_system_cost_table, savefolder=generators_save_folder
    )
    plot_co2_emissions_by_scenario_avg(yearly_system_cost_table, generators_save_folder)
    plot_production_by_scenario(yearly_system_cost_table, generators_save_folder)
    plot_production_cost_by_scenario(yearly_system_cost_table, generators_save_folder)
    plot_production_cost_with_emission_by_scenario(
        yearly_system_cost_table, generators_save_folder
    )

    # Example loop over years, weeks, scenarios, and carrier types
    years = [int(y) for y in scenarios.keys()]
    weeks = [int(w) for w in week_weights.keys()]
    carriers = generators.reset_index()["carrier"].unique()
    # if figures_folder:
    #     generation_curves_folder = os.path.join(generators_save_folder, "generation_curves")
    #     os.makedirs(generation_curves_folder, exist_ok=True)
    # else:
    #     generation_curves_folder = None

    # for year in years:
    #     for scenario in scenarios[str(year)]:
    #         for week in weeks:
    #             for carrier in carriers:
    #                 plot_generation_curves_by_carrier(
    #                     generation,
    #                     generators,
    #                     year,
    #                     week,
    #                     scenario,
    #                     carrier,
    #                     savefolder=generation_curves_folder,  # or None if you don’t want to save
    #                 )

    # years = [int(y) for y in scenarios.keys()]
    # weeks = [int(w) for w in week_weights.keys()]
    # for year in years:
    #     for scen in scenarios[str(year)]:
    #         for week in weeks:
    #             analytics.plot_utilization_time_series_by_carrier(
    #                 generation,
    #                 capacity_factors,
    #                 generators,
    #                 year,
    #                 week,
    #                 scen,
    #                 savefolder=generators_save_table_folder,
    #             )

    # years = [int(y) for y in scenarios.keys()]
    # weeks = [int(w) for w in week_weights.keys()]
    # carriers = generators.reset_index()["carrier"].unique()

    # for year in years:
    #     for scen in scenarios[str(year)]:
    #         for week in weeks:
    #             for carrier in carriers:
    #                 plot_fraction_of_max_generation_by_carrier(
    #                     generation,
    #                     capacity_factors,
    #                     generators,
    #                     year,
    #                     week,
    #                     scen,
    #                     carrier,
    #                     savefolder=generation_curves_folder,  # or None if you don’t want to save
    #                 )

    plot_utilization_hierarchy(
        generation,
        capacity_factors,
        generators,
        week_weights,
        scenarios,
        generators_save_folder,
    )

    summary_table = make_system_summary_table(
        generation,
        generators,
        week_weights,
        scenarios,
        scenario_probabilities,
        CO2_price,
        savefolder=macro_save_table_folder,
    )
    summary_by_carrier_table = make_system_summary_table_by_carrier(
        generation,
        generators,
        week_weights,
        scenarios,
        scenario_probabilities,
        CO2_price,
        macro_save_table_folder,
    )
    # Preprocessing
    if "extended_by" in branches.columns:
        branches.drop(columns=["extended_by"], inplace=True)
    if "branch" in branch_capacity.index.names:
        branch_capacity = branch_capacity.rename_axis(index={"branch": "line"})
    if "branch" in power_flow.index.names:
        power_flow = power_flow.rename_axis(index={"branch": "line"})
    branches["extension_potential"] = p_max_new_branch

    branches = extend_branches_table(branches, branch_capacity)
    plot_branch_new_capacity(branches, branches_save_folder)
    branch_buildout_summary = make_branch_buildout_summary(
        branches, branches_save_table_folder
    )
    branch_flow_table_per_line = make_branch_flow_metrics(
        power_flow, branches, week_weights, savefolder=branches_save_table_folder
    )
    branch_flow_summary = make_aggregate_branch_flow_summary(
        power_flow, branches, week_weights, savefolder=branches_save_table_folder
    )
    plot_new_branches_for_years_with_investments(
        branches, branch_capacity, nodes, savefolder=branches_save_folder
    )

    if len(batteries) > 0:
        batteries = extend_batteries_table(batteries, battery_capacity)
        plot_battery_investment_by_year(batteries, savefolder=batteries_save_folder)
        plot_battery_investment_cost_by_year(
            batteries, savefolder=batteries_save_folder
        )
        battery_summary = make_battery_system_summary(
            batteries,
            battery_discharging,
            week_weights,
            scenarios,
            scenario_probabilities,
            tables_folder,
        )
        plot_cycles_per_scenario_with_average(
            battery_discharging,
            week_weights,
            reference_cycle_mwh=400,
            savefolder=batteries_save_folder,
        )

    plot_curtailment_by_scenario(curtailment, week_weights, savefolder=figures_folder)
    plot_load_shedding_by_scenario(
        load_shedding, week_weights, savefolder=figures_folder
    )
    curtailment_and_load_shedding_table = make_curtailment_load_shedding_cost_table(
        curtailment, load_shedding, week_weights, CC, VOLL, savefolder=tables_folder
    )

    # Macro analysis

    system_summary = make_system_summary_table(
        generation,
        generators,
        week_weights,
        scenarios,
        scenario_probabilities,
        CO2_price,
        savefolder=macro_save_table_folder,
    )
    if len(batteries) > 0:
        battery_summary = make_battery_system_summary(
            batteries,
            battery_discharging,
            week_weights,
            scenarios,
            scenario_probabilities,
            savefolder=batteries_save_table_folder,
        )
    else:
        # Dummy columns
        columns = ["num_buildout", "capacity", "investment_cost", "cycles_per_year"]

        # Create the DataFrame with NaNs
        empty_df = pd.DataFrame(index=years, columns=columns)
        empty_df.index.name = "year"
        battery_summary = empty_df

    branch_summary = make_branch_buildout_summary(branches, branches_save_table_folder)
    yearly_system_cost_table = make_yearly_system_costs_table(
        generation,
        generators,
        week_weights,
        CO2_price,
        savefolder=macro_save_table_folder,
    )
    curtailment_and_load_shedding_table = make_curtailment_load_shedding_cost_table(
        curtailment, load_shedding, week_weights, CC, VOLL, savefolder=tables_folder
    )

    plot_yearly_costs_stacked_avg(
        system_summary,
        battery_summary,
        branch_summary,
        yearly_system_cost_table,
        curtailment_and_load_shedding_table,
        savefolder=macro_save_folder,
    )
    plot_yearly_costs_stacked_by_scenario(
        system_summary,
        battery_summary,
        branch_summary,
        yearly_system_cost_table,
        curtailment_and_load_shedding_table,
        savefolder=macro_save_folder,
    )
    total_system_cost = compute_total_system_cost(
        system_summary,
        battery_summary,
        branch_summary,
        yearly_system_cost_table,
        curtailment_and_load_shedding_table,
        scenarios,
        scenario_probabilities,
        savefolder=macro_save_table_folder,
    )
    print(total_system_cost)
    assert (
        abs(total_system_cost["total_cost"].sum() - model_info["Objective Value"][0])
        < 1000
    ), f"Mismatch in total system cost: {total_system_cost['total_cost'].sum():,.0f} vs {model_info['Objective Value'][0]:,.0f}"

    mega_table = make_mega_cost_table(
        system_summary,
        battery_summary,
        branch_summary,
        yearly_system_cost_table,
        curtailment_and_load_shedding_table,
        scenarios,
        scenario_probabilities,
        savefolder=macro_save_table_folder,
    )
    # End  by saving the extended generators, transmission lines, and batteries tables to and extended tables folder
    if tables_folder:
        # Create the extended tables folder if it doesn't exist
        extended_tables_folder = os.path.join(tables_folder, "extended_tables")
        os.makedirs(extended_tables_folder, exist_ok=True)
        generators.to_csv(
            os.path.join(extended_tables_folder, "extended_generators.csv")
        )
        branches.to_csv(os.path.join(extended_tables_folder, "extended_branches.csv"))
        batteries.to_csv(os.path.join(extended_tables_folder, "extended_batteries.csv"))

    # region Dual Variables
    if SAVE_FIGURES:
        # Create the figures folder if it doesn't exist
        duals_folder = os.path.join(RESULTS_FOLDER, "duals")
        os.makedirs(duals_folder, exist_ok=True)
        tables_folder = os.path.join(duals_folder, "tables")
        figures_folder = os.path.join(duals_folder, "figures")
        os.makedirs(tables_folder, exist_ok=True)
        os.makedirs(figures_folder, exist_ok=True)

    # Preprocessing dual variables so scenario-duals are probability-adjusted.
    # Build a DataFrame of scenario probabilities (p_{ω,y})
    prob_rows = []
    for year_str, scen_list in scenarios.items():
        year = int(year_str)
        probs = scenario_probabilities[year_str]
        for scen, p in zip(scen_list, probs):
            prob_rows.append({"year": year, "scenario": scen, "p": p})
    prob_df = pd.DataFrame(prob_rows).set_index(["year", "scenario"])

    # Loop through duals and apply 1/p weighting where appropriate
    for key, df in dual_variables.items():
        # Only apply to duals that vary by scenario
        if "scenario" in df.index.names:
            temp = df.reset_index().merge(prob_df, on=["year", "scenario"], how="left")
            # Compute 1/p weight
            temp["weight"] = 1.0 / temp["p"]
            # Apply weight to the dual values
            temp["dual_value"] = temp["dual_value"] * temp["weight"]

            # Reconstruct a weighted-dual Series with the original index
            weighted = temp.set_index(df.index.names)
            # Display the first two rows
            dual_variables[key] = weighted

    # read dual variables
    battery_charge_new_max_duals = dual_variables["battery_charge_new_max_duals"]
    battery_charge_old_duals = dual_variables["battery_charge_old_duals"]
    battery_discharge_new_max_duals = dual_variables["battery_discharge_new_max_duals"]
    battery_discharge_old_duals = dual_variables["battery_discharge_old_duals"]
    branch_extension_duals = dual_variables["branch_extension_duals"]
    branch_flow_old_duals_min = dual_variables["branch_flow_old_duals_min"]
    branch_flow_old_duals_max = dual_variables["branch_flow_old_duals_max"]
    branch_flow_new_duals_min = dual_variables["branch_flow_new_duals_min"]
    branch_flow_new_duals_max = dual_variables["branch_flow_new_duals_max"]
    emissions_duals = dual_variables["emissions_duals"]
    gen_extension_duals = dual_variables["gen_extension_duals"]
    gen_output_new_duals = dual_variables["gen_output_new_duals"]
    gen_output_old_duals = dual_variables["gen_output_old_duals"]
    load_shedding_duals = dual_variables["load_shedding_duals"]
    power_balance_duals = dual_variables["power_balance_duals"]

    lmp_table = make_nodal_price_table(
        power_balance_duals, savefolder=dual_save_table_folder
    )
    annual_lmp_table = make_annual_lmp_table(
        power_balance_duals, savefolder=dual_save_table_folder
    )

    plot_annual_lmp_by_scenario(power_balance_duals, savefolder=dual_folder)
    plot_annual_lmp_by_scenario_simple(power_balance_duals, savefolder=dual_folder)
    plot_annual_lmp_by_scenario_line(power_balance_duals, savefolder=dual_folder)
    plot_nodal_price_evolution_with_markers(lmp_table, savefolder=dual_folder)
    plot_nodal_price_grouped_bar(
        lmp_table, node_to_city=plotting.node_to_city, savefolder=dual_folder
    )
    branch_extension_duals = branch_extension_duals.rename_axis(
        index={"branch": "line"}
    )
    generators["production_cost"] = (
        generators["marginal_cost"] + generators["co2_emissions"] * CO2_price
    )
    freqs = compute_lmp_bucket_frequencies(
        power_balance_duals, generators, dual_save_table_folder
    )

    plot_lmp_bucket_frequencies(freqs, savefolder=dual_folder)
    # determine cap from your buckets:
    cap = freqs.loc[freqs.index != "other", "upper"].max()
    plot_lmp_bucket_percentages(freqs, savefolder=dual_folder)
    plot_lmp_bucket_percentages_by_year_scenario(
        power_balance_duals, freqs, savefolder=dual_folder
    )

    if freqs.loc["other", "absolute"] > 0.01:

        # determine cap from your buckets:
        cap = freqs.loc[freqs.index != "other", "upper"].max()

        plot_high_lmp_event_counts(power_balance_duals, cap, savefolder=dual_folder)
        plot_high_lmp_distribution(
            power_balance_duals, cap, n_bins=100, savefolder=dual_folder
        )
        high_tbl = make_high_lmp_frequency_table(
            power_balance_duals, cap=cap, n_bins=100, savefolder=dual_save_table_folder
        )
        top5 = get_top_high_lmp_buckets(
            high_tbl, top_n=10, savefolder=dual_save_table_folder
        )
        plot_top_high_lmp_buckets(high_tbl, top_n=6, savefolder=dual_folder)
        plot_top_high_bucket_detail(
            power_balance_duals, high_tbl, width=1.0, savefolder=dual_folder
        )
        plot_lmp_histogram_70plus(
           power_balance_duals, cap=70, bin_width=2.0, savefolder=dual_folder
        )
    gap_subset = extract_duals_in_lmp_bucket(
        power_balance_duals, freqs, "gap_wind_thermal"
    )
    gap_summary = summarize_gap_duals_by_year_scenario(
        gap_subset,
        power_balance_duals=power_balance_duals,
        savefolder=dual_save_table_folder,
    )
    plot_gap_lmp_distribution(gap_subset, "gap_wind_thermal", savefolder=dual_folder)


    freq_table = make_lmp_frequency_table(
        power_balance_duals, bin_width=1.0, cap=70, savefolder=dual_save_table_folder
    )
    sorted_freq_table = freq_table.sort_values(by="count", ascending=False)
    plot_top_10_dual_intervals(
        power_balance_duals, sorted_freq_table, n_bins=10, savefolder=dual_folder
    )
    create_top_10_dual_intervals_tables(
        power_balance_duals, freq_table, n_bins=10, savefolder=tables_folder
    )

    summary_by_carrier = summarise_extension_status_by_carrier(generators)
    summary_table = summarize_dual_nonzero_counts(
        dual_variables, dual_save_table_folder
    )
    summary = summarize_duals_by_year_scenario(
        dual_variables, savefolder=dual_save_table_folder
    )
    nz = list_nonzero_gen_extension_duals(
        gen_extension_duals, savefolder=dual_save_table_folder
    )

    if not show_plots:
        # Restore the original show function
        plt.show = original_show
        print(30 * "-")
        print(
            f"Post Optimization Analysis completed for model_id: {model_config['model_id']}, model: {model_config["model_name"]}, run_id: {model_config["run_id"]}. \n Results saved in {RESULTS_FOLDER}"
        )
        print(30 * "-")


if __name__ == "__main__":
    data_folder_name = "elec_s_37_all"
    run_analytics_on_input_data(data_folder_name)

    # model_config_test_path = r"C:\Users\tinus\OneDrive\Dokumenter\0 Master\code\master_project\runs\single_runs\GTSEP_v0-test small-Feb11_Tue_h10\model_info\config.yaml"
    # import yaml

    # model_config = yaml.safe_load(open(model_config_test_path))
    # # print(model_config)
    # analyze_run(model_config)
