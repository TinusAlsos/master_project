""" This module contains functions for running automated analytics on the data in different stages of the pipeline. """

import os

import numpy as np
import pandas as pd
import src.plotting as plotting
import src.utils as utils
import matplotlib.pyplot as plt
import gurobipy as gp

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, "processed")


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
        batteries["new_power_capacity"] = batteries["P_discharge_max"] * battery_build.loc[batteries.index.values, "value"]
        batteries["new_energy_capacity"] = batteries["new_power_capacity"] * batteries["hour_capacity"]
    else:
        batteries["new_power_capacity"] = battery_build.loc[batteries.index.values, "value"]
        batteries["new_energy_capacity"] = batteries["new_power_capacity"] * batteries["hour_capacity"]
        battery_build["value"] = (battery_build["value"] != 0).astype(int)



def check_line_errors(branches: pd.DataFrame, power_flow: pd.DataFrame) -> None:
    # Make sure power flow is + if branch doesn't exist
    for branch in branches[branches["exists"] == 0].index:
        if power_flow[branch].abs().sum() != 0:
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
        batteries.loc[
            batteries_build[batteries_build["value"] == 1].index.values,
            "capital_cost",
        ]
        * batteries.loc[
            batteries_build[batteries_build["value"] == 1].index.values, "SOC_max"
        ]
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


if __name__ == "__main__":
    data_folder_name = "elec_s_37_all"
    run_analytics_on_input_data(data_folder_name)

    # model_config_test_path = r"C:\Users\tinus\OneDrive\Dokumenter\0 Master\code\master_project\runs\single_runs\GTSEP_v0-test small-Feb11_Tue_h10\model_info\config.yaml"
    # import yaml

    # model_config = yaml.safe_load(open(model_config_test_path))
    # # print(model_config)
    # analyze_run(model_config)
