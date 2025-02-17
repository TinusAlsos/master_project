"""
preprocessing.py

Reads in data from the PyPSA2023 data and extracts and preprocesses the data for the optimization model to use. 

"""

import math
from pypsa import Network
import os
import pandas as pd
import yaml
import argparse
from .utils import (
    calculate_crf,
    load_battery_config_by_name,
    load_config,
    load_preprocessing_config_by_name,
)


DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
NETWORKS_FOLDER = os.path.join(DATA_FOLDER, "raw", "networks")
SAVE_FOLDER = os.path.join(DATA_FOLDER, "processed")


def read_pypsa_network(network_name: str) -> Network:
    """
    Reads in the PyPSA network from the data folder.

    Args:
    network_name (str): The name of the PyPSA network to read in. Example: "elec_s_37", "elec_s_128"

    Returns:
    network (pypsa.Network): The PyPSA network object.
    """
    networks_path = os.path.join(NETWORKS_FOLDER, f"{network_name}.nc")
    if not os.path.exists(networks_path):
        raise FileNotFoundError(f"Network file not found: {networks_path}")
    network = Network(networks_path)
    return network


def get_components_data(network: Network, components: list[str]) -> pd.DataFrame:
    """
    Extracts the data for a specific component from the PyPSA network.

    Args:
    network (pypsa.Network): The PyPSA network object.
    component_names (list[str]): The names of the components to extract the data for. Example: "buses", "carriers"

    Returns:
    component_data (pd.DataFrame): The data for the component.
    """
    data = {component: getattr(network, component) for component in components}
    return data


def filter_data_by_countries(
    data: dict[pd.DataFrame], country_codes: list[str]
) -> dict[pd.DataFrame]:
    """Filters each DataFrame in the data dictionary to only include entries that match the specified country codes.

    Parameters:
        data (dict): A dictionary where each value is a DataFrame, and each key is the name of the DataFrame.
        country_codes (list): A list of country codes to filter by (e.g., ["ES", "PT"] for Spain and Portugal).

    Returns:
        dict: A dictionary of filtered DataFrames.
    """

    if not country_codes:
        return data
    filtered_data = {}
    for name, df in data.items():
        if "country" in df.columns:
            filtered_df = df[df["country"].isin(country_codes)]
        else:
            filtered_df = df
        filtered_data[name] = filtered_df
    return filtered_data


def filter_data_by_buses(
    data: dict[pd.DataFrame], valid_buses: list[str]
) -> dict[pd.DataFrame]:
    """Filters each DataFrame in the data dictionary to only include entries with specified valid buses.

    Parameters:
        data (dict): A dictionary where each value is a DataFrame, and each key is the name of the DataFrame.
        valid_buses (list): A list of valid bus IDs to filter by.

    Returns:
        dict: A dictionary of filtered DataFrames.
    """
    filtered_data = {}
    for name, df in data.items():
        if name == "buses":
            filtered_df = df.loc[valid_buses]
        elif "bus" in df.columns:
            print(f"Filtering {name} by 'bus' column")
            print(f"Length of {name} before filtering: {len(df)}")
            filtered_df = df[df["bus"].isin(valid_buses)]
            print(f"Length of {name} after filtering: {len(filtered_df)}")
        elif "bus0" in df.columns and "bus1" in df.columns:
            print(f"Filtering {name} by 'bus0' and 'bus1' columns")
            print(f"Length of {name} before filtering: {len(df)}")
            filtered_df = df[
                df["bus0"].isin(valid_buses) & df["bus1"].isin(valid_buses)
            ]
            print(f"Length of {name} after filtering: {len(filtered_df)}")
        else:
            filtered_df = df
        filtered_data[name] = filtered_df
    return filtered_data


def get_valid_buses_from_query(data: dict[pd.DataFrame], query: str = "") -> list[str]:
    """Returns a list of valid bus IDs based on a query string.

    Parameters:
        data (dict): A dictionary where each value is a DataFrame, and each key is the name of the DataFrame.
        query (str): A query string to filter the buses DataFrame.

    Returns:
        list: A list of valid bus IDs.
    """
    buses = data["buses"]
    if query:
        buses = buses.query(query)
    valid_buses = buses.index.tolist()
    return valid_buses


def filter_generators_by_carrier(
    data: dict[pd.DataFrame], carriers: list[str] = None
) -> None:
    """Filters the generators DataFrame to only include entries with specified carriers.

    Parameters:
        data (dict): A dictionary where each value is a DataFrame, and each key is the name of the DataFrame.
        carriers (list): A list of carrier names to filter by.

    Returns:
        None
    """
    if carriers is None:
        return
    generators = data["generators"]
    print(f"Filtering generators by carriers: {carriers}")
    print(f"Length of generators before filtering: {len(generators)}")
    filtered_generators = generators[generators["carrier"].isin(carriers)]
    print(f"Carriers dropped: {set(generators['carrier']) - set(carriers)}")
    print(f"Length of generators after filtering: {len(filtered_generators)}")
    data["generators"] = filtered_generators


def get_time_dependent_generators(
    data: dict[pd.DataFrame], network: Network
) -> list[str]:
    """Returns a DataFrame of time-dependent generators.

    Parameters:
        data (dict): A dictionary where each value is a DataFrame, and each key is the name of the DataFrame.
        network (pypsa.Network): The PyPSA network object.

    Returns:
        list[str]: A list of time-dependent generators.
    """
    generators = data["generators"]
    time_dependent_generators = [
        generator
        for generator in generators.index
        if generator in network.generators_t.p_max_pu.columns
    ]
    return time_dependent_generators


def get_capacity_factors(data: dict[pd.DataFrame], network: Network) -> pd.DataFrame:
    """Returns a DataFrame of capacity factors for time-dependent generators.

    Parameters:
        data (dict): A dictionary where each value is a DataFrame, and each key is the name of the DataFrame.
        network (pypsa.Network): The PyPSA network object.

    Returns:
        pd.DataFrame: A DataFrame of capacity factors for all generators.
    """
    time_dependent_generators = get_time_dependent_generators(data, network)
    static_generators = list(
        set(data["generators"].index) - set(time_dependent_generators)
    )
    # Set static generators capacity factors to 1.0 at all timesteps
    ones_df = pd.DataFrame(1.0, index=network.snapshots, columns=static_generators)
    capacity_factors = network.generators_t.p_max_pu[time_dependent_generators]
    capacity_factors = pd.concat([capacity_factors, ones_df], axis=1)
    return capacity_factors


def get_hourly_demand(data: dict[pd.DataFrame], network: Network) -> pd.DataFrame:
    """Returns a DataFrame of hourly demand for each bus.

    Parameters:
        data (dict): A dictionary where each value is a DataFrame, and each key is the name of the DataFrame.
        network (pypsa.Network): The PyPSA network object.

    Returns:
        pd.DataFrame: A DataFrame of hourly demand for each bus.
    """
    relevant_nodes = data["buses"].index
    demand = network.loads_t.p_set.loc[:, relevant_nodes]
    return demand


def filter_data_by_columns(
    data: dict[pd.DataFrame],
    carrier_columns: list[str],
    bus_columns: list[str],
    generator_columns: list[str],
    line_columns: list[str],
) -> None:
    """Filters the data DataFrames to only include specified columns.

    Parameters:
        data (dict): A dictionary where each value is a DataFrame, and each key is the name of the DataFrame.
        carrier_columns (list): A list of carrier columns to keep.
        bus_columns (list): A list of bus columns to keep.
        generator_columns (list): A list of generator columns to keep.
        line_columns (list): A list of line columns to keep.

    Returns:
        None
    """
    for name, df in data.items():
        if name == "carriers":
            data[name] = df[carrier_columns]
        elif name == "buses":
            data[name] = df[bus_columns]
        elif name == "generators":
            data[name] = df[generator_columns]
        elif name == "lines":
            data[name] = df[line_columns]
        else:
            continue
    # Merge carriers into generators
    data["generators"] = pd.merge(
        data["generators"], data["carriers"], left_on="carrier", right_index=True
    )
    # Delete carriers DataFrame
    del data["carriers"]


def generate_batteries(
    data: dict[pd.DataFrame], battery_template: dict
) -> pd.DataFrame:
    # Calculate the Capital Recovery Factor (CRF) for the battery
    crf = calculate_crf(battery_template["lifetime"], battery_template["discount_rate"])
    annualized_cost = crf * battery_template["investment_cost_permwh"]
    leakage_rate_per_hour = 1 - math.pow(1 - battery_template["daily_leakage"], 1 / 24)
    cdrate = battery_template["cdrate"]
    P_discharge_max = battery_template["P_discharge_max"]
    P_charge_max = P_discharge_max * cdrate
    battery = {
        "MC": battery_template["marginal_cost"],
        "capital_cost": annualized_cost,
        "hour_capacity": battery_template["hour_capacity"],
        "cdrate": cdrate,
        "P_discharge_max": P_discharge_max,
        "P_discharge_min": battery_template["P_discharge_min"],
        "P_charge_max": P_charge_max,
        "P_charge_min": battery_template["P_charge_min"],
        "SOC_max": battery_template["SOC_max"],
        "SOC_min": battery_template["SOC_min"],
        "delta": leakage_rate_per_hour,
        "eta_charge": battery_template["efficiency_charge"],
        "eta_discharge": battery_template["efficiency_discharge"],
    }

    batteries = pd.DataFrame(
        [battery] * len(data["buses"]), index=data["buses"].index + " bat"
    ).rename_axis("battery")
    # Add the bus (node) as the first column
    batteries["node"] = [
        " ".join(name.split(" ")[:-1]) for name in batteries.index.values
    ]  # Extract the bus name
    batteries = batteries[
        ["node"] + [col for col in batteries.columns if col != "node"]
    ]  # Rearrange columns
    return batteries


def config_name_to_path(config_name: str) -> str:
    """Converts a configuration name to a configuration file path. Also edits the config name if necessary and checks if the file exists.

    Args:
        config_name (str): The name of the configuration file.

    Returns:
        str: The path to the configuration file.
    """
    if isinstance(config_name, str):
        if os.path.exists(config_name):
            return config_name
        else:
            if "config" not in config_name:
                config_name = "config_" + config_name
            return os.path.join(DATA_FOLDER, "configs", f"{config_name}.yaml")


def run_preprocessing(config: dict | str = "") -> None:
    """Runs the preprocessing pipeline using the specified configuration.

    Args:
        config (dict or str): The configuration dictionary, the name of the configuration file (i.e. example_config), or the path to the configuration file.
    """

    if not config:
        config = load_preprocessing_config_by_name()
    elif isinstance(config, str):
        # Check if the config is a file path
        if os.path.exists(config):
            config = load_config(config)
        else:
            # Assume the config is a name
            if "config" not in config:
                config = "config_" + config
            config = load_preprocessing_config_by_name(config)

    # Filtering
    network = read_pypsa_network(config["network_name"])
    data = get_components_data(network, config["components"])
    data = filter_data_by_countries(data, config["countries"])
    # Update p_nom values for "offwind-ac" generators
    data["generators"].loc[data["generators"]["carrier"] == "offwind-ac", "p_nom"] = (
        data["generators"].loc[
            data["generators"]["carrier"] == "offwind-ac", "p_nom_max"
        ]
    )
    valid_buses = get_valid_buses_from_query(data, config["query"])
    data = filter_data_by_buses(data, valid_buses)
    filter_generators_by_carrier(data, config["carriers"])
    data["generators"] = data["generators"].query("active == True")
    filter_data_by_columns(
        data,
        config["carrier_columns"],
        config["bus_columns"],
        config["generator_columns"],
        config["line_columns"],
    )

    identifier = (
        config["network_name"] + "_" + "_".join(config["countries"])
        if config["countries"]
        else config["network_name"] + "_all"
    )
    if config["id"]:
        identifier  += "_" + config["id"]
    output_folder = os.path.join(SAVE_FOLDER, identifier)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data["lines"].rename(columns={"s_nom": "p_max"}, inplace=True)
    max_loss = config["branches_max_loss"]
    data["lines"]["loss_factor"] = (
        max_loss * data["lines"]["length"] / data["lines"]["length"].max()
    )
    for name, df in data.items():
        if name in config["components"] or not config["components"]:
            df = df.rename_axis(df.index.name.lower())
            if name == "buses":
                name = "nodes"
            if name == "lines":
                name = "branches"
            df.to_csv(os.path.join(output_folder, f"{name}.csv"))

    # Calculate generators costs
    generator_costs_dataframe = pd.DataFrame(
        {
            "capital_cost [€/MW/year]": data["generators"]
            .groupby("carrier")["capital_cost"]
            .mean(),
        }
    )
    generator_costs_dataframe.to_csv(os.path.join(output_folder, "generator_costs.csv"))
    capacity_factors = get_capacity_factors(data, network)
    hourly_demand = get_hourly_demand(data, network)
    capacity_factors.to_csv(os.path.join(output_folder, "capacity_factors.csv"))
    hourly_demand.to_csv(os.path.join(output_folder, "hourly_demand.csv"))

    # Batteries
    # Save the config file as a record of the processing steps
    output_config_path = os.path.join(output_folder, "config.yaml")
    with open(output_config_path, "w") as f:
        yaml.dump(config, f)

    battery_template = load_battery_config_by_name(config["battery_config"])
    batteries = generate_batteries(data, battery_template)
    batteries.to_csv(os.path.join(output_folder, "batteries.csv"))
    print(f"Preprocessing completed. Data saved to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the preprocessing pipeline with optional name as input."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="The name of the configuration file to use for preprocessing.",
    )
    args = parser.parse_args()
    run_preprocessing(args.name)
