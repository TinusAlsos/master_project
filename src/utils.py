"""
This module contains utility functions that are used in the project."""

import copy
import json
import os
from typing import Any
import yaml
import pandas as pd
import datetime
import uuid


CONFIG_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
MODELS_CONFIG_FOLDER = os.path.join(CONFIG_FOLDER, "models")
PREPROCESSING_CONFIG_FOLDER = os.path.join(CONFIG_FOLDER, "preprocessing")
BATTERY_CONFIG_FOLDER = os.path.join(PREPROCESSING_CONFIG_FOLDER, "battery")
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def deep_update(source: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively update the `source` dictionary with values from `overrides`.
    For keys that exist in both dictionaries:
      - if the value is a dictionary, update it recursively.
      - otherwise, override the value from `source` with that from `overrides`.
    """
    for key, value in overrides.items():
        if key in source and isinstance(source[key], dict) and isinstance(value, dict):
            deep_update(source[key], value)
        else:
            source[key] = value
    return source


def load_config(custom_config_path: str = "") -> dict:
    """
    Loads a custom configuration file and merges it with the base configuration.
    Missing entries in the custom file will default to the base configuration.
    """
    if not custom_config_path:
        print("No custom configuration file provided. Using base configuration.")
        base_config_path = os.path.join(CONFIG_FOLDER, "base_config.yaml")
    # Determine the folder containing the custom config file
    folder = os.path.dirname(custom_config_path)

    if not os.path.exists(custom_config_path):
        raise FileNotFoundError(
            f"Custom configuration file not found: {custom_config_path}"
        )

    # Construct the path to the base configuration file
    base_config_path = os.path.join(folder, "base_config.yaml")
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(
            f"Base configuration file not found: {base_config_path}"
        )

    # Load the base configuration
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Load the custom configuration
    with open(custom_config_path, "r") as f:
        custom_config = yaml.safe_load(f)

    # Merge custom config into the base config
    merged_config = deep_update(base_config, custom_config)
    print(f"Configuration loaded from {custom_config_path}")
    return merged_config


def load_model_config(path_or_name: str = "") -> dict:
    """
    Loads a configuration file by name and returns it as a dictionary.
    """
    print(f"Path_or_name: {path_or_name}")
    if not path_or_name:
        path_or_name = os.path.join(MODELS_CONFIG_FOLDER, "base_config.yaml")

    # Check if its a path
    if os.path.exists(path_or_name):
        config_path = path_or_name
    else:
        if not path_or_name.endswith(".yaml"):
            path_or_name += ".yaml"
        if not path_or_name.startswith("config"):
            path_or_name = f"config_{path_or_name}"
        config_path = os.path.join(MODELS_CONFIG_FOLDER, path_or_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model configuration file not found: {config_path}")
    return load_config(config_path)


def load_config_by_name(config_name: str = "") -> dict:
    """
    Loads a configuration file by name and returns it as a dictionary.
    """
    if not config_name:
        load_config()

    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    # Construct the path to the configuration file
    config_path = os.path.join(CONFIG_FOLDER, config_name)

    return load_config(config_path)


def load_preprocessing_config_by_name(config_name: str = "") -> dict:
    """
    Loads a preprocessing configuration file by name and returns it as a dictionary.
    """

    if not config_name:
        print("No preprocessing configuration file provided. Using base configuration.")
        config_name = "base_config"
    # Construct the path to the configuration file
    config_path = os.path.join(PREPROCESSING_CONFIG_FOLDER, f"{config_name}.yaml")

    return load_config(config_path)


def calculate_crf(lifetime, discount_rate):
    """Calculate the Capital Recovery Factor (CRF), which represents the annual payment
    required to repay a loan over a specified lifetime at a given discount rate.

    Args:
        lifetime (int): The number of periods (years) over which the loan is repaid. Must be > 0.
        discount_rate (float): The discount (interest) rate per period, expressed as a decimal (e.g., 0.05 for 5%).

    Returns:
        float: The Capital Recovery Factor, representing the annual repayment factor.
    """
    return (discount_rate * (1 + discount_rate) ** lifetime) / (
        (1 + discount_rate) ** lifetime - 1
    )


def load_battery_config_by_name(path_or_name: str = "") -> dict:
    """
    Loads a battery configuration file by name and returns it as a dictionary.
    """
    if not path_or_name:
        print("No battery configuration file provided. Using base configuration.")
        path_or_name = os.path.join(BATTERY_CONFIG_FOLDER, "base_config.yaml")

    if os.path.exists(path_or_name):
        config_path = path_or_name
    else:
        if not path_or_name.endswith(".yaml"):
            path_or_name += ".yaml"
        if not path_or_name.startswith("config"):
            path_or_name = f"config_{path_or_name}"
        config_path = os.path.join(BATTERY_CONFIG_FOLDER, path_or_name)

    return load_config(config_path)


def copy_and_modify_config(
    config_name_or_path, overwrite_dict, new_filename=None, suffix="_modified"
):
    # Handle full path or construct from name
    if os.path.exists(config_name_or_path):
        config_path = config_name_or_path
    else:
        if not config_name_or_path.startswith("config"):
            config_name_or_path = f"config_{config_name_or_path}"
        if not config_name_or_path.endswith(".yaml"):
            config_name_or_path += ".yaml"
        config_path = os.path.join(MODELS_CONFIG_FOLDER, config_name_or_path)

    assert os.path.exists(config_path), f"Config file not found at: {config_path}"

    # Load the original config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply overwrites
    new_config = copy.deepcopy(config)
    for key, value in overwrite_dict.items():
        new_config[key] = value

    # Determine output path
    config_dir = os.path.dirname(config_path)
    config_base = os.path.basename(config_path)
    config_stem, config_ext = os.path.splitext(config_base)

    if new_filename:
        if not new_filename.endswith(".yaml"):
            new_filename += ".yaml"
        if not new_filename.startswith("config"):
            new_filename = f"config_{new_filename}"
        new_path = os.path.join(config_dir, new_filename)
    else:
        new_filename = f"{config_stem}{suffix}{config_ext}"
        new_path = os.path.join(config_dir, new_filename)

    # Save new config
    with open(new_path, "w") as f:
        yaml.dump(new_config, f, sort_keys=False)

    print(f"New config saved to: {new_path}")
    return new_path


def delete_config_file(path_or_name: str = "") -> None:
    """
    Deletes a configuration file by name or path.
    """
    if not path_or_name:
        return
    # Check if its a path
    if os.path.exists(path_or_name):
        config_path = path_or_name
    else:
        if not path_or_name.endswith(".yaml"):
            path_or_name += ".yaml"
        if not path_or_name.startswith("config"):
            path_or_name = f"config_{path_or_name}"
        config_path = os.path.join(MODELS_CONFIG_FOLDER, path_or_name)
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Model configuration file not found: {config_path}, created from {path_or_name}"
            )
    # Delete the file
    os.remove(config_path)
    print(f"Deleted config file: {config_path}")


def load_csv_files_from_folder(data_folder_path: str) -> dict[str, pd.DataFrame]:
    if not os.path.exists(data_folder_path):
        raise FileNotFoundError(
            f"{data_folder_path} not found (should be the path to a folder containing processed data in csv files)"
        )
    data = {}
    for file in os.listdir(data_folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(data_folder_path, file)
            file_name = file.split(".")[0]
            data[file_name] = pd.read_csv(file_path, index_col=0)
            if file_name in ["hourly_demand", "capacity_factors"]:
                data[file_name].index = pd.to_datetime(data[file_name].index)
    return data


def load_csv_files_from_folder_multi(data_folder_path: str) -> dict[str, pd.DataFrame]:
    if not os.path.exists(data_folder_path):
        raise FileNotFoundError(
            f"{data_folder_path} not found (should be the path to a folder containing processed data in csv files)"
        )
    data = {}
    for file in os.listdir(data_folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(data_folder_path, file)
            file_name = file.split(".")[0]
            if file_name in [
                "generator_capacity",
                "battery_capacity",
                "branch_capacity",
            ]:
                data[file_name] = pd.read_csv(file_path, index_col=0)
            else:
                data[file_name] = pd.read_csv(file_path, index_col=["year", "hour"])

    return data


def load_csv_files_from_folder_with_scenarios(
    data_folder_path: str,
) -> dict[str, pd.DataFrame]:
    """For model stochastic v1"""
    data = {}
    for file in os.listdir(data_folder_path):
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(data_folder_path, file)
        file_name = file.split(".")[0]

        df = pd.read_csv(file_path)
        if "value" in df.columns:
            value_name = "value"
        elif "dual_value" in df.columns:
            value_name = "dual_value"
        else:
            raise ValueError(
                "No value or dual_value column found in the dataframe. This is unexpected."
            )
        if "scenario" in df.columns:
            if file_name == "emissions_duals":
                base_index_cols = ["scenario", "year"]
            else:
                base_index_cols = ["scenario", "year", "week", "hour"]
        else:
            base_index_cols = ["year"]
        unique_index_cols = df.columns.difference(base_index_cols).tolist()
        unique_index_cols.remove(value_name)
        index_cols = unique_index_cols + base_index_cols
        df.set_index(index_cols, inplace=True)
        data[file_name] = df
    return data


def load_csv_files_from_folder_multi_weeks(
    data_folder_path: str,
) -> dict[str, pd.DataFrame]:
    if not os.path.exists(data_folder_path):
        raise FileNotFoundError(
            f"{data_folder_path} not found (should be the path to a folder containing processed data in csv files)"
        )

    data = {}

    for file in os.listdir(data_folder_path):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(data_folder_path, file)
        file_name = file.split(".")[0]

        df = pd.read_csv(file_path)

        if file_name in [
            "generator_capacity",
            "battery_capacity",
            "branch_capacity",
            "branch_extension_duals",
            "gen_extension_duals",
        ]:
            # Assume first column after 'year' is the ID (generator, battery, line, etc.)
            index_cols = ["year", df.columns[1]]
            df.set_index(index_cols, inplace=True)
        else:
            # Assume time-dependent dataframes have year, week, hour, and some index (node, generator, etc.)
            index_cols = ["year", "week", "hour", df.columns[3]]
            df.set_index(index_cols, inplace=True)

        data[file_name] = df

    return data


def load_multi_year_csv_files_from_folder(
    years: list[int], data_folder_path: str
) -> dict[str, pd.DataFrame]:
    """temporary quick fix for multi-year data loading. I use the same data as for a single year, but I just post-process the dataframes to be in a multi-year format. All data is the same accross all years, so results are relatively meaningless."""
    if not os.path.exists(data_folder_path):
        raise FileNotFoundError(
            f"{data_folder_path} not found (should be the path to a folder containing processed data in csv files)"
        )
    data = {}
    for file in os.listdir(data_folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(data_folder_path, file)
            file_name = file.split(".")[0]
            df = pd.read_csv(file_path, index_col=0)
            if file_name in ["hourly_demand", "capacity_factors"]:
                df.index = range(len(df))
                df.index.name = "hour"
            new_dfs = []
            demand_multiplier = 1
            if file_name == "nodes":
                data[file_name] = df
                continue
            for year in years:
                temp_df = df.copy()
                if file_name == "hourly_demand":
                    temp_df = temp_df * demand_multiplier
                    demand_multiplier += 1
                temp_df.index = pd.MultiIndex.from_product(
                    [[year], temp_df.index], names=["year", temp_df.index.name]
                )
                new_dfs.append(temp_df)
            data[file_name] = pd.concat(new_dfs)
    return data


def load_multi_year_csv_files_with_week_from_folder(
    years: list[int], data_folder_path: str, yearly_discount: int = 10
) -> dict[str, pd.DataFrame]:
    """temporary quick fix for multi-year data loading. I use the same data as for a single year, but I just post-process the dataframes to be in a multi-year format. All data is the same accross all years, so results are relatively meaningless."""
    if not os.path.exists(data_folder_path):
        raise FileNotFoundError(
            f"{data_folder_path} not found (should be the path to a folder containing processed data in csv files)"
        )
    data = {}
    for file in os.listdir(data_folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(data_folder_path, file)
            file_name = file.split(".")[0]
            if file_name in ["hourly_demand", "capacity_factors"]:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df["week"] = df.index.isocalendar().week
                df["month"] = df.index.month
                df["hour"] = df.index.hour
                hours = df["hour"].unique()
            else:
                df = pd.read_csv(file_path, index_col=0)
            new_dfs = []
            if file_name == "nodes":
                data[file_name] = df
                continue
            for year in years:
                temp_df = df.copy()
                if len(temp_df) == 0:
                    continue
                if file_name in ["hourly_demand", "capacity_factors"]:
                    iso_info = temp_df.index.to_series().dt.isocalendar()
                    temp_df["hour_in_week"] = temp_df.groupby(
                        [temp_df.index.year, iso_info.week]
                    ).cumcount()
                    temp_df.index = pd.MultiIndex.from_arrays(
                        [
                            df.index.year * 0 + year,
                            df.index.isocalendar().week,
                            temp_df["hour_in_week"],
                        ],
                        names=["year", "week", "hour"],
                    )
                    temp_df.drop(columns="hour_in_week", inplace=True)
                    temp_df.drop(columns=["week", "month", "hour"], inplace=True)
                else:
                    temp_df.index = pd.MultiIndex.from_product(
                        [[year], temp_df.index], names=["year", temp_df.index.name]
                    )
                new_dfs.append(temp_df)
            if file_name == "batteries" and len(temp_df) == 0:
                data[file_name] = temp_df
            else:
                data[file_name] = pd.concat(new_dfs)

    for component in ["generators", "branches", "batteries"]:
        if len(data[component]) == 0:
            continue
        for i, year in enumerate(years[1:], start=1):
            discount_amount = yearly_discount * i
            # Make an explicit copy of that year's slice
            sub = data[component].xs(year, level="year").copy()

            # Apply the discount
            sub["capital_cost"] -= discount_amount

            # Write it back into the original DataFrame
            # Get the subset for the given year
            data_comp_year = data[component].loc[year]

            # Update the 'capital_cost' in the selected year using sub (matched by generator index)
            # This works if all generators in sub are present in data[component] for that year.
            data_comp_year.update(sub["capital_cost"])

    hourly_demand = load_hourly_demand(data_folder_path)
    data["hourly_demand"] = hourly_demand
    return data


def load_hourly_demand(data_folder_path: str) -> pd.DataFrame:
    """Load the hourly demand data from the specified folder."""
    if not os.path.exists(data_folder_path):
        raise FileNotFoundError(
            f"{data_folder_path} not found (should be the path to a folder containing processed data in csv files)"
        )
    file_path = os.path.join(data_folder_path, "hourly_demand.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"hourly_demand.csv not found in {data_folder_path}")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df["week"] = df.index.isocalendar().week
    df["month"] = df.index.month
    df["hour"] = df.index.hour
    hours = df["hour"].unique()
    iso_info = df.index.to_series().dt.isocalendar()
    df["hour_in_week"] = df.groupby([iso_info.week]).cumcount()
    df.index = pd.MultiIndex.from_arrays(
        [
            df.index.isocalendar().week,
            df["hour_in_week"],
        ],
        names=["week", "hour"],
    )
    df.drop(columns="hour_in_week", inplace=True)
    df.drop(columns=["week", "month", "hour"], inplace=True)

    return df


def load_multi_year_csv_files_with_week_from_folder_with_demand_scaling(
    years: list[int], weeks, data_folder_path: str
) -> dict[str, pd.DataFrame]:
    """temporary quick fix for multi-year data loading. I use the same data as for a single year, but I just post-process the dataframes to be in a multi-year format. All data is the same accross all years, so results are relatively meaningless."""
    if not os.path.exists(data_folder_path):
        raise FileNotFoundError(
            f"{data_folder_path} not found (should be the path to a folder containing processed data in csv files)"
        )
    data = {}
    for file in os.listdir(data_folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(data_folder_path, file)
            file_name = file.split(".")[0]
            if file_name in ["hourly_demand", "capacity_factors"]:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df["week"] = df.index.isocalendar().week
                df["month"] = df.index.month
                df["hour"] = df.index.hour
                hours = df["hour"].unique()
            else:
                df = pd.read_csv(file_path, index_col=0)
            new_dfs = []
            demand_multiplier = 1
            if file_name == "nodes":
                data[file_name] = df
                continue
            for year in years:
                temp_df = df.copy()

                if file_name == "hourly_demand":
                    temp_df = temp_df * demand_multiplier
                    demand_multiplier += 1
                if file_name in ["hourly_demand", "capacity_factors"]:
                    iso_info = temp_df.index.to_series().dt.isocalendar()
                    temp_df["hour_in_week"] = temp_df.groupby(
                        [temp_df.index.year, iso_info.week]
                    ).cumcount()
                    temp_df.index = pd.MultiIndex.from_arrays(
                        [
                            df.index.year * 0 + year,
                            df.index.isocalendar().week,
                            temp_df["hour_in_week"],
                        ],
                        names=["year", "week", "hour"],
                    )
                    temp_df.drop(columns="hour_in_week", inplace=True)
                    temp_df.drop(columns=["week", "month", "hour"], inplace=True)
                else:
                    temp_df.index = pd.MultiIndex.from_product(
                        [[year], temp_df.index], names=["year", temp_df.index.name]
                    )
                new_dfs.append(temp_df)
            data[file_name] = pd.concat(new_dfs)
    return data


def read_jsons_from_dir(input_dir: str):
    """
    Reads all JSON files from a directory into a dictionary.

    Args:
        input_dir (str): Path to the directory containing JSON files.

    Returns:
        dict: A dictionary where keys are filenames (without .json)
              and values are the contents of the JSON files.
    """
    json_dict = {}

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                key = os.path.splitext(filename)[0]
                json_dict[key] = json.load(f)

    return json_dict


def subset_by_weeks(df, year, weeks):
    """
    Returns a subset of the dataframe for specific ISO weeks of a given year.

    Args:
        df (pd.DataFrame): DataFrame with a DateTimeIndex.
        year (int): The year to filter by.
        weeks (list or set): A collection of ISO week numbers to include.

    Returns:
        pd.DataFrame: DataFrame subset for the specified weeks.
    """
    # Create a mask that filters by year and if the week number is in the provided weeks list
    week_mask = (df.index.year == year) & (
        df.index.to_series().dt.isocalendar().week.isin(weeks)
    )
    return df[week_mask]


def subset_by_months(df, year, months):
    """
    Returns a subset of the dataframe for specific months of a given year.

    Args:
        df (pd.DataFrame): DataFrame with a DateTimeIndex.
        year (int): The year to filter by.
        months (list or set): A collection of month numbers (1-12) to include.

    Returns:
        pd.DataFrame: DataFrame subset for the specified months.
    """
    # Create a mask that filters by year and if the month is in the provided months list
    month_mask = (df.index.year == year) & (df.index.month.isin(months))
    return df[month_mask]


def load_scenario_multiplier(scenario_file_name: str = "") -> pd.DataFrame:
    """
    Loads the scenario multiplier file given its scenario_file_name
    and returns it as a DataFrame.
    """
    base_name = "base_scenarios"
    scenario_multipliers_folder = os.path.join(DATA_FOLDER, "scenario_multipliers")
    if not scenario_file_name:
        scenario_file_name = base_name
    path = os.path.join(scenario_multipliers_folder, f"{scenario_file_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scenario multiplier file not found: {path}")
    df = pd.read_csv(path, index_col=0)
    return df


def generate_unique_filename_id() -> str:
    """
    Generates a unique, filesystem-safe identifier for filenames.
    Format: YYYYMMDD_HHMMSS_UUID4
    Example: 20240512_145803_ab12cd34
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:8]  # Shorten UUID to 8 chars
    return f"{timestamp}_{unique_suffix}"


if __name__ == "__main__":
    df = load_scenario_multiplier()
