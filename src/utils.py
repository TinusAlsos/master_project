"""
This module contains utility functions that are used in the project."""

import os
from typing import Any
import yaml
import pandas as pd


CONFIG_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
MODELS_CONFIG_FOLDER = os.path.join(CONFIG_FOLDER, "models")
PREPROCESSING_CONFIG_FOLDER = os.path.join(CONFIG_FOLDER, "preprocessing")
BATTERY_CONFIG_FOLDER = os.path.join(PREPROCESSING_CONFIG_FOLDER, "battery")


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


def load_battery_config_by_name(config_name: str = "") -> dict:
    """
    Loads a battery configuration file by name and returns it as a dictionary.
    """

    if not config_name:
        print("No battery configuration file provided. Using base configuration.")
        config_name = "base_config"
    # Construct the path to the configuration file
    config_path = os.path.join(BATTERY_CONFIG_FOLDER, f"{config_name}.yaml")

    return load_config(config_path)


def load_csv_files_from_folder(data_folder_path: str) -> dict[str, pd.DataFrame]:
    if not os.path.exists(data_folder_path):
        raise FileNotFoundError(
            f"{data_folder_path} not found (should be the path to a folder containing processed data in csv files)"
        )
    data = {}
    for file in sorted(os.listdir(data_folder_path)):
        if file.endswith(".csv"):
            file_path = os.path.join(data_folder_path, file)
            file_name = file.split(".")[0]
            data[file_name] = pd.read_csv(file_path, index_col=0)
            if file_name in ["hourly_demand", "capacity_factors"]:
                data[file_name].index = pd.to_datetime(data[file_name].index)
    return data


if __name__ == "__main__":
    print(CONFIG_FOLDER)
