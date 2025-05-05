"""This script is just to test if I can generate copies of configs with custom changes"""

from src.utils import copy_and_modify_config


if __name__ == "__main__":

    base_config_name = "config_37_sv1_4_weeks"

    value_to_change = "scenario_file"

    new_values = ["EVPI_" + str(i) for i in range(1, 7)]
    new_values.append("mean")
    print(new_values)

    value_change_dictionaries = {name: {value_to_change: name} for name in new_values}
    print(value_change_dictionaries)

    for name, value_change_dict in value_change_dictionaries.items():
        copy_and_modify_config(
            config_name_or_path=base_config_name,
            overwrite_dict=value_change_dict,
            suffix="_" + name,
        )
