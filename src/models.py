from time import time
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from src.utils import (
    load_csv_files_from_folder,
    load_hourly_demand,
    load_model_config,
    load_multi_year_csv_files_from_folder,
    load_multi_year_csv_files_with_week_from_folder_with_demand_scaling,
    load_multi_year_csv_files_with_week_from_folder,
    load_scenario_multiplier,
)
import os

TIMELIMIT = 60 * 60 * 6  # 6 hours

PROCESSED_DATA_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "processed"
)


def _create_mappings(
    nodes: pd.DataFrame,
    branches: pd.DataFrame,
    generators: pd.DataFrame,
    batteries: pd.DataFrame,
):
    """
    Create mappings that associate nodes with branches, batteries, and generators.

    Parameters:
        nodes (pd.DataFrame): DataFrame containing node information.
            - The index of this DataFrame should contain unique node identifiers.
        branches (pd.DataFrame): DataFrame containing branch information.
            - Must include columns "bus0" and "bus1", where "bus0" indicates the originating node and
              "bus1" the terminating node for each branch.
        generators (pd.DataFrame): DataFrame containing generator information.
            - Must include a column "bus" indicating the node where each generator is located.
        batteries (pd.DataFrame): DataFrame containing battery information.
            - Must include a column "node" indicating the node where each battery is located.

    Returns:
        tuple: A tuple containing four dictionaries in the following order:
            - branches_out_of_node (dict): Mapping from each node identifier to a list of branch identifiers
              that originate from that node.
            - branches_into_node (dict): Mapping from each node identifier to a list of branch identifiers
              that terminate at that node.
            - batteries_at_node (dict): Mapping from each node identifier to a list of battery identifiers
              at that node.
            - generators_at_node (dict): Mapping from each node identifier to a list of generator identifiers
              at that node.
    """
    if "year" in batteries.index.names:
        y0 = branches.index.unique(level="year")[0]
        branches = branches.loc[y0, :]
        generators = generators.loc[y0, :]
        if len(batteries) > 0:
            batteries = batteries.loc[y0, :]

    N = nodes.index.to_list()
    B = branches.index.to_list()
    G = generators.index.to_list()
    S = batteries.index.to_list()

    branches_out_of_node = {n: [] for n in N}
    branches_into_node = {n: [] for n in N}

    for b in B:
        bus0 = branches.loc[b, "bus0"]
        bus1 = branches.loc[b, "bus1"]
        branches_out_of_node[bus0].append(b)
        branches_into_node[bus1].append(b)

    # Initialize mapping for batteries at nodes (S_n)
    batteries_at_node = {n: [] for n in N}
    for s in S:
        node = batteries.loc[s, "node"]
        batteries_at_node[node].append(s)

    # Initialize mapping for generators at nodes (G_n)
    generators_at_node = {n: [] for n in N}
    for g in G:
        node = generators.loc[g, "bus"]
        generators_at_node[node].append(g)

    return (
        branches_out_of_node,
        branches_into_node,
        batteries_at_node,
        generators_at_node,
    )


# Helper function to reshape a variable with time and other indices
def _reshape_variable(data, index_name, column_name):
    """Reshape the data to have time as rows and other index (e.g., generator) as columns."""
    reshaped = data.reset_index().pivot(
        index="snapshot", columns=index_name, values="value"
    )
    reshaped.columns.name = None  # Remove the name of the columns for cleaner output
    return reshaped


# Helper function to reshape a variable with time and other indices
def _reshape_multi(data, index, column_name, value_name):
    """Reshape the data to have time as rows and other index (e.g., generator) as columns."""
    reshaped = data.reset_index().pivot(
        index=index, columns=column_name, values=value_name
    )
    reshaped.columns.name = None  # Remove the name of the columns for cleaner output
    return reshaped


# region GTSEP_v0
def GTSEP_v0(config: dict) -> gp.Model:
    """GTSEP model from the specialization project."""
    # region Model setup and running
    must_have_keys = [
        "data_folder_name",
        "VOLL",
        "CC",
        "CO2_price",
        "E_limit",
        "p_max_new_branch",
        "p_min_new_branch",
        "expansion_factor",
        "MS",
        "model_name",
        "MIPGap",
    ]
    for key in must_have_keys:
        if key not in config:
            raise KeyError(
                f"Required key '{key}' not found in config. \nRequired keys: {must_have_keys}\nConfig keys: {config.keys()}"
            )

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

    # Load data
    data_folder_path = os.path.join(PROCESSED_DATA_FOLDER, data_folder_name)
    input_data = load_csv_files_from_folder(data_folder_path)
    batteries = input_data["batteries"]
    branches = input_data["branches"]
    capacity_factors = input_data["capacity_factors"]
    generators = input_data["generators"]
    generator_costs = input_data["generator_costs"]
    hourly_demand = input_data["hourly_demand"]
    nodes = input_data["nodes"]

    # Data processing
    # Create new branches
    # Add a new column 'exists' to the original branches dataframe and set it to 1
    branches["exists"] = 1
    # Create a copy of the dataframe for the "new" branches
    branches_new = branches.copy()
    # Update the index by appending " new" to the original index
    branches_new.index = branches_new.index.astype(str) + " new"
    # Set the 'exists' column to 0 for the new branches
    branches_new["exists"] = 0
    # Concatenate the original dataframe and the new dataframe
    branches = pd.concat([branches, branches_new])
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
    batteries["exists"] = 0

    # Create sets
    N = nodes.index.to_list()
    G_old = generators[generators["exists"] == 1].index.to_list()
    G_new = generators[generators["exists"] == 0].index.to_list()
    G = generators.index.to_list()
    B_old = branches[branches["exists"] == 1].index.to_list()
    B_new = branches[branches["exists"] == 0].index.to_list()
    B = branches.index.to_list()
    S_new = batteries[batteries["exists"] == 0].index.to_list()
    S_old = batteries[batteries["exists"] == 1].index.to_list()
    S = batteries.index.to_list()
    T = hourly_demand.index.to_list()

    # Create mappings
    (
        branches_out_of_node,
        branches_into_node,
        batteries_at_node,
        generators_at_node,
    ) = _create_mappings(nodes, branches, generators, batteries)

    build_start_time = time()
    # Create model
    model_name = model_name if model_name else "GTSEP_v0"
    model = gp.Model(model_name)

    # Decision variables
    g = model.addVars(G, T, name="g", lb=0)  # Power generation dispatch
    f = model.addVars(B, T, name="f", lb=-GRB.INFINITY, ub=GRB.INFINITY)  # Power flow
    sh = model.addVars(N, T, name="sh", lb=0)  # Load shedding
    c = model.addVars(G, T, name="c", lb=0)  # Curtailment
    g_ch = model.addVars(S, T, name="g_ch", lb=0)  # Battery charging
    g_dis = model.addVars(S, T, name="g_dis", lb=0)  # Battery discharging
    soc = model.addVars(S, T, name="soc", lb=0)  # State of charge
    x = model.addVars(G_new, vtype=GRB.BINARY, name="x")  # Binary for new generators
    y = model.addVars(B_new, vtype=GRB.BINARY, name="y")  # Binary for new branches
    z = model.addVars(S_new, vtype=GRB.BINARY, name="z")  # Binary for new batteries
    p_i_max = model.addVars(
        G_new, name="p_i_max", lb=0
    )  # Max capacity of new generators
    p_b_max = model.addVars(B_new, name="p_b_max", lb=0)  # Max capacity of new branches

    # Objective function: Minimize cost
    objective = (
        gp.quicksum(
            (
                generators.loc[i, "marginal_cost"]
                + generators.loc[i, "co2_emissions"] * CO2_price
            )
            * g[i, t]
            for i in G
            for t in T
        )
        # + gp.quicksum(
        #     batteries.loc[s, "MC"] * g_dis[s, t] * batteries.loc[s, "eta_discharge"]
        #     for s in S
        #     for t in T
        # )
        + gp.quicksum(VOLL * sh[n, t] for n in N for t in T)
        + gp.quicksum(CC * c[i, t] for i in G for t in T)
        + gp.quicksum(generators.loc[i, "capital_cost"] * p_i_max[i] for i in G_new)
        + gp.quicksum(branches.loc[b, "capital_cost"] * p_b_max[b] for b in B_new)
        + gp.quicksum(
            batteries.loc[s, "capital_cost"]
            * batteries.loc[s, "P_discharge_max"]
            * batteries.loc[s, "hour_capacity"]
            * z[s]
            for s in S_new
        )
    )
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
    # 1. Power balance
    for n in N:
        for t in T:
            model.addConstr(
                gp.quicksum(g[i, t] - c[i, t] for i in generators_at_node[n])
                + gp.quicksum(
                    f[b, t] * (1 - branches.loc[b, "loss_factor"])
                    for b in branches_into_node[n]
                )
                - gp.quicksum(f[b, t] for b in branches_out_of_node[n])
                - gp.quicksum(
                    g_ch[s, t] - batteries.loc[s, "eta_discharge"] * g_dis[s, t]
                    for s in batteries_at_node[n]
                )
                + sh[n, t]
                == hourly_demand.loc[t, n]
            )

    # 2a. Load shedding limits
    for n in N:
        for t in T:
            model.addConstr(sh[n, t] <= MS * hourly_demand.loc[t, n])

    # 2b. Curtailment limits
    for i in G:
        for t in T:
            model.addConstr(c[i, t] <= g[i, t])

    # 3a. Generator output limits (old generators)
    for i in G_old:
        p_max = generators.loc[i, "p_nom"]
        for t in T:
            capacity_factor = capacity_factors.loc[t, i]
            model.addConstr(g[i, t] <= p_max * capacity_factor)
            # Lower bound is 0 by default

    # 3b. Generator output limits (new generators)
    for i in G_new:
        for t in T:
            original_generator_id = " ".join(i.split(" ")[:-1])
            capacity_factor = capacity_factors.loc[t, original_generator_id]
            model.addConstr(g[i, t] <= x[i] * p_i_max[i] * capacity_factor)
            # Lower bound is 0 by default

    # 3c. New generator capacity limits
    for i in G_new:
        p_max = generators.loc[i, "p_nom"]
        model.addConstr(p_i_max[i] <= expansion_factor * p_max)

    # 4a. Branch flow limits (old branches)
    for b in B_old:
        for t in T:
            model.addConstr(f[b, t] >= -branches.loc[b, "p_max"])
            model.addConstr(f[b, t] <= branches.loc[b, "p_max"])

    # 4b. Branch flow limits (new branches)
    for b in B_new:
        for t in T:
            model.addConstr(f[b, t] >= -y[b] * p_b_max[b])
            model.addConstr(f[b, t] <= y[b] * p_b_max[b])

    # 4c. New branch capacity limits
    for b in B_new:
        model.addConstr(p_b_max[b] >= y[b] * p_min_new_branch)
        model.addConstr(p_b_max[b] <= y[b] * p_max_new_branch)

    # # 5. Emission restrictions
    # model.addConstr(
    #     gp.quicksum(g[i, t] * generators.loc[i, "co2_emissions"] for i in G for t in T)
    #     <= E_limit
    # )

    # 6a. Battery charging limits, old batteries
    for s in S_old:
        for t in T:
            model.addConstr(g_ch[s, t] >= batteries.loc[s, "P_charge_min"])
            model.addConstr(g_ch[s, t] <= batteries.loc[s, "P_charge_max"])

    # 6b. Battery charging limits, new batteries
    for s in S_new:
        for t in T:
            model.addConstr(g_ch[s, t] >= z[s] * batteries.loc[s, "P_charge_min"])
            model.addConstr(g_ch[s, t] <= z[s] * batteries.loc[s, "P_charge_max"])

    # 7a. Battery discharging limits, old batteries
    for s in S_old:
        for t in T:
            model.addConstr(g_dis[s, t] >= batteries.loc[s, "P_discharge_min"])
            model.addConstr(g_dis[s, t] <= batteries.loc[s, "P_discharge_max"])

    # 7b. Battery discharging limits, new batteries
    for s in S_new:
        for t in T:
            model.addConstr(g_dis[s, t] >= z[s] * batteries.loc[s, "P_discharge_min"])
            model.addConstr(g_dis[s, t] <= z[s] * batteries.loc[s, "P_discharge_max"])

    # 8. State of charge limits
    for s in S:
        for t in T:
            model.addConstr(soc[s, t] >= batteries.loc[s, "SOC_min"])
            model.addConstr(soc[s, t] <= batteries.loc[s, "SOC_max"])

    # 9. Battery state of charge dynamics
    for s in S:
        for t in T[1:]:  # Exclude time t=0
            model.addConstr(
                soc[s, t]
                == soc[s, t - pd.Timedelta("1h")]
                + batteries.loc[s, "eta_charge"] * g_ch[s, t]
                - g_dis[s, t] / batteries.loc[s, "eta_discharge"]
            )

    # 10. Initial state of charge
    for s in S:
        model.addConstr(soc[s, T[0]] == batteries.loc[s, "SOC_min"])

    # Optimize the model
    model.setParam("MIPGap", MIPGap)
    model.setParam("Timelimit", TIMELIMIT)
    model.setParam("BarConvTol", MIPGap)

    build_end_time = time()

    print(f"Model built in {build_end_time - build_start_time} seconds.")
    model_optimize_start_time = time()
    model.optimize()
    model_optimize_end_time = time()
    # endregion

    # region Post-processing and saving results
    save_folder = config.get("save_folder", None)
    decision_variables_folder = os.path.join(save_folder, "decision_variables")
    if not os.path.exists(decision_variables_folder):
        os.makedirs(decision_variables_folder)
    # Save generation
    generation_data = [(t, i, g[i, t].X) for i in G for t in T]
    generation_df = pd.DataFrame(
        generation_data, columns=["snapshot", "generator", "value"]
    )
    generation_reshaped = _reshape_variable(generation_df, "generator", "snapshot")
    generation_reshaped.to_csv(
        os.path.join(decision_variables_folder, "generation.csv")
    )

    # Save power flow
    power_flow_data = [(t, b, f[b, t].X) for b in B for t in T]
    power_flow_df = pd.DataFrame(
        power_flow_data, columns=["snapshot", "branch", "value"]
    )
    power_flow_reshaped = _reshape_variable(power_flow_df, "branch", "snapshot")
    power_flow_reshaped.to_csv(
        os.path.join(decision_variables_folder, "power_flow.csv")
    )

    # Save load shedding
    load_shedding_data = [(t, n, sh[n, t].X) for n in N for t in T]
    load_shedding_df = pd.DataFrame(
        load_shedding_data, columns=["snapshot", "node", "value"]
    )
    load_shedding_reshaped = _reshape_variable(load_shedding_df, "node", "snapshot")
    load_shedding_reshaped.to_csv(
        os.path.join(decision_variables_folder, "load_shedding.csv")
    )

    # Save curtailment
    curtailment_data = [(t, i, c[i, t].X) for i in G for t in T]
    curtailment_df = pd.DataFrame(
        curtailment_data, columns=["snapshot", "generator", "value"]
    )
    curtailment_reshaped = _reshape_variable(curtailment_df, "generator", "snapshot")
    curtailment_reshaped.to_csv(
        os.path.join(decision_variables_folder, "curtailment.csv")
    )

    # Save battery charging
    battery_charging_data = [(t, s, g_ch[s, t].X) for s in S for t in T]
    battery_charging_df = pd.DataFrame(
        battery_charging_data, columns=["snapshot", "battery", "value"]
    )
    battery_charging_reshaped = _reshape_variable(
        battery_charging_df, "battery", "snapshot"
    )
    battery_charging_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_charging.csv")
    )

    # Save battery discharging
    battery_discharging_data = [(t, s, g_dis[s, t].X) for s in S for t in T]
    battery_discharging_df = pd.DataFrame(
        battery_discharging_data, columns=["snapshot", "battery", "value"]
    )
    battery_discharging_reshaped = _reshape_variable(
        battery_discharging_df, "battery", "snapshot"
    )
    battery_discharging_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_discharging.csv")
    )

    # Save battery state of charge
    battery_soc_data = [(t, s, soc[s, t].X) for s in S for t in T]
    battery_soc_df = pd.DataFrame(
        battery_soc_data, columns=["snapshot", "battery", "value"]
    )
    battery_soc_reshaped = _reshape_variable(battery_soc_df, "battery", "snapshot")
    battery_soc_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_soc.csv")
    )

    # Save generator build
    generator_build_data = [(i, x[i].X) for i in G_new]
    generator_build_df = pd.DataFrame(
        generator_build_data, columns=["generator", "value"]
    )
    generator_build_df.to_csv(
        os.path.join(decision_variables_folder, "generator_build.csv"), index=False
    )

    # Save branch build
    branch_build_data = [(b, y[b].X) for b in B_new]
    branch_build_df = pd.DataFrame(branch_build_data, columns=["branch", "value"])
    branch_build_df.to_csv(
        os.path.join(decision_variables_folder, "branch_build.csv"), index=False
    )

    # Save battery build
    battery_build_data = [(s, z[s].X) for s in S_new]
    battery_build_df = pd.DataFrame(battery_build_data, columns=["battery", "value"])
    battery_build_df.to_csv(
        os.path.join(decision_variables_folder, "battery_build.csv"), index=False
    )

    # Save generator capacities
    generator_capacity_data = [(i, p_i_max[i].X) for i in G_new]
    generator_capacity_df = pd.DataFrame(
        generator_capacity_data, columns=["generator", "value"]
    )
    generator_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "generator_capacity.csv"), index=False
    )

    # Save branch capacities
    branch_capacity_data = [(b, p_b_max[b].X) for b in B_new]
    branch_capacity_df = pd.DataFrame(branch_capacity_data, columns=["branch", "value"])
    branch_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "branch_capacity.csv"), index=False
    )

    # endregion

    return (
        model,
        build_end_time - build_start_time,
        model_optimize_end_time - model_optimize_start_time,
    )


# endregion


# region GTSEP_v1
def GTSEP_v1(config: dict) -> gp.Model:
    """GTSEP model from the specialization project. Modeling battery investments as continous variables (unconstrained investments in batteries)."""
    # region Model setup and running
    must_have_keys = [
        "data_folder_name",
        "VOLL",
        "CC",
        "CO2_price",
        "E_limit",
        "p_max_new_branch",
        "p_min_new_branch",
        "expansion_factor",
        "MS",
        "model_name",
        "MIPGap",
    ]
    for key in must_have_keys:
        if key not in config:
            raise KeyError(
                f"Required key '{key}' not found in config. \nRequired keys: {must_have_keys}\nConfig keys: {config.keys()}"
            )

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

    # Load data
    data_folder_path = os.path.join(PROCESSED_DATA_FOLDER, data_folder_name)
    input_data = load_csv_files_from_folder(data_folder_path)
    batteries = input_data["batteries"]
    branches = input_data["branches"]
    capacity_factors = input_data["capacity_factors"]
    generators = input_data["generators"]
    generator_costs = input_data["generator_costs"]
    hourly_demand = input_data["hourly_demand"]
    nodes = input_data["nodes"]

    # Data processing
    # Create new branches
    # Add a new column 'exists' to the original branches dataframe and set it to 1
    branches["exists"] = 1
    # Create a copy of the dataframe for the "new" branches
    branches_new = branches.copy()
    # Update the index by appending " new" to the original index
    branches_new.index = branches_new.index.astype(str) + " new"
    # Set the 'exists' column to 0 for the new branches
    branches_new["exists"] = 0
    # Concatenate the original dataframe and the new dataframe
    branches = pd.concat([branches, branches_new])
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
    batteries["exists"] = 0

    # Create sets
    N = nodes.index.to_list()
    G_old = generators[generators["exists"] == 1].index.to_list()
    G_new = generators[generators["exists"] == 0].index.to_list()
    G = generators.index.to_list()
    B_old = branches[branches["exists"] == 1].index.to_list()
    B_new = branches[branches["exists"] == 0].index.to_list()
    B = branches.index.to_list()
    S_new = batteries[batteries["exists"] == 0].index.to_list()
    S_old = batteries[batteries["exists"] == 1].index.to_list()
    S = batteries.index.to_list()
    T = hourly_demand.index.to_list()

    # Create mappings
    (
        branches_out_of_node,
        branches_into_node,
        batteries_at_node,
        generators_at_node,
    ) = _create_mappings(nodes, branches, generators, batteries)

    build_start_time = time()
    # Create model
    model_name = model_name if model_name else "GTSEP_v0"
    model = gp.Model(model_name)

    # Decision variables
    g = model.addVars(G, T, name="g", lb=0)  # Power generation dispatch
    f = model.addVars(B, T, name="f", lb=-GRB.INFINITY, ub=GRB.INFINITY)  # Power flow
    sh = model.addVars(N, T, name="sh", lb=0)  # Load shedding
    c = model.addVars(G, T, name="c", lb=0)  # Curtailment
    g_ch = model.addVars(S, T, name="g_ch", lb=0)  # Battery charging
    g_dis = model.addVars(S, T, name="g_dis", lb=0)  # Battery discharging
    soc = model.addVars(S, T, name="soc", lb=0)  # State of charge
    x = model.addVars(G_new, vtype=GRB.BINARY, name="x")  # Binary for new generators
    y = model.addVars(B_new, vtype=GRB.BINARY, name="y")  # Binary for new branches
    soc_s_max = model.addVars(
        S_new, name="soc_s_max", lb=0
    )  # Max SOC for new batteries
    p_i_max = model.addVars(
        G_new, name="p_i_max", lb=0
    )  # Max capacity of new generators
    p_b_max = model.addVars(B_new, name="p_b_max", lb=0)  # Max capacity of new branches

    # Objective function: Minimize cost
    objective = (
        gp.quicksum(
            (
                generators.loc[i, "marginal_cost"]
                + generators.loc[i, "co2_emissions"] * CO2_price
            )
            * g[i, t]
            for i in G
            for t in T
        )
        # + gp.quicksum(
        #     batteries.loc[s, "MC"] * g_dis[s, t] * batteries.loc[s, "eta_discharge"]
        #     for s in S
        #     for t in T
        # )
        + gp.quicksum(VOLL * sh[n, t] for n in N for t in T)
        + gp.quicksum(CC * c[i, t] for i in G for t in T)
        + gp.quicksum(generators.loc[i, "capital_cost"] * p_i_max[i] for i in G_new)
        + gp.quicksum(branches.loc[b, "capital_cost"] * p_b_max[b] for b in B_new)
        + gp.quicksum(batteries.loc[s, "capital_cost"] * soc_s_max[s] for s in S_new)
    )
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
    # 1. Power balance
    for n in N:
        for t in T:
            model.addConstr(
                gp.quicksum(g[i, t] - c[i, t] for i in generators_at_node[n])
                + gp.quicksum(
                    f[b, t] * (1 - branches.loc[b, "loss_factor"])
                    for b in branches_into_node[n]
                )
                - gp.quicksum(f[b, t] for b in branches_out_of_node[n])
                - gp.quicksum(
                    g_ch[s, t] - batteries.loc[s, "eta_discharge"] * g_dis[s, t]
                    for s in batteries_at_node[n]
                )
                + sh[n, t]
                == hourly_demand.loc[t, n]
            )

    # 2a. Load shedding limits
    for n in N:
        for t in T:
            model.addConstr(sh[n, t] <= MS * hourly_demand.loc[t, n])

    # 2b. Curtailment limits
    for i in G:
        for t in T:
            model.addConstr(c[i, t] <= g[i, t])

    # 3a. Generator output limits (old generators)
    for i in G_old:
        p_max = generators.loc[i, "p_nom"]
        for t in T:
            capacity_factor = capacity_factors.loc[t, i]
            model.addConstr(g[i, t] <= p_max * capacity_factor)
            # Lower bound is 0 by default

    # 3b. Generator output limits (new generators)
    for i in G_new:
        for t in T:
            original_generator_id = " ".join(i.split(" ")[:-1])
            capacity_factor = capacity_factors.loc[t, original_generator_id]
            model.addConstr(g[i, t] <= x[i] * p_i_max[i] * capacity_factor)
            # Lower bound is 0 by default

    # 3c. New generator capacity limits
    for i in G_new:
        p_max = generators.loc[i, "p_nom"]
        model.addConstr(p_i_max[i] <= expansion_factor * p_max)

    # 4a. Branch flow limits (old branches)
    for b in B_old:
        for t in T:
            model.addConstr(f[b, t] >= -branches.loc[b, "p_max"])
            model.addConstr(f[b, t] <= branches.loc[b, "p_max"])

    # 4b. Branch flow limits (new branches)
    for b in B_new:
        for t in T:
            model.addConstr(f[b, t] >= -y[b] * p_b_max[b])
            model.addConstr(f[b, t] <= y[b] * p_b_max[b])

    # 4c. New branch capacity limits
    for b in B_new:
        model.addConstr(p_b_max[b] >= y[b] * p_min_new_branch)
        model.addConstr(p_b_max[b] <= y[b] * p_max_new_branch)

    # # 5. Emission restrictions
    # model.addConstr(
    #     gp.quicksum(g[i, t] * generators.loc[i, "co2_emissions"] for i in G for t in T)
    #     <= E_limit
    # )

    # 6a. Battery charging limits, old batteries
    for s in S_old:
        for t in T:
            model.addConstr(g_ch[s, t] >= batteries.loc[s, "P_charge_min"])
            model.addConstr(g_ch[s, t] <= batteries.loc[s, "P_charge_max"])

    # 6b. Battery charging limits, new batteries
    for s in S_new:
        for t in T:
            model.addConstr(g_ch[s, t] >= 0)
            model.addConstr(
                g_ch[s, t]
                <= soc_s_max[s]
                / (batteries.loc[s, "hour_capacity"] * batteries.loc[s, "cdrate"])
            )

    # 7a. Battery discharging limits, old batteries
    for s in S_old:
        for t in T:
            model.addConstr(g_dis[s, t] >= batteries.loc[s, "P_discharge_min"])
            model.addConstr(g_dis[s, t] <= batteries.loc[s, "P_discharge_max"])

    # 7b. Battery discharging limits, new batteries
    for s in S_new:
        for t in T:
            model.addConstr(g_dis[s, t] >= 0)
            model.addConstr(
                g_dis[s, t] <= soc_s_max[s] / (batteries.loc[s, "hour_capacity"])
            )

    # 8. State of charge limits
    for s in S:
        for t in T:
            model.addConstr(soc[s, t] >= batteries.loc[s, "SOC_min"] * soc_s_max[s])
            model.addConstr(soc[s, t] <= batteries.loc[s, "SOC_max"] * soc_s_max[s])

    # 9. Battery state of charge dynamics
    for s in S:
        for t in T[1:]:  # Exclude time t=0
            model.addConstr(
                soc[s, t]
                == soc[s, t - pd.Timedelta("1h")]
                + batteries.loc[s, "eta_charge"] * g_ch[s, t]
                - g_dis[s, t] / batteries.loc[s, "eta_discharge"]
            )

    # 10. Initial state of charge
    for s in S:
        model.addConstr(soc[s, T[0]] == batteries.loc[s, "SOC_min"] * soc_s_max[s])

    # Optimize the model
    model.setParam("MIPGap", MIPGap)
    model.setParam("Timelimit", TIMELIMIT)
    model.setParam("BarConvTol", MIPGap)

    build_end_time = time()

    print(f"Model built in {build_end_time - build_start_time} seconds.")
    model_optimize_start_time = time()
    model.optimize()
    model_optimize_end_time = time()
    # endregion

    # region Post-processing and saving results
    save_folder = config.get("save_folder", None)
    decision_variables_folder = os.path.join(save_folder, "decision_variables")
    if not os.path.exists(decision_variables_folder):
        os.makedirs(decision_variables_folder)
    # Save generation
    generation_data = [(t, i, g[i, t].X) for i in G for t in T]
    generation_df = pd.DataFrame(
        generation_data, columns=["snapshot", "generator", "value"]
    )
    generation_reshaped = _reshape_variable(generation_df, "generator", "snapshot")
    generation_reshaped.to_csv(
        os.path.join(decision_variables_folder, "generation.csv")
    )

    # Save power flow
    power_flow_data = [(t, b, f[b, t].X) for b in B for t in T]
    power_flow_df = pd.DataFrame(
        power_flow_data, columns=["snapshot", "branch", "value"]
    )
    power_flow_reshaped = _reshape_variable(power_flow_df, "branch", "snapshot")
    power_flow_reshaped.to_csv(
        os.path.join(decision_variables_folder, "power_flow.csv")
    )

    # Save load shedding
    load_shedding_data = [(t, n, sh[n, t].X) for n in N for t in T]
    load_shedding_df = pd.DataFrame(
        load_shedding_data, columns=["snapshot", "node", "value"]
    )
    load_shedding_reshaped = _reshape_variable(load_shedding_df, "node", "snapshot")
    load_shedding_reshaped.to_csv(
        os.path.join(decision_variables_folder, "load_shedding.csv")
    )

    # Save curtailment
    curtailment_data = [(t, i, c[i, t].X) for i in G for t in T]
    curtailment_df = pd.DataFrame(
        curtailment_data, columns=["snapshot", "generator", "value"]
    )
    curtailment_reshaped = _reshape_variable(curtailment_df, "generator", "snapshot")
    curtailment_reshaped.to_csv(
        os.path.join(decision_variables_folder, "curtailment.csv")
    )

    # Save battery charging
    battery_charging_data = [(t, s, g_ch[s, t].X) for s in S for t in T]
    battery_charging_df = pd.DataFrame(
        battery_charging_data, columns=["snapshot", "battery", "value"]
    )
    battery_charging_reshaped = _reshape_variable(
        battery_charging_df, "battery", "snapshot"
    )
    battery_charging_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_charging.csv")
    )

    # Save battery discharging
    battery_discharging_data = [(t, s, g_dis[s, t].X) for s in S for t in T]
    battery_discharging_df = pd.DataFrame(
        battery_discharging_data, columns=["snapshot", "battery", "value"]
    )
    battery_discharging_reshaped = _reshape_variable(
        battery_discharging_df, "battery", "snapshot"
    )
    battery_discharging_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_discharging.csv")
    )

    # Save battery state of charge
    battery_soc_data = [(t, s, soc[s, t].X) for s in S for t in T]
    battery_soc_df = pd.DataFrame(
        battery_soc_data, columns=["snapshot", "battery", "value"]
    )
    battery_soc_reshaped = _reshape_variable(battery_soc_df, "battery", "snapshot")
    battery_soc_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_soc.csv")
    )

    # Save generator build
    generator_build_data = [(i, x[i].X) for i in G_new]
    generator_build_df = pd.DataFrame(
        generator_build_data, columns=["generator", "value"]
    )
    generator_build_df.to_csv(
        os.path.join(decision_variables_folder, "generator_build.csv"), index=False
    )

    # Save branch build
    branch_build_data = [(b, y[b].X) for b in B_new]
    branch_build_df = pd.DataFrame(branch_build_data, columns=["branch", "value"])
    branch_build_df.to_csv(
        os.path.join(decision_variables_folder, "branch_build.csv"), index=False
    )

    # Save battery build
    battery_build_data = [(s, soc_s_max[s].X) for s in S_new]
    battery_build_df = pd.DataFrame(battery_build_data, columns=["battery", "value"])
    battery_build_df.to_csv(
        os.path.join(decision_variables_folder, "battery_build.csv"), index=False
    )

    # Save generator capacities
    generator_capacity_data = [(i, p_i_max[i].X) for i in G_new]
    generator_capacity_df = pd.DataFrame(
        generator_capacity_data, columns=["generator", "value"]
    )
    generator_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "generator_capacity.csv"), index=False
    )

    # Save branch capacities
    branch_capacity_data = [(b, p_b_max[b].X) for b in B_new]
    branch_capacity_df = pd.DataFrame(branch_capacity_data, columns=["branch", "value"])
    branch_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "branch_capacity.csv"), index=False
    )

    # endregion

    return (
        model,
        build_end_time - build_start_time,
        model_optimize_end_time - model_optimize_start_time,
    )


# endregion

# region GTSEP_v1_multi


def GTSEP_v1_multi(config: dict) -> gp.Model:
    """GTSEP_v1 model but with multiple time periods considered."""
    # region Model setup and running
    must_have_keys = [
        "data_folder_name",
        "VOLL",
        "CC",
        "CO2_price",
        "E_limit",
        "p_max_new_branch",
        "p_min_new_branch",
        "expansion_factor",
        "MS",
        "model_name",
        "MIPGap",
        "years",
        "discount_rate",
    ]
    for key in must_have_keys:
        if key not in config:
            raise KeyError(
                f"Required key '{key}' not found in config. \nRequired keys: {must_have_keys}\nConfig keys: {config.keys()}"
            )

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

    if not years:
        raise ValueError("Years must be provided for a multi-year model.")

    # Load data
    data_folder_path = os.path.join(PROCESSED_DATA_FOLDER, data_folder_name)
    input_data = load_multi_year_csv_files_from_folder(years, data_folder_path)
    batteries = input_data["batteries"]
    branches = input_data["branches"]
    capacity_factors = input_data["capacity_factors"]
    generators = input_data["generators"]
    generator_costs = input_data["generator_costs"]
    hourly_demand = input_data["hourly_demand"]
    nodes = input_data["nodes"]

    # Data processing
    # Create new branches
    # Add a new column 'exists' to the original branches dataframe and set it to 1
    branches["exists"] = 1
    # Create a copy of the dataframe for the "new" branches
    branches_new = branches.copy()
    # Update the index by appending " new" to the original index
    branches_new.index = branches_new.index.set_levels(
        branches_new.index.levels[1].astype(str) + " new", level="line"
    )
    # Set the 'exists' column to 0 for the new branches
    branches_new["exists"] = 0
    # Concatenate the original dataframe and the new dataframe
    branches = pd.concat([branches, branches_new])
    # Add a new column 'exists' to the original dataframe and set it to 1
    generators["exists"] = 1
    # Create a copy of the dataframe for the "new" generators
    generators_new = generators.copy()
    # Update the index by appending " new" to the original index
    generators_new.index = generators_new.index.set_levels(
        generators_new.index.levels[1].astype(str) + " new", level="generator"
    )
    # Set the 'exists' column to 0 for the new generators
    generators_new["exists"] = 0
    # Concatenate the original dataframe and the new dataframe
    generators = pd.concat([generators, generators_new])
    batteries["exists"] = 0

    # Create sets
    N = nodes.index.to_list()
    G_old = (
        generators[generators["exists"] == 1]
        .index.get_level_values("generator")
        .unique()
        .to_list()
    )
    G_new = (
        generators[generators["exists"] == 0]
        .index.get_level_values("generator")
        .unique()
        .to_list()
    )
    G = generators.index.get_level_values("generator").unique().to_list()
    B_old = (
        branches[branches["exists"] == 1]
        .index.get_level_values("line")
        .unique()
        .to_list()
    )
    B_new = (
        branches[branches["exists"] == 0]
        .index.get_level_values("line")
        .unique()
        .to_list()
    )
    B = branches.index.get_level_values("line").unique().to_list()
    S_new = (
        batteries[batteries["exists"] == 0]
        .index.get_level_values("battery")
        .unique()
        .to_list()
    )
    S_old = (
        batteries[batteries["exists"] == 1]
        .index.get_level_values("battery")
        .unique()
        .to_list()
    )
    S = batteries.index.get_level_values("battery").unique().to_list()
    Y = hourly_demand.index.get_level_values("year").unique().to_list()
    T = hourly_demand.index.get_level_values("hour").unique().to_list()

    # Create mappings
    (
        branches_out_of_node,
        branches_into_node,
        batteries_at_node,
        generators_at_node,
    ) = _create_mappings(nodes, branches, generators, batteries)

    # create Yy mapping, accessed as Yy[y] and returns a list of years up to and including y from Y
    Yy = {y: [x for x in Y if x <= y] for y in Y}

    build_start_time = time()
    # Create model
    model_name = model_name if model_name else "GTSEP_v0"
    model = gp.Model(model_name)

    # Decision variables
    g = model.addVars(G, Y, T, name="g", lb=0)  # Power generation dispatch
    f = model.addVars(
        B, Y, T, name="f", lb=-GRB.INFINITY, ub=GRB.INFINITY
    )  # Power flow
    sh = model.addVars(N, Y, T, name="sh", lb=0)  # Load shedding
    c = model.addVars(G, Y, T, name="c", lb=0)  # Curtailment
    g_ch = model.addVars(S, Y, T, name="g_ch", lb=0)  # Battery charging
    g_dis = model.addVars(S, Y, T, name="g_dis", lb=0)  # Battery discharging
    soc = model.addVars(S, Y, T, name="soc", lb=0)  # State of charge
    # x = model.addVars(G_new, Y, vtype=GRB.BINARY, name="xi")  # Binary for new generators
    # y = model.addVars(B_new, Y, vtype=GRB.BINARY, name="xb")  # Binary for new branches
    soc_s_max = model.addVars(
        S_new, Y, name="soc_s_max", lb=0
    )  # Max SOC for new batteries
    p_i_max = model.addVars(
        G_new, Y, name="p_i_max", lb=0
    )  # Max capacity of new generators
    p_b_max = model.addVars(
        B_new, Y, name="p_b_max", lb=0
    )  # Max capacity of new branches

    # Cumulative capacity helper variables
    # Cumulative capacity for new generators
    p_i_cum_max = model.addVars(G_new, Y, name="p_i_cum_max", lb=0)
    for i in G_new:
        for y in Y:
            model.addConstr(
                p_i_cum_max[i, y]
                == gp.quicksum(p_i_max[i, y_marked] for y_marked in Yy[y])
            )
    # Cumulative capacity for new branches
    p_b_cum_max = model.addVars(B_new, Y, name="p_b_cum_max", lb=0)
    for b in B_new:
        for y in Y:
            model.addConstr(
                p_b_cum_max[b, y]
                == gp.quicksum(p_b_max[b, y_marked] for y_marked in Yy[y])
            )
    # Cumulative capacity for new batteries
    soc_s_cum_max = model.addVars(S_new, Y, name="soc_s_cum_max", lb=0)
    for s in S_new:
        for y in Y:
            model.addConstr(
                soc_s_cum_max[s, y]
                == gp.quicksum(soc_s_max[s, y_marked] for y_marked in Yy[y])
            )

    objective = 0.0
    for y in Y:
        OC = (
            gp.quicksum(
                (
                    generators.loc[(y, i), "marginal_cost"]
                    + generators.loc[(y, i), "co2_emissions"] * CO2_price
                )
                * g[i, y, t]
                for i in G
                for t in T
            )
            + gp.quicksum(VOLL * sh[n, y, t] for n in N for t in T)
            + gp.quicksum(CC * c[i, y, t] for i in G for t in T)
        )
        AIC = (
            gp.quicksum(
                generators.loc[(y, i), "capital_cost"] * p_i_max[i, y] for i in G_new
            )
            + gp.quicksum(
                branches.loc[(y, b), "capital_cost"] * p_b_max[b, y] for b in B_new
            )
            + gp.quicksum(
                batteries.loc[(y, s), "capital_cost"] * soc_s_max[s, y] for s in S_new
            )
        )
        objective += OC + AIC
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
    # 1. Power balance
    for n in N:
        for y in Y:
            for t in T:
                model.addConstr(
                    gp.quicksum(g[i, y, t] - c[i, y, t] for i in generators_at_node[n])
                    + gp.quicksum(
                        f[b, y, t] * (1 - branches.loc[(y, b), "loss_factor"])
                        for b in branches_into_node[n]
                    )
                    - gp.quicksum(f[b, y, t] for b in branches_out_of_node[n])
                    - gp.quicksum(
                        g_ch[s, y, t]
                        - batteries.loc[(y, s), "eta_discharge"] * g_dis[s, y, t]
                        for s in batteries_at_node[n]
                    )
                    + sh[n, y, t]
                    == hourly_demand.loc[(y, t), n]
                )

    # 2a. Load shedding limits
    for n in N:
        for y in Y:
            for t in T:
                model.addConstr(sh[n, y, t] <= MS * hourly_demand.loc[(y, t), n])

    # 2b. Curtailment limits
    for i in G:
        for y in Y:
            for t in T:
                model.addConstr(c[i, y, t] <= g[i, y, t])

    # 3a. Generator output limits (old generators)
    for i in G_old:
        for y in Y:
            p_max = generators.loc[(y, i), "p_nom"]
            for t in T:
                capacity_factor = capacity_factors.loc[(y, t), i]
                model.addConstr(g[i, y, t] <= p_max * capacity_factor)
                # Lower bound is 0 by default

    # 3b. Generator output limits (new generators)
    for i in G_new:
        for y in Y:
            for t in T:
                original_generator_id = " ".join(i.split(" ")[:-1])
                capacity_factor = capacity_factors.loc[(y, t), original_generator_id]
                model.addConstr(g[i, y, t] <= capacity_factor * p_i_cum_max[i, y])
                # Lower bound is 0 by default

    # 3c. New generator capacity limits
    for i in G_new:
        for y in Y:
            p_max = generators.loc[(y, i), "p_nom"]
            model.addConstr(p_i_max[i, y] <= expansion_factor * p_max)

    # 4a. Branch flow limits (old branches)
    for b in B_old:
        for y in Y:
            for t in T:
                model.addConstr(f[b, y, t] >= -branches.loc[(y, b), "p_max"])
                model.addConstr(f[b, y, t] <= branches.loc[(y, b), "p_max"])

    # 4b. Branch flow limits (new branches)
    for b in B_new:
        for y in Y:
            for t in T:
                model.addConstr(f[b, y, t] >= -p_b_cum_max[b, y])
                model.addConstr(f[b, y, t] <= p_b_cum_max[b, y])

    # 4c. New branch capacity limits
    for b in B_new:
        for y in Y:
            model.addConstr(p_b_max[b, y] <= p_max_new_branch)

    # # 5. Emission restrictions
    # model.addConstr(
    #     gp.quicksum(
    #         g[i, y, t] * generators.loc[i, "co2_emissions"]
    #         for i in G
    #         for y in Y
    #         for t in T
    #     )
    #     <= E_limit
    # )

    # 6a. Battery charging limits, old batteries
    for s in S_old:
        for y in Y:
            for t in T:
                model.addConstr(g_ch[s, y, t] >= batteries.loc[(y, s), "P_charge_min"])
                model.addConstr(g_ch[s, y, t] <= batteries.loc[(y, s), "P_charge_max"])

    # 6b. Battery charging limits, new batteries
    for s in S_new:
        for y in Y:
            for t in T:
                model.addConstr(
                    g_ch[s, y, t]
                    <= soc_s_cum_max[s, y]
                    / (
                        batteries.loc[(y, s), "hour_capacity"]
                        * batteries.loc[(y, s), "cdrate"]
                    )
                )

    # 7a. Battery discharging limits, old batteries
    for s in S_old:
        for y in Y:
            for t in T:
                model.addConstr(
                    g_dis[s, y, t] >= batteries.loc[(y, s), "P_discharge_min"]
                )
                model.addConstr(
                    g_dis[s, y, t] <= batteries.loc[(y, s), "P_discharge_max"]
                )

    # 7b. Battery discharging limits, new batteries
    for s in S_new:
        for y in Y:
            for t in T:
                model.addConstr(
                    g_dis[s, y, t]
                    <= soc_s_cum_max[s, y] / (batteries.loc[(y, s), "hour_capacity"])
                )

    # 8. State of charge limits
    for s in S:
        for y in Y:
            for t in T:
                model.addConstr(
                    soc[s, y, t] >= batteries.loc[(y, s), "SOC_min"] * soc_s_max[s, y]
                )
                model.addConstr(
                    soc[s, y, t] <= batteries.loc[(y, s), "SOC_max"] * soc_s_max[s, y]
                )

    # 9. Battery state of charge dynamics
    for s in S:
        for y in Y:
            for t in T[1:]:  # Exclude time t=0
                model.addConstr(
                    soc[s, y, t]
                    == soc[s, y, t - 1]
                    + batteries.loc[(y, s), "eta_charge"] * g_ch[s, y, t]
                    - g_dis[s, y, t] / batteries.loc[(y, s), "eta_discharge"]
                )

    # 10a. Initial state of charge and state of charge end of period
    for s in S:
        for y in Y:
            model.addConstr(
                soc[s, y, T[0]] == batteries.loc[(y, s), "SOC_min"] * soc_s_max[s, y]
            )
            model.addConstr(
                soc[s, y, T[-1]] == batteries.loc[(y, s), "SOC_min"] * soc_s_max[s, y]
            )

    # Optimize the model
    model.setParam("MIPGap", MIPGap)
    model.setParam("Timelimit", TIMELIMIT)
    model.setParam("BarConvTol", MIPGap)
    model.setParam("BarHomogeneous", 1)

    build_end_time = time()

    print(f"Model built in {build_end_time - build_start_time} seconds.")
    model_optimize_start_time = time()
    model.optimize()
    model_optimize_end_time = time()
    # endregion

    # region Post-processing and saving results
    save_folder = config.get("save_folder", "")
    decision_variables_folder = os.path.join(save_folder, "decision_variables")
    if not os.path.exists(decision_variables_folder):
        os.makedirs(decision_variables_folder)
    # Save generation
    generation_data = [(y, t, i, g[i, y, t].X) for i in G for y in Y for t in T]
    generation_df = pd.DataFrame(
        generation_data, columns=["year", "hour", "generator", "value"]
    )
    generation_reshaped = _reshape_multi(
        generation_df, ["year", "hour"], "generator", "value"
    )
    generation_reshaped.to_csv(
        os.path.join(decision_variables_folder, "generation.csv")
    )

    # Save power flow
    power_flow_data = [(y, t, b, f[b, y, t].X) for b in B for y in Y for t in T]
    power_flow_df = pd.DataFrame(
        power_flow_data, columns=["year", "hour", "branch", "value"]
    )
    power_flow_reshaped = _reshape_multi(
        power_flow_df, ["year", "hour"], "branch", "value"
    )
    power_flow_reshaped.to_csv(
        os.path.join(decision_variables_folder, "power_flow.csv")
    )

    # Save load shedding
    load_shedding_data = [(y, t, n, sh[n, y, t].X) for n in N for y in Y for t in T]
    load_shedding_df = pd.DataFrame(
        load_shedding_data, columns=["year", "hour", "node", "value"]
    )
    load_shedding_reshaped = _reshape_multi(
        load_shedding_df, ["year", "hour"], "node", "value"
    )
    load_shedding_reshaped.to_csv(
        os.path.join(decision_variables_folder, "load_shedding.csv")
    )

    # Save curtailment
    curtailment_data = [(y, t, i, c[i, y, t].X) for i in G for y in Y for t in T]
    curtailment_df = pd.DataFrame(
        curtailment_data, columns=["year", "hour", "generator", "value"]
    )
    curtailment_reshaped = _reshape_multi(
        curtailment_df, ["year", "hour"], "generator", "value"
    )
    curtailment_reshaped.to_csv(
        os.path.join(decision_variables_folder, "curtailment.csv")
    )

    # Save battery charging
    battery_charging_data = [
        (y, t, s, g_ch[s, y, t].X) for s in S for y in Y for t in T
    ]
    battery_charging_df = pd.DataFrame(
        battery_charging_data, columns=["year", "hour", "battery", "value"]
    )
    battery_charging_reshaped = _reshape_multi(
        battery_charging_df, ["year", "hour"], "battery", "value"
    )
    battery_charging_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_charging.csv")
    )

    # Save battery discharging
    battery_discharging_data = [
        (y, t, s, g_dis[s, y, t].X) for s in S for y in Y for t in T
    ]
    battery_discharging_df = pd.DataFrame(
        battery_discharging_data, columns=["year", "hour", "battery", "value"]
    )
    battery_discharging_reshaped = _reshape_multi(
        battery_discharging_df, ["year", "hour"], "battery", "value"
    )
    battery_discharging_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_discharging.csv")
    )

    # Save battery state of charge
    battery_soc_data = [(y, t, s, soc[s, y, t].X) for s in S for y in Y for t in T]
    battery_soc_df = pd.DataFrame(
        battery_soc_data, columns=["year", "hour", "battery", "value"]
    )
    battery_soc_reshaped = _reshape_multi(
        battery_soc_df, ["year", "hour"], "battery", "value"
    )
    battery_soc_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_soc.csv")
    )

    # Save battery build
    battery_capacity_data = [(y, s, soc_s_max[s, y].X) for s in S_new for y in Y]
    battery_capacity_df = pd.DataFrame(
        battery_capacity_data, columns=["year", "battery", "value"]
    )
    battery_capacity_df = _reshape_multi(
        battery_capacity_df, "battery", "year", "value"
    )
    battery_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "battery_capacity.csv"), index=True
    )

    # Save generator capacities
    generator_capacity_data = [(y, i, p_i_max[i, y].X) for i in G_new for y in Y]
    generator_capacity_df = pd.DataFrame(
        generator_capacity_data, columns=["year", "generator", "value"]
    )
    generator_capacity_df = _reshape_multi(
        generator_capacity_df, "generator", "year", "value"
    )
    generator_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "generator_capacity.csv"), index=True
    )

    # Save branch capacities
    branch_capacity_data = [(y, b, p_b_max[b, y].X) for b in B_new for y in Y]
    branch_capacity_df = pd.DataFrame(
        branch_capacity_data, columns=["year", "branch", "value"]
    )
    branch_capacity_df = _reshape_multi(branch_capacity_df, "branch", "year", "value")
    branch_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "branch_capacity.csv"), index=True
    )

    # endregion

    return (
        model,
        build_end_time - build_start_time,
        model_optimize_end_time - model_optimize_start_time,
    )


# endregion


def GTSEP_v1a_multi(config: dict) -> gp.Model:
    """GTSEP_v1 model but with multiple time periods considered. Uses clustering"""
    # region Model setup and running
    must_have_keys = [
        "data_folder_name",
        "VOLL",
        "CC",
        "CO2_price",
        "E_limit",
        "p_max_new_branch",
        "p_min_new_branch",
        "expansion_factor",
        "MS",
        "model_name",
        "MIPGap",
        "years",
        "discount_rate",
        "representative_period_unit",
        "representative_periods",
    ]
    for key in must_have_keys:
        if key not in config:
            raise KeyError(
                f"Required key '{key}' not found in config. \nRequired keys: {must_have_keys}\nConfig keys: {config.keys()}"
            )

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
    if not representative_period_unit == "week":
        raise ValueError(
            "Only 'week' is supported as representative_period_unit as of now."
        )

    if not years:
        raise ValueError("Years must be provided for a multi-year model.")

    # Load data
    data_folder_path = os.path.join(PROCESSED_DATA_FOLDER, data_folder_name)
    input_data = load_multi_year_csv_files_with_week_from_folder_with_demand_scaling(
        years, weeks, data_folder_path
    )
    batteries = input_data["batteries"]
    branches = input_data["branches"]
    capacity_factors = input_data["capacity_factors"]
    generators = input_data["generators"]
    generator_costs = input_data["generator_costs"]
    hourly_demand = input_data["hourly_demand"]
    nodes = input_data["nodes"]

    # Set extenstion potential for branches and generators
    branches["extension_potential"] = p_max_new_branch * branches["extendable"]
    generators["extension_potential"] = (
        generators["p_nom"] * generators["extendable"] * expansion_factor
    )

    # Create sets
    N = nodes.index.to_list()
    G_new = (
        generators[generators["extendable"] == True]
        .index.get_level_values("generator")
        .unique()
        .tolist()
    )
    G = generators.index.get_level_values("generator").unique().tolist()
    G_old = G

    B_new = (
        branches[branches["extendable"] == True]
        .index.get_level_values("line")
        .unique()
        .tolist()
    )
    B = branches.index.get_level_values("line").unique().tolist()
    B_old = B

    S = batteries.index.get_level_values("battery").unique().tolist()
    S_new = S  # Assuming all batteries are new investments
    S_old = []  # Assuming no old batteries

    N = nodes.index.tolist()
    Y = hourly_demand.index.get_level_values("year").unique().tolist()
    W = weeks
    T = (
        hourly_demand[hourly_demand.index.get_level_values("week").isin(weeks)]
        .index.get_level_values("hour")
        .unique()
        .tolist()
    )

    if not all(
        len(T)
        == len(
            hourly_demand[
                (hourly_demand.index.get_level_values("week") == w)
                & (hourly_demand.index.get_level_values("year") == y)
            ]
        )
        for w in W
        for y in Y
    ):
        raise ValueError(f"All weeks must have the same number of hours.")

    week_weights = {w: 1 / len(W) * 8760 / len(T) for w in W}

    # Create mappings
    (
        branches_out_of_node,
        branches_into_node,
        batteries_at_node,
        generators_at_node,
    ) = _create_mappings(nodes, branches, generators, batteries)

    # create Yy mapping, accessed as Yy[y] and returns a list of years up to and including y from Y
    Yy = {y: [x for x in Y if x <= y] for y in Y}

    build_start_time = time()
    # Create model
    model_name = model_name if model_name else "GTSEP_v0"
    model = gp.Model(model_name)

    # Decision variables (with representative weeks)
    g = model.addVars(G, Y, W, T, name="g", lb=0)  # Power generation dispatch
    f = model.addVars(
        B, Y, W, T, name="f", lb=-GRB.INFINITY, ub=GRB.INFINITY
    )  # Power flow
    # Dispatch from capacity extensions
    g_new = model.addVars(G_new, Y, W, T, name="g_new", lb=0)
    f_new = model.addVars(
        B_new, Y, W, T, name="f_new", lb=-GRB.INFINITY, ub=GRB.INFINITY
    )
    sh = model.addVars(N, Y, W, T, name="sh", lb=0)  # Load shedding
    c = model.addVars(G, Y, W, T, name="c", lb=0)  # Curtailment
    g_ch = model.addVars(S, Y, W, T, name="g_ch", lb=0)  # Battery charging
    g_dis = model.addVars(S, Y, W, T, name="g_dis", lb=0)  # Battery discharging
    soc = model.addVars(S, Y, W, T, name="soc", lb=0)  # State of charge

    # Investment variables remain unchanged (no weekly dependency)
    soc_s_max = model.addVars(
        S_new, Y, name="soc_s_max", lb=0
    )  # Max SOC for new batteries
    p_i_max = model.addVars(
        G_new, Y, name="p_i_max", lb=0
    )  # Max capacity of new generators
    p_b_max = model.addVars(
        B_new, Y, name="p_b_max", lb=0
    )  # Max capacity of new branches

    # Cumulative capacity helper variables (unchanged)
    p_i_cum_max = model.addVars(G_new, Y, name="p_i_cum_max", lb=0)
    for i in G_new:
        for y in Y:
            model.addConstr(
                p_i_cum_max[i, y]
                == gp.quicksum(p_i_max[i, y_marked] for y_marked in Yy[y])
            )

    p_b_cum_max = model.addVars(B_new, Y, name="p_b_cum_max", lb=0)
    for b in B_new:
        for y in Y:
            model.addConstr(
                p_b_cum_max[b, y]
                == gp.quicksum(p_b_max[b, y_marked] for y_marked in Yy[y])
            )

    soc_s_cum_max = model.addVars(S_new, Y, name="soc_s_cum_max", lb=0)
    for s in S_new:
        for y in Y:
            model.addConstr(
                soc_s_cum_max[s, y]
                == gp.quicksum(soc_s_max[s, y_marked] for y_marked in Yy[y])
            )

    objective = 0.0
    for y in Y:
        OC = gp.quicksum(
            week_weights[w]
            * (
                gp.quicksum(
                    (
                        generators.loc[(y, i), "marginal_cost"]
                        + generators.loc[(y, i), "co2_emissions"] * CO2_price
                    )
                    * (g[i, y, w, t] + (g_new[i, y, w, t] if i in G_new else 0))
                    for i in G
                    for t in T
                )
                + gp.quicksum(VOLL * sh[n, y, w, t] for n in N for t in T)
                + gp.quicksum(CC * c[i, y, w, t] for i in G for t in T)
            )
            for w in W
        )
        AIC = (
            gp.quicksum(
                generators.loc[(y, i), "capital_cost"] * p_i_max[i, y] for i in G_new
            )
            + gp.quicksum(
                branches.loc[(y, b), "capital_cost"] * p_b_max[b, y] for b in B_new
            )
            + gp.quicksum(
                batteries.loc[(y, s), "capital_cost"] * soc_s_max[s, y] for s in S_new
            )
        )
        objective += OC + AIC

    model.setObjective(objective, GRB.MINIMIZE)

    # 1. Power balance
    for n in N:
        for y in Y:
            for w in W:
                for t in T:
                    model.addConstr(
                        gp.quicksum(
                            g[i, y, w, t]
                            - c[i, y, w, t]
                            + (g_new[i, y, w, t] if i in G_new else 0)
                            for i in generators_at_node[n]
                        )
                        + gp.quicksum(
                            (f[b, y, w, t] + (f_new[b, y, w, t] if b in B_new else 0))
                            * (1 - branches.loc[(y, b), "loss_factor"])
                            for b in branches_into_node[n]
                        )
                        - gp.quicksum(
                            f[b, y, w, t] + (f_new[b, y, w, t] if b in B_new else 0)
                            for b in branches_out_of_node[n]
                        )
                        - gp.quicksum(
                            g_ch[s, y, w, t]
                            - batteries.loc[(y, s), "eta_discharge"] * g_dis[s, y, w, t]
                            for s in batteries_at_node[n]
                        )
                        + sh[n, y, w, t]
                        == hourly_demand.loc[(y, w, t), n],
                        name=f"C_power_balance[{n},{y},{w},{t}]",
                    )

    # 2a. Load shedding limits
    for n in N:
        for y in Y:
            for w in W:
                for t in T:
                    model.addConstr(
                        sh[n, y, w, t] <= MS * hourly_demand.loc[(y, w, t), n],
                        name=f"C_load_shedding_limit[{n},{y},{w},{t}]",
                    )

    # 2b. Curtailment limits
    for i in G:
        for y in Y:
            for w in W:
                for t in T:
                    model.addConstr(
                        c[i, y, w, t] <= g[i, y, w, t],
                        name=f"C_curtailment_limit[{i},{y},{w},{t}]",
                    )

    # 3a. Generator output limits (old)
    for i in G_old:
        for y in Y:
            p_max = generators.loc[(y, i), "p_nom"]
            for w in W:
                for t in T:
                    capacity_factor = capacity_factors.loc[(y, w, t), i]
                    model.addConstr(
                        g[i, y, w, t] <= p_max * capacity_factor,
                        name=f"C_gen_output_old[{i},{y},{w},{t}]",
                    )

    # 3b. Generator output limits (new)
    for i in G_new:
        for y in Y:
            for w in W:
                for t in T:
                    capacity_factor = capacity_factors.loc[(y, w, t), i]
                    model.addConstr(
                        g_new[i, y, w, t] <= capacity_factor * p_i_cum_max[i, y],
                        name=f"C_gen_output_new[{i},{y},{w},{t}]",
                    )

    # 3c. Generator capacity extension limits
    for i in G_new:
        for y in Y:
            extension_limit = generators.loc[(y, i), "extension_potential"]
            model.addConstr(
                p_i_max[i, y] <= extension_limit, name=f"C_gen_extension_limit[{i},{y}]"
            )

    # 4a. Branch flow limits (old)
    for b in B_old:
        for y in Y:
            for w in W:
                for t in T:
                    model.addConstr(
                        f[b, y, w, t] >= -branches.loc[(y, b), "p_max"],
                        name=f"C_branch_old_min[{b},{y},{w},{t}]",
                    )
                    model.addConstr(
                        f[b, y, w, t] <= branches.loc[(y, b), "p_max"],
                        name=f"C_branch_old_max[{b},{y},{w},{t}]",
                    )

    # 4b. Branch flow limits (new)
    for b in B_new:
        for y in Y:
            for w in W:
                for t in T:
                    model.addConstr(
                        f_new[b, y, w, t] >= -p_b_cum_max[b, y],
                        name=f"C_branch_new_min[{b},{y},{w},{t}]",
                    )
                    model.addConstr(
                        f_new[b, y, w, t] <= p_b_cum_max[b, y],
                        name=f"C_branch_new_max[{b},{y},{w},{t}]",
                    )

    # 4c. New branch capacity limits
    for b in B_new:
        for y in Y:
            extension_limit = branches.loc[(y, b), "extension_potential"]
            model.addConstr(
                p_b_max[b, y] <= extension_limit,
                name=f"C_branch_extension_limit[{b},{y}]",
            )

    # 5. Emission restrictions
    model.addConstr(
        gp.quicksum(
            week_weights[w]
            * (g[i, y, w, t] + g_new[i, y, w, t])
            * generators.loc[(y, i), "co2_emissions"]
            for i in G
            for y in Y
            for w in W
            for t in T
        )
        <= E_limit,
        name="C_emission_limit",
    )

    # 6a. Battery charging limits, old batteries
    for s in S_old:
        for y in Y:
            for w in W:
                for t in T:
                    model.addConstr(
                        g_ch[s, y, w, t] >= batteries.loc[(y, s), "P_charge_min"],
                        name=f"C_batt_charge_old_min[{s},{y},{w},{t}]",
                    )
                    model.addConstr(
                        g_ch[s, y, w, t] <= batteries.loc[(y, s), "P_charge_max"],
                        name=f"C_batt_charge_old_max[{s},{y},{w},{t}]",
                    )

    # 6b. Battery charging limits, new batteries
    for s in S_new:
        for y in Y:
            for w in W:
                for t in T:
                    model.addConstr(
                        g_ch[s, y, w, t]
                        <= soc_s_cum_max[s, y]
                        / (
                            batteries.loc[(y, s), "hour_capacity"]
                            * batteries.loc[(y, s), "cdrate"]
                        ),
                        name=f"C_batt_charge_new_max[{s},{y},{w},{t}]",
                    )

    # 7a. Battery discharging limits, old batteries
    for s in S_old:
        for y in Y:
            for w in W:
                for t in T:
                    model.addConstr(
                        g_dis[s, y, w, t] >= batteries.loc[(y, s), "P_discharge_min"],
                        name=f"C_batt_discharge_old_min[{s},{y},{w},{t}]",
                    )
                    model.addConstr(
                        g_dis[s, y, w, t] <= batteries.loc[(y, s), "P_discharge_max"],
                        name=f"C_batt_discharge_old_max[{s},{y},{w},{t}]",
                    )

    # 7b. Battery discharging limits, new batteries
    for s in S_new:
        for y in Y:
            for w in W:
                for t in T:
                    model.addConstr(
                        g_dis[s, y, w, t]
                        <= soc_s_cum_max[s, y] / batteries.loc[(y, s), "hour_capacity"],
                        name=f"C_batt_discharge_new_max[{s},{y},{w},{t}]",
                    )

    # 8. State of charge limits
    for s in S:
        for y in Y:
            for w in W:
                for t in T:
                    model.addConstr(
                        soc[s, y, w, t]
                        >= batteries.loc[(y, s), "SOC_min"] * soc_s_max[s, y],
                        name=f"C_soc_min[{s},{y},{w},{t}]",
                    )
                    model.addConstr(
                        soc[s, y, w, t]
                        <= batteries.loc[(y, s), "SOC_max"] * soc_s_max[s, y],
                        name=f"C_soc_max[{s},{y},{w},{t}]",
                    )

    # 9. Battery state of charge dynamics
    for s in S:
        for y in Y:
            for w in W:
                for t in T[1:]:  # Exclude time t=0
                    model.addConstr(
                        soc[s, y, w, t]
                        == soc[s, y, w, t - 1]
                        + batteries.loc[(y, s), "eta_charge"] * g_ch[s, y, w, t]
                        - g_dis[s, y, w, t] / batteries.loc[(y, s), "eta_discharge"],
                        name=f"C_soc_dynamics[{s},{y},{w},{t}]",
                    )

    # 10. Initial and final SOC
    for s in S:
        for y in Y:
            for w in W:
                model.addConstr(
                    soc[s, y, w, T[0]]
                    == batteries.loc[(y, s), "SOC_min"] * soc_s_max[s, y],
                    name=f"C_soc_init[{s},{y},{w}]",
                )
                model.addConstr(
                    soc[s, y, w, T[-1]]
                    == batteries.loc[(y, s), "SOC_min"] * soc_s_max[s, y],
                    name=f"C_soc_final[{s},{y},{w}]",
                )

    # Optimize the model
    model.setParam("MIPGap", MIPGap)
    model.setParam("Timelimit", TIMELIMIT)
    model.setParam("BarConvTol", MIPGap)
    model.setParam("BarHomogeneous", 1)
    # model.setParam("DualReductions", 1)

    build_end_time = time()

    print(f"Model built in {build_end_time - build_start_time} seconds.")
    model_optimize_start_time = time()
    model.optimize()
    model_optimize_end_time = time()
    # endregion

    # region Post-processing and saving results
    save_folder = config.get("save_folder", "")
    decision_variables_folder = os.path.join(save_folder, "decision_variables")
    os.makedirs(decision_variables_folder, exist_ok=True)
    dual_variables_folder = os.path.join(save_folder, "dual_variables")
    os.makedirs(dual_variables_folder, exist_ok=True)

    # Save generation
    generation_data = [
        (y, w, t, i, g[i, y, w, t].X + (g_new[i, y, w, t].X if i in G_new else 0))
        for i in G
        for y in Y
        for w in W
        for t in T
    ]

    generation_df = pd.DataFrame(
        generation_data, columns=["year", "week", "hour", "generator", "value"]
    )
    generation_df.to_csv(
        os.path.join(decision_variables_folder, "generation.csv"), index=False
    )

    # Save power flow
    power_flow_data = [
        (y, w, t, b, f[b, y, w, t].X + (f_new[b, y, w, t].X if b in B_new else 0))
        for b in B
        for y in Y
        for w in W
        for t in T
    ]

    power_flow_df = pd.DataFrame(
        power_flow_data, columns=["year", "week", "hour", "branch", "value"]
    )
    power_flow_df.to_csv(
        os.path.join(decision_variables_folder, "power_flow.csv"), index=False
    )

    # Save load shedding
    load_shedding_data = [
        (y, w, t, n, sh[n, y, w, t].X) for n in N for y in Y for w in W for t in T
    ]
    load_shedding_df = pd.DataFrame(
        load_shedding_data, columns=["year", "week", "hour", "node", "value"]
    )
    load_shedding_df.to_csv(
        os.path.join(decision_variables_folder, "load_shedding.csv"), index=False
    )

    # Save curtailment
    curtailment_data = [
        (y, w, t, i, c[i, y, w, t].X) for i in G for y in Y for w in W for t in T
    ]
    curtailment_df = pd.DataFrame(
        curtailment_data, columns=["year", "week", "hour", "generator", "value"]
    )
    curtailment_df.to_csv(
        os.path.join(decision_variables_folder, "curtailment.csv"), index=False
    )

    # Save battery charging
    battery_charging_data = [
        (y, w, t, s, g_ch[s, y, w, t].X) for s in S for y in Y for w in W for t in T
    ]
    battery_charging_df = pd.DataFrame(
        battery_charging_data, columns=["year", "week", "hour", "battery", "value"]
    )
    battery_charging_df.to_csv(
        os.path.join(decision_variables_folder, "battery_charging.csv"), index=False
    )

    # Save battery discharging
    battery_discharging_data = [
        (y, w, t, s, g_dis[s, y, w, t].X) for s in S for y in Y for w in W for t in T
    ]
    battery_discharging_df = pd.DataFrame(
        battery_discharging_data, columns=["year", "week", "hour", "battery", "value"]
    )
    battery_discharging_df.to_csv(
        os.path.join(decision_variables_folder, "battery_discharging.csv"), index=False
    )

    # Save battery state of charge
    battery_soc_data = [
        (y, w, t, s, soc[s, y, w, t].X) for s in S for y in Y for w in W for t in T
    ]
    battery_soc_df = pd.DataFrame(
        battery_soc_data, columns=["year", "week", "hour", "battery", "value"]
    )
    battery_soc_df.to_csv(
        os.path.join(decision_variables_folder, "battery_soc.csv"), index=False
    )

    # Save battery build
    battery_capacity_data = [(y, s, soc_s_max[s, y].X) for s in S_new for y in Y]
    battery_capacity_df = pd.DataFrame(
        battery_capacity_data, columns=["year", "battery", "value"]
    )
    battery_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "battery_capacity.csv"), index=False
    )

    # Save generator capacities
    generator_capacity_data = [(y, i, p_i_max[i, y].X) for i in G_new for y in Y]
    generator_capacity_df = pd.DataFrame(
        generator_capacity_data, columns=["year", "generator", "value"]
    )
    generator_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "generator_capacity.csv"), index=False
    )

    # Save branch capacities
    branch_capacity_data = [(y, b, p_b_max[b, y].X) for b in B_new for y in Y]
    branch_capacity_df = pd.DataFrame(
        branch_capacity_data, columns=["year", "branch", "value"]
    )
    branch_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "branch_capacity.csv"), index=False
    )

    # Duals
    # Save dual variables for constraints
    # Save dual variables
    # 1. Power balance duals (LMPs / shadow prices)
    power_balance_duals = [
        (
            y,
            w,
            t,
            n,
            model.getConstrByName(f"C_power_balance[{n},{y},{w},{t}]").Pi
            / week_weights[w],
        )
        for n in N
        for y in Y
        for w in W
        for t in T
    ]
    power_balance_duals_df = pd.DataFrame(
        power_balance_duals, columns=["year", "week", "hour", "node", "dual_value"]
    )
    power_balance_duals_df.to_csv(
        os.path.join(dual_variables_folder, "power_balance_duals.csv"), index=False
    )

    # 2. Load shedding limit duals
    load_shedding_duals = [
        (
            y,
            w,
            t,
            n,
            model.getConstrByName(f"C_load_shedding_limit[{n},{y},{w},{t}]").Pi
            / week_weights[w],
        )
        for n in N
        for y in Y
        for w in W
        for t in T
    ]
    load_shedding_duals_df = pd.DataFrame(
        load_shedding_duals, columns=["year", "week", "hour", "node", "dual_value"]
    )
    load_shedding_duals_df.to_csv(
        os.path.join(dual_variables_folder, "load_shedding_duals.csv"), index=False
    )

    # 3. Generator output limits (old)
    gen_old_output_duals = [
        (
            y,
            w,
            t,
            i,
            model.getConstrByName(f"C_gen_output_old[{i},{y},{w},{t}]").Pi
            / week_weights[w],
        )
        for i in G_old
        for y in Y
        for w in W
        for t in T
    ]
    gen_old_output_duals_df = pd.DataFrame(
        gen_old_output_duals,
        columns=["year", "week", "hour", "generator", "dual_value"],
    )
    gen_old_output_duals_df.to_csv(
        os.path.join(dual_variables_folder, "gen_output_old_duals.csv"), index=False
    )

    # 4. Generator output limits (new)
    gen_new_output_duals = [
        (
            y,
            w,
            t,
            i,
            model.getConstrByName(f"C_gen_output_new[{i},{y},{w},{t}]").Pi
            / week_weights[w],
        )
        for i in G_new
        for y in Y
        for w in W
        for t in T
    ]
    gen_new_output_duals_df = pd.DataFrame(
        gen_new_output_duals,
        columns=["year", "week", "hour", "generator", "dual_value"],
    )
    gen_new_output_duals_df.to_csv(
        os.path.join(dual_variables_folder, "gen_output_new_duals.csv"), index=False
    )

    # 6. Branch flow limit duals (old)
    branch_old_duals = []
    for b in B_old:
        for y in Y:
            for w in W:
                for t in T:
                    dual_min = (
                        model.getConstrByName(f"C_branch_old_min[{b},{y},{w},{t}]").Pi
                        / week_weights[w]
                    )
                    dual_max = (
                        model.getConstrByName(f"C_branch_old_max[{b},{y},{w},{t}]").Pi
                        / week_weights[w]
                    )
                    branch_old_duals.append((y, w, t, b, "min", dual_min))
                    branch_old_duals.append((y, w, t, b, "max", dual_max))

    branch_old_duals_df = pd.DataFrame(
        branch_old_duals,
        columns=["year", "week", "hour", "branch", "bound", "dual_value"],
    )
    branch_old_duals_df.to_csv(
        os.path.join(dual_variables_folder, "branch_flow_old_duals.csv"), index=False
    )

    # 7. Branch flow limit duals (new)
    branch_new_duals = []
    for b in B_new:
        for y in Y:
            for w in W:
                for t in T:
                    dual_min = (
                        model.getConstrByName(f"C_branch_new_min[{b},{y},{w},{t}]").Pi
                        / week_weights[w]
                    )
                    dual_max = (
                        model.getConstrByName(f"C_branch_new_max[{b},{y},{w},{t}]").Pi
                        / week_weights[w]
                    )
                    branch_new_duals.append((y, w, t, b, "min", dual_min))
                    branch_new_duals.append((y, w, t, b, "max", dual_max))

    branch_new_duals_df = pd.DataFrame(
        branch_new_duals,
        columns=["year", "week", "hour", "branch", "bound", "dual_value"],
    )
    branch_new_duals_df.to_csv(
        os.path.join(dual_variables_folder, "branch_flow_new_duals.csv"), index=False
    )

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

    # 6a. Battery charging limits (old)
    for bound in ["min", "max"]:
        charging_duals = [
            (
                y,
                w,
                t,
                s,
                model.getConstrByName(f"C_batt_charge_old_{bound}[{s},{y},{w},{t}]").Pi
                / week_weights[w],
            )
            for s in S_old
            for y in Y
            for w in W
            for t in T
        ]
        df = pd.DataFrame(
            charging_duals, columns=["year", "week", "hour", "battery", "dual_value"]
        )
        df.to_csv(
            os.path.join(
                dual_variables_folder, f"battery_charge_old_{bound}_duals.csv"
            ),
            index=False,
        )

    # 6b. Battery charging limits (new)
    charging_new_duals = [
        (
            y,
            w,
            t,
            s,
            model.getConstrByName(f"C_batt_charge_new_max[{s},{y},{w},{t}]").Pi
            / week_weights[w],
        )
        for s in S_new
        for y in Y
        for w in W
        for t in T
    ]
    pd.DataFrame(
        charging_new_duals, columns=["year", "week", "hour", "battery", "dual_value"]
    ).to_csv(
        os.path.join(dual_variables_folder, "battery_charge_new_max_duals.csv"),
        index=False,
    )

    # 7a. Battery discharging limits (old)
    for bound in ["min", "max"]:
        discharging_duals = [
            (
                y,
                w,
                t,
                s,
                model.getConstrByName(
                    f"C_batt_discharge_old_{bound}[{s},{y},{w},{t}]"
                ).Pi
                / week_weights[w],
            )
            for s in S_old
            for y in Y
            for w in W
            for t in T
        ]
        df = pd.DataFrame(
            discharging_duals, columns=["year", "week", "hour", "battery", "dual_value"]
        )
        df.to_csv(
            os.path.join(
                dual_variables_folder, f"battery_discharge_old_{bound}_duals.csv"
            ),
            index=False,
        )

    # 7b. Battery discharging limits (new)
    discharging_new_duals = [
        (
            y,
            w,
            t,
            s,
            model.getConstrByName(f"C_batt_discharge_new_max[{s},{y},{w},{t}]").Pi
            / week_weights[w],
        )
        for s in S_new
        for y in Y
        for w in W
        for t in T
    ]
    pd.DataFrame(
        discharging_new_duals, columns=["year", "week", "hour", "battery", "dual_value"]
    ).to_csv(
        os.path.join(dual_variables_folder, "battery_discharge_new_max_duals.csv"),
        index=False,
    )

    # 3c. Generator extension limits
    gen_ext_duals = [
        (y, i, model.getConstrByName(f"C_gen_extension_limit[{i},{y}]").Pi)
        for i in G_new
        for y in Y
    ]
    pd.DataFrame(gen_ext_duals, columns=["year", "generator", "dual_value"]).to_csv(
        os.path.join(dual_variables_folder, "gen_extension_duals.csv"), index=False
    )

    # 4c. Branch extension limits
    branch_ext_duals = [
        (y, b, model.getConstrByName(f"C_branch_extension_limit[{b},{y}]").Pi)
        for b in B_new
        for y in Y
    ]
    pd.DataFrame(branch_ext_duals, columns=["year", "branch", "dual_value"]).to_csv(
        os.path.join(dual_variables_folder, "branch_extension_duals.csv"), index=False
    )

    # endregion

    return (
        model,
        build_end_time - build_start_time,
        model_optimize_end_time - model_optimize_start_time,
        week_weights,
    )


def GTSEP_stochastic_v1(config: dict) -> gp.Model:
    """GTSEP_v1 model but with multiple time periods considered. Uses clustering"""
    # region Model setup and running
    must_have_keys = [
        "data_folder_name",
        "VOLL",
        "CC",
        "CO2_price",
        "E_limit",
        "p_max_new_branch",
        "p_min_new_branch",
        "expansion_factor",
        "MS",
        "model_name",
        "MIPGap",
        "years",
        "discount_rate",
        "representative_period_unit",
        "representative_periods",
        "scenario_file",
    ]
    for key in must_have_keys:
        if key not in config:
            raise KeyError(
                f"Required key '{key}' not found in config. \nRequired keys: {must_have_keys}\nConfig keys: {config.keys()}"
            )

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
    if not representative_period_unit == "week":
        raise ValueError(
            "Only 'week' is supported as representative_period_unit as of now."
        )

    if not years:
        raise ValueError("Years must be provided for a multi-year model.")

    # Load data
    data_folder_path = os.path.join(PROCESSED_DATA_FOLDER, data_folder_name)
    input_data = load_multi_year_csv_files_with_week_from_folder(
        years, data_folder_path
    )
    batteries = input_data["batteries"]
    branches = input_data["branches"]
    capacity_factors = input_data["capacity_factors"]
    generators = input_data["generators"]
    generator_costs = input_data["generator_costs"]
    nodes = input_data["nodes"]

    hourly_demand = input_data["hourly_demand"]
    scenario_multiplier = load_scenario_multiplier(scenario_file)
    # Check that years in scenario_multiplier match the years in the data
    scenario_years = scenario_multiplier.index.values.tolist()
    assert set(years) == set(
        scenario_years
    ), f"Years in scenario_multiplier {scenario_years} do not match the years in the data {years}."
    scenarios_list = scenario_multiplier.columns.tolist()
    scenarios = {
        year: [name for name in scenario_multiplier.loc[year].dropna().index]
        for year in scenario_multiplier.index
    }

    # Your work begins here

    # ----- 1. Scenario sets -----
    Omega = scenarios_list  # e.g. ['NT','GA','DE']
    Omega_y = scenarios  # e.g. {2040:['NT','GA','DE'],...}

    # ----- 2. Extension potential (unchanged) -----
    branches["extension_potential"] = p_max_new_branch * branches["extendable"]
    generators["extension_potential"] = (
        generators["p_nom"] * generators["extendable"] * expansion_factor
    )

    # ----- 3. Static sets -----
    N = nodes.index.to_list()
    G_new = (
        generators[generators["extendable"] == True]
        .index.get_level_values("generator")
        .unique()
        .tolist()
    )
    G = generators.index.get_level_values("generator").unique().tolist()
    G_old = G

    B_new = (
        branches[branches["extendable"] == True]
        .index.get_level_values("line")
        .unique()
        .tolist()
    )
    B = branches.index.get_level_values("line").unique().tolist()
    B_old = B

    S = batteries.index.get_level_values("battery").unique().tolist()
    S_new = S  # all new
    S_old = []  # none old

    Y = scenario_years
    W = weeks
    # Derive T as the hours for the first representative week…
    first_week = W[0]
    T = (
        hourly_demand.xs(first_week, level="week")
        .index.get_level_values("hour")
        .unique()
        .tolist()
    )

    # …then confirm every week in W has exactly the same hours
    for w in W:
        hours_w = (
            hourly_demand.xs(w, level="week").index.get_level_values("hour").unique()
        )
        if set(hours_w) != set(T):
            raise ValueError(
                f"Week {w} has hours {sorted(hours_w)}, expected {sorted(T)}"
            )

    # (Optional) Sort T so it’s in ascending order
    T = sorted(T)

    week_weights = {w: 1 / len(W) * 8760 / len(T) for w in W}
    (
        branches_out_of_node,
        branches_into_node,
        batteries_at_node,
        generators_at_node,
    ) = _create_mappings(nodes, branches, generators, batteries)

    # create Yy mapping, accessed as Yy[y] and returns a list of years up to and including y from Y
    Yy = {y: [yy for yy in Y if yy <= y] for y in Y}

    build_start_time = time()
    # Create model
    model_name = model_name if model_name else "GTSEP_v0"
    model = gp.Model(model_name)

    # --- Decision variables (ω before y,w,t) ---
    g = model.addVars(G, Omega, Y, W, T, name="g", lb=0)
    g_new = model.addVars(G_new, Omega, Y, W, T, name="g_new", lb=0)

    f = model.addVars(B, Omega, Y, W, T, name="f", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    f_new = model.addVars(
        B_new, Omega, Y, W, T, name="f_new", lb=-GRB.INFINITY, ub=GRB.INFINITY
    )

    sh = model.addVars(N, Omega, Y, W, T, name="sh", lb=0)
    c = model.addVars(G, Omega, Y, W, T, name="c", lb=0)

    g_ch = model.addVars(S, Omega, Y, W, T, name="g_ch", lb=0)
    g_dis = model.addVars(S, Omega, Y, W, T, name="g_dis", lb=0)
    soc = model.addVars(S, Omega, Y, W, T, name="soc", lb=0)

    # investment‐only vars (no ω, w, t)
    p_i_max = model.addVars(G_new, Y, name="p_i_max", lb=0)
    p_b_max = model.addVars(B_new, Y, name="p_b_max", lb=0)
    soc_s_max = model.addVars(S_new, Y, name="soc_s_max", lb=0)

    # Cumulative capacity helper variables (unchanged)
    p_i_cum_max = model.addVars(G_new, Y, name="p_i_cum_max", lb=0)
    for i in G_new:
        for y in Y:
            model.addConstr(
                p_i_cum_max[i, y]
                == gp.quicksum(p_i_max[i, y_marked] for y_marked in Yy[y])
            )

    p_b_cum_max = model.addVars(B_new, Y, name="p_b_cum_max", lb=0)
    for b in B_new:
        for y in Y:
            model.addConstr(
                p_b_cum_max[b, y]
                == gp.quicksum(p_b_max[b, y_marked] for y_marked in Yy[y])
            )

    soc_s_cum_max = model.addVars(S_new, Y, name="soc_s_cum_max", lb=0)
    for s in S_new:
        for y in Y:
            model.addConstr(
                soc_s_cum_max[s, y]
                == gp.quicksum(soc_s_max[s, y_marked] for y_marked in Yy[y])
            )

    # --- Objective
    obj = 0.0
    for y in Y:
        # annual investment costs
        obj += gp.quicksum(
            generators.loc[(y, i), "capital_cost"] * p_i_max[i, y] for i in G_new
        )
        obj += gp.quicksum(
            branches.loc[(y, b), "capital_cost"] * p_b_max[b, y] for b in B_new
        )
        obj += gp.quicksum(
            batteries.loc[(y, s), "capital_cost"] * soc_s_max[s, y] for s in S_new
        )

        # ops cost per scenario
        for ω in Omega_y[y]:
            p_scen = 1.0 / len(Omega_y[y])
            obj += p_scen * gp.quicksum(
                week_weights[w]
                * (
                    gp.quicksum(
                        (
                            generators.loc[(y, i), "marginal_cost"]
                            + generators.loc[(y, i), "co2_emissions"] * CO2_price
                        )
                        * (
                            g[i, ω, y, w, t]
                            + (g_new[i, ω, y, w, t] if i in G_new else 0)
                        )
                        for i in G
                        for t in T
                    )
                    + gp.quicksum(VOLL * sh[n, ω, y, w, t] for n in N for t in T)
                    + gp.quicksum(CC * c[i, ω, y, w, t] for i in G for t in T)
                )
                for w in W
            )

    model.setObjective(obj, GRB.MINIMIZE)

    # --- Time‐staged constraints (ω before y,w,t) ---
    # 1) Power balance
    for n in N:
        for y in Y:
            for ω in Omega_y[y]:
                for w in W:
                    for t in T:
                        expr = (
                            gp.quicksum(
                                g[i, ω, y, w, t]
                                - c[i, ω, y, w, t]
                                + (g_new[i, ω, y, w, t] if i in G_new else 0)
                                for i in generators_at_node[n]
                            )
                            + gp.quicksum(
                                (
                                    f[b, ω, y, w, t]
                                    + (f_new[b, ω, y, w, t] if b in B_new else 0)
                                )
                                * (1 - branches.loc[(y, b), "loss_factor"])
                                for b in branches_into_node[n]
                            )
                            - gp.quicksum(
                                f[b, ω, y, w, t]
                                + (f_new[b, ω, y, w, t] if b in B_new else 0)
                                for b in branches_out_of_node[n]
                            )
                            - gp.quicksum(
                                g_ch[s, ω, y, w, t]
                                - batteries.loc[(y, s), "eta_discharge"]
                                * g_dis[s, ω, y, w, t]
                                for s in batteries_at_node[n]
                            )
                            + sh[n, ω, y, w, t]
                        )
                        rhs = (
                            hourly_demand.loc[(w, t), n] * scenario_multiplier.loc[y, ω]
                        )
                        model.addConstr(
                            expr == rhs, name=f"C_power_balance[{n},{ω},{y},{w},{t}]"
                        )

    # 2) Load‐shedding limit
    for n in N:
        for y in Y:
            for ω in Omega_y[y]:
                for w in W:
                    for t in T:
                        model.addConstr(
                            sh[n, ω, y, w, t]
                            <= MS
                            * hourly_demand.loc[(w, t), n]
                            * scenario_multiplier.loc[y, ω],
                            name=f"C_load_shedding_limit[{n},{ω},{y},{w},{t}]",
                        )

    # 3) Curtailment limit
    for i in G:
        for y in Y:
            for ω in Omega_y[y]:
                for w in W:
                    for t in T:
                        model.addConstr(
                            c[i, ω, y, w, t] <= g[i, ω, y, w, t],
                            name=f"C_curtailment_limit[{i},{ω},{y},{w},{t}]",
                        )

    # 4) Generator output limits (old)
    for i in G_old:
        for y in Y:
            p_max = generators.loc[(y, i), "p_nom"]
            for ω in Omega_y[y]:
                for w in W:
                    for t in T:
                        capacity_factor = capacity_factors.loc[(y, w, t), i]
                        model.addConstr(
                            g[i, ω, y, w, t] <= p_max * capacity_factor,
                            name=f"C_gen_output_old[{i},{ω},{y},{w},{t}]",
                        )

    # 5) Generator output limits (new)
    for i in G_new:
        for y in Y:
            for ω in Omega_y[y]:
                for w in W:
                    for t in T:
                        cap = capacity_factors.loc[(y, w, t), i] * p_i_cum_max[i, y]
                        model.addConstr(
                            g_new[i, ω, y, w, t] <= cap,
                            name=f"C_gen_output_new[{i},{ω},{y},{w},{t}]",
                        )

    # 6) Generator capacity‐extension limits
    for i in G_new:
        for y in Y:
            limit = generators.loc[(y, i), "extension_potential"]
            model.addConstr(
                p_i_max[i, y] <= limit, name=f"C_gen_extension_limit[{i},{y}]"
            )

    # 7) Branch flow limits (old)
    for b in B_old:
        for y in Y:
            for ω in Omega_y[y]:
                for w in W:
                    for t in T:
                        pmax = branches.loc[(y, b), "p_max"]
                        model.addConstr(
                            f[b, ω, y, w, t] <= pmax,
                            name=f"C_branch_old_max[{b},{ω},{y},{w},{t}]",
                        )
                        model.addConstr(
                            f[b, ω, y, w, t] >= -pmax,
                            name=f"C_branch_old_min[{b},{ω},{y},{w},{t}]",
                        )

    # 8) Branch flow upper limits (new)
    for b in B_new:
        for y in Y:
            for ω in Omega_y[y]:
                for w in W:
                    for t in T:
                        model.addConstr(
                            f_new[b, ω, y, w, t] <= p_b_cum_max[b, y],
                            name=f"C_branch_new_max[{b},{ω},{y},{w},{t}]",
                        )
                        model.addConstr(
                            f_new[b, ω, y, w, t] >= -p_b_cum_max[b, y],
                            name=f"C_branch_new_min[{b},{ω},{y},{w},{t}]",
                        )

    # 9) Branch capacity‐extension limits
    for b in B_new:
        for y in Y:
            limit = branches.loc[(y, b), "extension_potential"]
            model.addConstr(
                p_b_max[b, y] <= limit, name=f"C_branch_extension_limit[{b},{y}]"
            )

    # 10) Emission restriction (sum over ω)
    for ω in Omega:
        for y in Y:
            expr = gp.quicksum(
                week_weights[w]
                * (g[i, ω, y, w, t] + g_new[i, ω, y, w, t])
                * generators.loc[(y, i), "co2_emissions"]
                for i in G
                for w in W
                for t in T
            )
            model.addConstr(expr <= E_limit, name=f"C_emission_limit[{ω},{y}]")

    # 11a) Battery charging limits (old)
    for s in S_old:
        for y in Y:
            for ω in Omega_y[y]:
                for w in W:
                    for t in T:
                        pmin = batteries.loc[(y, s), "P_charge_min"]
                        pmax = batteries.loc[(y, s), "P_charge_max"]
                        model.addConstr(
                            g_ch[s, ω, y, w, t] >= pmin,
                            name=f"C_batt_charge_old_min[{s},{ω},{y},{w},{t}]",
                        )
                        model.addConstr(
                            g_ch[s, ω, y, w, t] <= pmax,
                            name=f"C_batt_charge_old_max[{s},{ω},{y},{w},{t}]",
                        )

    # 11b) Battery charging limits (new)
    for s in S_new:
        for y in Y:
            for ω in Omega_y[y]:
                for w in W:
                    for t in T:
                        cap = soc_s_cum_max[s, y] / (
                            batteries.loc[(y, s), "hour_capacity"]
                            * batteries.loc[(y, s), "cdrate"]
                        )
                        model.addConstr(
                            g_ch[s, ω, y, w, t] <= cap,
                            name=f"C_batt_charge_new_max[{s},{ω},{y},{w},{t}]",
                        )

    # 12a) Battery discharging limits (old)
    for s in S_old:
        for y in Y:
            for ω in Omega_y[y]:
                for w in W:
                    for t in T:
                        pmin = batteries.loc[(y, s), "P_discharge_min"]
                        pmax = batteries.loc[(y, s), "P_discharge_max"]
                        model.addConstr(
                            g_dis[s, ω, y, w, t] >= pmin,
                            name=f"C_batt_discharge_old_min[{s},{ω},{y},{w},{t}]",
                        )
                        model.addConstr(
                            g_dis[s, ω, y, w, t] <= pmax,
                            name=f"C_batt_discharge_old_max[{s},{ω},{y},{w},{t}]",
                        )

    # 12b) Battery discharging limits (new)
    for s in S_new:
        for y in Y:
            for ω in Omega_y[y]:
                for w in W:
                    for t in T:
                        cap = (
                            soc_s_cum_max[s, y] / batteries.loc[(y, s), "hour_capacity"]
                        )
                        model.addConstr(
                            g_dis[s, ω, y, w, t] <= cap,
                            name=f"C_batt_discharge_new_max[{s},{ω},{y},{w},{t}]",
                        )

    # 13) State‐of‐charge limits
    for s in S:
        for y in Y:
            for ω in Omega_y[y]:
                for w in W:
                    for t in T:
                        soc_min = batteries.loc[(y, s), "SOC_min"] * soc_s_max[s, y]
                        soc_max = batteries.loc[(y, s), "SOC_max"] * soc_s_max[s, y]
                        model.addConstr(
                            soc[s, ω, y, w, t] >= soc_min,
                            name=f"C_soc_min[{s},{ω},{y},{w},{t}]",
                        )
                        model.addConstr(
                            soc[s, ω, y, w, t] <= soc_max,
                            name=f"C_soc_max[{s},{ω},{y},{w},{t}]",
                        )

    # 14) SOC dynamics
    for s in S:
        for y in Y:
            for ω in Omega_y[y]:
                for w in W:
                    for t in T[1:]:
                        prev = soc[s, ω, y, w, t - 1]
                        charge = (
                            batteries.loc[(y, s), "eta_charge"] * g_ch[s, ω, y, w, t]
                        )
                        discharge = (
                            g_dis[s, ω, y, w, t]
                            / batteries.loc[(y, s), "eta_discharge"]
                        )
                        model.addConstr(
                            soc[s, ω, y, w, t] == prev + charge - discharge,
                            name=f"C_soc_dynamics[{s},{ω},{y},{w},{t}]",
                        )

    # 15) SOC initial and final (return to SOC_min each week)
    for s in S:
        for y in Y:
            for ω in Omega_y[y]:
                for w in W:
                    soc0 = batteries.loc[(y, s), "SOC_min"] * soc_s_max[s, y]
                    model.addConstr(
                        soc[s, ω, y, w, T[0]] == soc0,
                        name=f"C_soc_init[{s},{ω},{y},{w}]",
                    )
                    model.addConstr(
                        soc[s, ω, y, w, T[-1]] == soc0,
                        name=f"C_soc_final[{s},{ω},{y},{w}]",
                    )

        # Optimize the model
    model.setParam("MIPGap", MIPGap)
    model.setParam("Timelimit", TIMELIMIT)
    model.setParam("BarConvTol", MIPGap)
    model.setParam("BarHomogeneous", 1)
    build_end_time = time()
    print(f"Model built in {build_end_time - build_start_time} seconds.")
    model_optimize_start_time = time()
    model.optimize()
    model_optimize_end_time = time()
    # endregion

    # region Post-processing and saving results
    save_folder = config.get("save_folder", "")
    decision_variables_folder = os.path.join(save_folder, "decision_variables")
    os.makedirs(decision_variables_folder, exist_ok=True)
    dual_variables_folder = os.path.join(save_folder, "dual_variables")
    os.makedirs(dual_variables_folder, exist_ok=True)

    model_info_folder = os.path.join(save_folder, "model_info")
    os.makedirs(model_info_folder, exist_ok=True)
    import json

    with open(os.path.join(model_info_folder, "scenarios.json"), "w") as file:
        json.dump(scenarios, file, indent=4)
    scenario_probabilities = {
        year: [1 / len(scenarios[year]) for _ in scenarios[year]] for year in years
    }
    with open(
        os.path.join(model_info_folder, "scenario_probabilities.json"), "w"
    ) as file:
        json.dump(scenario_probabilities, file, indent=4)
    with open(os.path.join(model_info_folder, "week_weights.json"), "w") as file:
        json.dump(week_weights, file, indent=4)

    # --- Save generation ---
    generation_data = [
        (
            omega,
            y,
            w,
            t,
            i,
            g[i, omega, y, w, t].X + (g_new[i, omega, y, w, t].X if i in G_new else 0),
        )
        for i in G
        for y in Y
        for omega in Omega_y[y]
        for w in W
        for t in T
    ]
    pd.DataFrame(
        generation_data,
        columns=["scenario", "year", "week", "hour", "generator", "value"],
    ).to_csv(os.path.join(decision_variables_folder, "generation.csv"), index=False)

    # --- Save power flow ---
    power_flow_data = [
        (
            omega,
            y,
            w,
            t,
            b,
            f[b, omega, y, w, t].X + (f_new[b, omega, y, w, t].X if b in B_new else 0),
        )
        for b in B
        for y in Y
        for omega in Omega_y[y]
        for w in W
        for t in T
    ]
    pd.DataFrame(
        power_flow_data, columns=["scenario", "year", "week", "hour", "branch", "value"]
    ).to_csv(os.path.join(decision_variables_folder, "power_flow.csv"), index=False)

    # --- Save load shedding ---
    load_shedding_data = [
        (omega, y, w, t, n, sh[n, omega, y, w, t].X)
        for n in N
        for y in Y
        for omega in Omega_y[y]
        for w in W
        for t in T
    ]
    pd.DataFrame(
        load_shedding_data,
        columns=["scenario", "year", "week", "hour", "node", "value"],
    ).to_csv(os.path.join(decision_variables_folder, "load_shedding.csv"), index=False)

    # --- Save curtailment ---
    curtailment_data = [
        (omega, y, w, t, i, c[i, omega, y, w, t].X)
        for i in G
        for y in Y
        for omega in Omega_y[y]
        for w in W
        for t in T
    ]
    pd.DataFrame(
        curtailment_data,
        columns=["scenario", "year", "week", "hour", "generator", "value"],
    ).to_csv(os.path.join(decision_variables_folder, "curtailment.csv"), index=False)

    # --- Save battery charging ---
    battery_charging_data = [
        (omega, y, w, t, s, g_ch[s, omega, y, w, t].X)
        for s in S
        for y in Y
        for omega in Omega_y[y]
        for w in W
        for t in T
    ]
    pd.DataFrame(
        battery_charging_data,
        columns=["scenario", "year", "week", "hour", "battery", "value"],
    ).to_csv(
        os.path.join(decision_variables_folder, "battery_charging.csv"), index=False
    )

    # --- Save battery discharging ---
    battery_discharging_data = [
        (omega, y, w, t, s, g_dis[s, omega, y, w, t].X)
        for s in S
        for y in Y
        for omega in Omega_y[y]
        for w in W
        for t in T
    ]
    pd.DataFrame(
        battery_discharging_data,
        columns=["scenario", "year", "week", "hour", "battery", "value"],
    ).to_csv(
        os.path.join(decision_variables_folder, "battery_discharging.csv"), index=False
    )

    # --- Save battery state of charge ---
    battery_soc_data = [
        (omega, y, w, t, s, soc[s, omega, y, w, t].X)
        for s in S
        for y in Y
        for omega in Omega_y[y]
        for w in W
        for t in T
    ]
    pd.DataFrame(
        battery_soc_data,
        columns=["scenario", "year", "week", "hour", "battery", "value"],
    ).to_csv(os.path.join(decision_variables_folder, "battery_soc.csv"), index=False)

    # --- Save investment capacities (no scenario) ---
    pd.DataFrame(
        [(y, s, soc_s_max[s, y].X) for s in S_new for y in Y],
        columns=["year", "battery", "value"],
    ).to_csv(
        os.path.join(decision_variables_folder, "battery_capacity.csv"), index=False
    )
    pd.DataFrame(
        [(y, i, p_i_max[i, y].X) for i in G_new for y in Y],
        columns=["year", "generator", "value"],
    ).to_csv(
        os.path.join(decision_variables_folder, "generator_capacity.csv"), index=False
    )
    pd.DataFrame(
        [(y, b, p_b_max[b, y].X) for b in B_new for y in Y],
        columns=["year", "branch", "value"],
    ).to_csv(
        os.path.join(decision_variables_folder, "branch_capacity.csv"), index=False
    )

    # --- Duals: Power balance ---
    power_balance_duals = [
        (
            omega,
            y,
            w,
            t,
            n,
            model.getConstrByName(f"C_power_balance[{n},{omega},{y},{w},{t}]").Pi
            / week_weights[w],
        )
        for n in N
        for y in Y
        for omega in Omega_y[y]
        for w in W
        for t in T
    ]
    pd.DataFrame(
        power_balance_duals,
        columns=["scenario", "year", "week", "hour", "node", "dual_value"],
    ).to_csv(
        os.path.join(dual_variables_folder, "power_balance_duals.csv"), index=False
    )

    # --- Duals: Load shedding limit ---
    load_shedding_duals = [
        (
            omega,
            y,
            w,
            t,
            n,
            model.getConstrByName(f"C_load_shedding_limit[{n},{omega},{y},{w},{t}]").Pi
            / week_weights[w],
        )
        for n in N
        for y in Y
        for omega in Omega_y[y]
        for w in W
        for t in T
    ]
    pd.DataFrame(
        load_shedding_duals,
        columns=["scenario", "year", "week", "hour", "node", "dual_value"],
    ).to_csv(
        os.path.join(dual_variables_folder, "load_shedding_duals.csv"), index=False
    )

    # --- Duals: Generator output limits (old) ---
    gen_old_output_duals = [
        (
            omega,
            y,
            w,
            t,
            i,
            model.getConstrByName(f"C_gen_output_old[{i},{omega},{y},{w},{t}]").Pi
            / week_weights[w],
        )
        for i in G_old
        for y in Y
        for omega in Omega_y[y]
        for w in W
        for t in T
    ]
    pd.DataFrame(
        gen_old_output_duals,
        columns=["scenario", "year", "week", "hour", "generator", "dual_value"],
    ).to_csv(
        os.path.join(dual_variables_folder, "gen_output_old_duals.csv"), index=False
    )

    # --- Duals: Generator output limits (new) ---
    gen_new_output_duals = [
        (
            omega,
            y,
            w,
            t,
            i,
            model.getConstrByName(f"C_gen_output_new[{i},{omega},{y},{w},{t}]").Pi
            / week_weights[w],
        )
        for i in G_new
        for y in Y
        for omega in Omega_y[y]
        for w in W
        for t in T
    ]
    pd.DataFrame(
        gen_new_output_duals,
        columns=["scenario", "year", "week", "hour", "generator", "dual_value"],
    ).to_csv(
        os.path.join(dual_variables_folder, "gen_output_new_duals.csv"), index=False
    )

    # --- Duals: Branch flow limits (old) ---
    branch_old_duals = []
    for b in B_old:
        for y in Y:
            for omega in Omega_y[y]:
                for w in W:
                    for t in T:
                        dual_min = (
                            model.getConstrByName(
                                f"C_branch_old_min[{b},{omega},{y},{w},{t}]"
                            ).Pi
                            / week_weights[w]
                        )
                        dual_max = (
                            model.getConstrByName(
                                f"C_branch_old_max[{b},{omega},{y},{w},{t}]"
                            ).Pi
                            / week_weights[w]
                        )
                        branch_old_duals.append((omega, y, w, t, b, "min", dual_min))
                        branch_old_duals.append((omega, y, w, t, b, "max", dual_max))
    pd.DataFrame(
        branch_old_duals,
        columns=["scenario", "year", "week", "hour", "branch", "bound", "dual_value"],
    ).to_csv(
        os.path.join(dual_variables_folder, "branch_flow_old_duals.csv"), index=False
    )

    # --- Duals: Branch flow limits (new) ---
    branch_new_duals = []
    for b in B_new:
        for y in Y:
            for omega in Omega_y[y]:
                for w in W:
                    for t in T:
                        dual_min = (
                            model.getConstrByName(
                                f"C_branch_new_min[{b},{omega},{y},{w},{t}]"
                            ).Pi
                            / week_weights[w]
                        )
                        dual_max = (
                            model.getConstrByName(
                                f"C_branch_new_max[{b},{omega},{y},{w},{t}]"
                            ).Pi
                            / week_weights[w]
                        )
                        branch_new_duals.append((omega, y, w, t, b, "min", dual_min))
                        branch_new_duals.append((omega, y, w, t, b, "max", dual_max))
    pd.DataFrame(
        branch_new_duals,
        columns=["scenario", "year", "week", "hour", "branch", "bound", "dual_value"],
    ).to_csv(
        os.path.join(dual_variables_folder, "branch_flow_new_duals.csv"), index=False
    )

    # --- Duals: Emissions constraint (single value) ---
    emissions_duals = []
    for y in Y:
        for omega in Omega_y[y]:
            dual = model.getConstrByName(f"C_emission_limit[{omega},{y}]").Pi
            emissions_duals.append((omega, y, dual))
    pd.DataFrame(
        emissions_duals,
        columns=["scenario", "year", "dual_value"],
    ).to_csv(os.path.join(dual_variables_folder, "emissions_duals.csv"), index=False)

    # --- Duals: Battery charging (old) ---
    batt_charge_old_duals = []
    for s in S_old:
        for y in Y:
            for omega in Omega_y[y]:
                for w in W:
                    for t in T:
                        dual_min = (
                            model.getConstrByName(
                                f"C_batt_charge_old_min[{s},{omega},{y},{w},{t}]"
                            ).Pi
                            / week_weights[w]
                        )
                        dual_max = (
                            model.getConstrByName(
                                f"C_batt_charge_old_max[{s},{omega},{y},{w},{t}]"
                            ).Pi
                            / week_weights[w]
                        )
                        batt_charge_old_duals.append(
                            (omega, y, w, t, s, "min", dual_min)
                        )
                        batt_charge_old_duals.append(
                            (omega, y, w, t, s, "max", dual_max)
                        )
    pd.DataFrame(
        batt_charge_old_duals,
        columns=["scenario", "year", "week", "hour", "battery", "bound", "dual_value"],
    ).to_csv(
        os.path.join(dual_variables_folder, "battery_charge_old_duals.csv"), index=False
    )

    # --- Duals: Battery charging (new) ---
    batt_charge_new_duals = [
        (
            omega,
            y,
            w,
            t,
            s,
            model.getConstrByName(f"C_batt_charge_new_max[{s},{omega},{y},{w},{t}]").Pi
            / week_weights[w],
        )
        for s in S_new
        for y in Y
        for omega in Omega_y[y]
        for w in W
        for t in T
    ]
    pd.DataFrame(
        batt_charge_new_duals,
        columns=["scenario", "year", "week", "hour", "battery", "dual_value"],
    ).to_csv(
        os.path.join(dual_variables_folder, "battery_charge_new_max_duals.csv"),
        index=False,
    )

    # --- Duals: Battery discharging (old) ---
    batt_discharge_old_duals = []
    for s in S_old:
        for y in Y:
            for omega in Omega_y[y]:
                for w in W:
                    for t in T:
                        dual_min = (
                            model.getConstrByName(
                                f"C_batt_discharge_old_min[{s},{omega},{y},{w},{t}]"
                            ).Pi
                            / week_weights[w]
                        )
                        dual_max = (
                            model.getConstrByName(
                                f"C_batt_discharge_old_max[{s},{omega},{y},{w},{t}]"
                            ).Pi
                            / week_weights[w]
                        )
                        batt_discharge_old_duals.append(
                            (omega, y, w, t, s, "min", dual_min)
                        )
                        batt_discharge_old_duals.append(
                            (omega, y, w, t, s, "max", dual_max)
                        )
    pd.DataFrame(
        batt_discharge_old_duals,
        columns=["scenario", "year", "week", "hour", "battery", "bound", "dual_value"],
    ).to_csv(
        os.path.join(dual_variables_folder, "battery_discharge_old_duals.csv"),
        index=False,
    )

    # --- Duals: Battery discharging (new) ---
    batt_discharge_new_duals = [
        (
            omega,
            y,
            w,
            t,
            s,
            model.getConstrByName(
                f"C_batt_discharge_new_max[{s},{omega},{y},{w},{t}]"
            ).Pi
            / week_weights[w],
        )
        for s in S_new
        for y in Y
        for omega in Omega_y[y]
        for w in W
        for t in T
    ]
    pd.DataFrame(
        batt_discharge_new_duals,
        columns=["scenario", "year", "week", "hour", "battery", "dual_value"],
    ).to_csv(
        os.path.join(dual_variables_folder, "battery_discharge_new_max_duals.csv"),
        index=False,
    )

    # --- Duals: Generator extension limits (no scenario) ---
    gen_ext_duals = [
        (y, i, model.getConstrByName(f"C_gen_extension_limit[{i},{y}]").Pi)
        for i in G_new
        for y in Y
    ]
    pd.DataFrame(gen_ext_duals, columns=["year", "generator", "dual_value"]).to_csv(
        os.path.join(dual_variables_folder, "gen_extension_duals.csv"), index=False
    )

    # --- Duals: Branch extension limits (no scenario) ---
    branch_ext_duals = [
        (y, b, model.getConstrByName(f"C_branch_extension_limit[{b},{y}]").Pi)
        for b in B_new
        for y in Y
    ]
    pd.DataFrame(branch_ext_duals, columns=["year", "branch", "dual_value"]).to_csv(
        os.path.join(dual_variables_folder, "branch_extension_duals.csv"), index=False
    )

    # endregion

    return (
        model,
        build_end_time - build_start_time,
        model_optimize_end_time - model_optimize_start_time,
        week_weights,
    )


# region GTSEP_v2
def GTSEP_v2(config: dict) -> gp.Model:
    """GTSEP model from the specialization project. Modeling battery investments as continous variables."""
    # region Model setup and running
    must_have_keys = [
        "data_folder_name",
        "VOLL",
        "CC",
        "CO2_price",
        "E_limit",
        "p_max_new_branch",
        "p_min_new_branch",
        "expansion_factor",
        "MS",
        "model_name",
        "MIPGap",
    ]
    for key in must_have_keys:
        if key not in config:
            raise KeyError(
                f"Required key '{key}' not found in config. \nRequired keys: {must_have_keys}\nConfig keys: {config.keys()}"
            )

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

    # Load data
    data_folder_path = os.path.join(PROCESSED_DATA_FOLDER, data_folder_name)
    input_data = load_csv_files_from_folder(data_folder_path)
    batteries = input_data["batteries"]
    branches = input_data["branches"]
    capacity_factors = input_data["capacity_factors"]
    generators = input_data["generators"]
    generator_costs = input_data["generator_costs"]
    hourly_demand = input_data["hourly_demand"]
    nodes = input_data["nodes"]

    # Data processing
    # Create new branches
    # Add a new column 'exists' to the original branches dataframe and set it to 1
    branches["exists"] = 1
    # Create a copy of the dataframe for the "new" branches
    branches_new = branches.copy()
    # Update the index by appending " new" to the original index
    branches_new.index = branches_new.index.astype(str) + " new"
    # Set the 'exists' column to 0 for the new branches
    branches_new["exists"] = 0
    # Concatenate the original dataframe and the new dataframe
    branches = pd.concat([branches, branches_new])
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
    batteries["exists"] = 0

    # Create sets
    N = nodes.index.to_list()
    G_old = generators[generators["exists"] == 1].index.to_list()
    G_new = generators[generators["exists"] == 0].index.to_list()
    G = generators.index.to_list()
    B_old = branches[branches["exists"] == 1].index.to_list()
    B_new = branches[branches["exists"] == 0].index.to_list()
    B = branches.index.to_list()
    S_new = batteries[batteries["exists"] == 0].index.to_list()
    S_old = batteries[batteries["exists"] == 1].index.to_list()
    S = batteries.index.to_list()
    time_indexes = pd.DataFrame({})
    time_indexes.index = hourly_demand.index
    time_indexes["week"] = hourly_demand.index.to_series().dt.isocalendar().week
    time_indexes["year"] = hourly_demand.index.to_series().dt.isocalendar().year
    time_indexes.loc[
        (time_indexes["week"] == 1) & (time_indexes["year"] == 2014), "week"
    ] = 53
    time_indexes
    T = hourly_demand.index.to_list()
    W = time_indexes["week"].unique().tolist()

    def week(t):
        return time_indexes.loc[t, "week"]

    # Create mappings
    (
        branches_out_of_node,
        branches_into_node,
        batteries_at_node,
        generators_at_node,
    ) = _create_mappings(nodes, branches, generators, batteries)

    build_start_time = time()
    # Create model
    model_name = model_name if model_name else "GTSEP_v0"
    model = gp.Model(model_name)

    # Decision variables
    g = model.addVars(G, T, name="g", lb=0)  # Power generation dispatch
    f = model.addVars(B, T, name="f", lb=-GRB.INFINITY, ub=GRB.INFINITY)  # Power flow
    sh = model.addVars(N, T, name="sh", lb=0)  # Load shedding
    c = model.addVars(G, T, name="c", lb=0)  # Curtailment
    g_ch = model.addVars(S, T, name="g_ch", lb=0)  # Battery charging
    g_dis = model.addVars(S, T, name="g_dis", lb=0)  # Battery discharging
    soc = model.addVars(S, T, name="soc", lb=0)  # State of charge
    x = model.addVars(G_new, vtype=GRB.BINARY, name="x")  # Binary for new generators
    y = model.addVars(B_new, vtype=GRB.BINARY, name="y")  # Binary for new branches
    soc_s_max = model.addVars(
        S_new, name="soc_s_max", lb=0
    )  # Max SOC for new batteries
    p_i_max = model.addVars(
        G_new, name="p_i_max", lb=0
    )  # Max capacity of new generators
    p_b_max = model.addVars(B_new, name="p_b_max", lb=0)  # Max capacity of new branches

    # v2 specific decision variable
    p_i_w = model.addVars(G, W, name="p_i_w", lb=0)  # Power generation in week w

    # Objective function: Minimize cost
    objective = (
        gp.quicksum(
            (
                generators.loc[i, "marginal_cost"]
                + generators.loc[i, "co2_emissions"] * CO2_price
            )
            * g[i, t]
            for i in G
            for t in T
        )
        # + gp.quicksum(
        #     batteries.loc[s, "MC"] * g_dis[s, t] * batteries.loc[s, "eta_discharge"]
        #     for s in S
        #     for t in T
        # )
        + gp.quicksum(VOLL * sh[n, t] for n in N for t in T)
        + gp.quicksum(CC * c[i, t] for i in G for t in T)
        + gp.quicksum(generators.loc[i, "capital_cost"] * p_i_max[i] for i in G_new)
        + gp.quicksum(branches.loc[b, "capital_cost"] * p_b_max[b] for b in B_new)
        + gp.quicksum(batteries.loc[s, "capital_cost"] * soc_s_max[s] for s in S_new)
    )
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
    # 1. Power balance
    for n in N:
        for t in T:
            model.addConstr(
                gp.quicksum(g[i, t] - c[i, t] for i in generators_at_node[n])
                + gp.quicksum(
                    f[b, t] * (1 - branches.loc[b, "loss_factor"])
                    for b in branches_into_node[n]
                )
                - gp.quicksum(f[b, t] for b in branches_out_of_node[n])
                - gp.quicksum(
                    g_ch[s, t] - batteries.loc[s, "eta_discharge"] * g_dis[s, t]
                    for s in batteries_at_node[n]
                )
                + sh[n, t]
                == hourly_demand.loc[t, n]
            )

    # 2a. Load shedding limits
    for n in N:
        for t in T:
            model.addConstr(sh[n, t] <= MS * hourly_demand.loc[t, n])

    # 2b. Curtailment limits
    for i in G:
        for t in T:
            model.addConstr(c[i, t] <= g[i, t])

    # 3a. Generator output limits (old generators)
    for i in G_old:
        p_max = generators.loc[i, "p_nom"]
        for w in W:
            model.addConstr(p_i_w[i, w] <= p_max)
            # lower bound is 0 by default
        for t in T:
            capacity_factor = capacity_factors.loc[t, i]
            model.addConstr(g[i, t] == p_i_w[i, week(t)] * capacity_factor)
            # Lower bound is 0 by default

    # 3b. Generator output limits (new generators)
    for i in G_new:
        for w in W:
            model.addConstr(p_i_w[i, w] <= x[i] * p_i_max[i])
            # Lower bound is 0 by default
        for t in T:
            original_generator_id = " ".join(i.split(" ")[:-1])
            capacity_factor = capacity_factors.loc[t, original_generator_id]
            model.addConstr(g[i, t] == p_i_w[i, w] * capacity_factor)
            # Lower bound is 0 by default

    # 3c. New generator capacity limits
    for i in G_new:
        p_max = generators.loc[i, "p_nom"]
        model.addConstr(p_i_max[i] <= expansion_factor * p_max)

    # 4a. Branch flow limits (old branches)
    for b in B_old:
        for t in T:
            model.addConstr(f[b, t] >= -branches.loc[b, "p_max"])
            model.addConstr(f[b, t] <= branches.loc[b, "p_max"])

    # 4b. Branch flow limits (new branches)
    for b in B_new:
        for t in T:
            model.addConstr(f[b, t] >= -y[b] * p_b_max[b])
            model.addConstr(f[b, t] <= y[b] * p_b_max[b])

    # 4c. New branch capacity limits
    for b in B_new:
        model.addConstr(p_b_max[b] >= y[b] * p_min_new_branch)
        model.addConstr(p_b_max[b] <= y[b] * p_max_new_branch)

    # # 5. Emission restrictions
    # model.addConstr(
    #     gp.quicksum(g[i, t] * generators.loc[i, "co2_emissions"] for i in G for t in T)
    #     <= E_limit
    # )

    # 6a. Battery charging limits, old batteries
    for s in S_old:
        for t in T:
            model.addConstr(g_ch[s, t] >= batteries.loc[s, "P_charge_min"])
            model.addConstr(g_ch[s, t] <= batteries.loc[s, "P_charge_max"])

    # 6b. Battery charging limits, new batteries
    for s in S_new:
        for t in T:
            # model.addConstr(g_ch[s, t] >= 0)
            model.addConstr(
                g_ch[s, t]
                <= soc_s_max[s]
                / (batteries.loc[s, "hour_capacity"] * batteries.loc[s, "cdrate"])
            )

    # 7a. Battery discharging limits, old batteries
    for s in S_old:
        for t in T:
            model.addConstr(g_dis[s, t] >= batteries.loc[s, "P_discharge_min"])
            model.addConstr(g_dis[s, t] <= batteries.loc[s, "P_discharge_max"])

    # 7b. Battery discharging limits, new batteries
    for s in S_new:
        for t in T:
            # model.addConstr(g_dis[s, t] >= 0)
            model.addConstr(
                g_dis[s, t] <= soc_s_max[s] / (batteries.loc[s, "hour_capacity"])
            )

    # 8. State of charge limits
    for s in S:
        for t in T:
            model.addConstr(soc[s, t] >= batteries.loc[s, "SOC_min"] * soc_s_max[s])
            model.addConstr(soc[s, t] <= batteries.loc[s, "SOC_max"] * soc_s_max[s])

    # 9. Battery state of charge dynamics
    for s in S:
        for t in T[1:]:  # Exclude time t=0
            model.addConstr(
                soc[s, t]
                == soc[s, t - pd.Timedelta("1h")]
                + batteries.loc[s, "eta_charge"] * g_ch[s, t]
                - g_dis[s, t] / batteries.loc[s, "eta_discharge"]
            )

    # 10. Initial state of charge
    for s in S:
        model.addConstr(soc[s, T[0]] == batteries.loc[s, "SOC_min"] * soc_s_max[s])

    # Optimize the model
    model.setParam("MIPGap", MIPGap)
    model.setParam("Timelimit", TIMELIMIT)
    model.setParam("BarConvTol", MIPGap)

    build_end_time = time()

    print(f"Model built in {build_end_time - build_start_time} seconds.")
    model_optimize_start_time = time()
    model.optimize()
    model_optimize_end_time = time()
    # endregion

    # region Post-processing and saving results
    save_folder = config.get("save_folder", None)
    decision_variables_folder = os.path.join(save_folder, "decision_variables")
    if not os.path.exists(decision_variables_folder):
        os.makedirs(decision_variables_folder)
    # Save generation
    generation_data = [(t, i, g[i, t].X) for i in G for t in T]
    generation_df = pd.DataFrame(
        generation_data, columns=["snapshot", "generator", "value"]
    )
    generation_reshaped = _reshape_variable(generation_df, "generator", "snapshot")
    generation_reshaped.to_csv(
        os.path.join(decision_variables_folder, "generation.csv")
    )

    # Save power flow
    power_flow_data = [(t, b, f[b, t].X) for b in B for t in T]
    power_flow_df = pd.DataFrame(
        power_flow_data, columns=["snapshot", "branch", "value"]
    )
    power_flow_reshaped = _reshape_variable(power_flow_df, "branch", "snapshot")
    power_flow_reshaped.to_csv(
        os.path.join(decision_variables_folder, "power_flow.csv")
    )

    # Save load shedding
    load_shedding_data = [(t, n, sh[n, t].X) for n in N for t in T]
    load_shedding_df = pd.DataFrame(
        load_shedding_data, columns=["snapshot", "node", "value"]
    )
    load_shedding_reshaped = _reshape_variable(load_shedding_df, "node", "snapshot")
    load_shedding_reshaped.to_csv(
        os.path.join(decision_variables_folder, "load_shedding.csv")
    )

    # Save curtailment
    curtailment_data = [(t, i, c[i, t].X) for i in G for t in T]
    curtailment_df = pd.DataFrame(
        curtailment_data, columns=["snapshot", "generator", "value"]
    )
    curtailment_reshaped = _reshape_variable(curtailment_df, "generator", "snapshot")
    curtailment_reshaped.to_csv(
        os.path.join(decision_variables_folder, "curtailment.csv")
    )

    # Save battery charging
    battery_charging_data = [(t, s, g_ch[s, t].X) for s in S for t in T]
    battery_charging_df = pd.DataFrame(
        battery_charging_data, columns=["snapshot", "battery", "value"]
    )
    battery_charging_reshaped = _reshape_variable(
        battery_charging_df, "battery", "snapshot"
    )
    battery_charging_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_charging.csv")
    )

    # Save battery discharging
    battery_discharging_data = [(t, s, g_dis[s, t].X) for s in S for t in T]
    battery_discharging_df = pd.DataFrame(
        battery_discharging_data, columns=["snapshot", "battery", "value"]
    )
    battery_discharging_reshaped = _reshape_variable(
        battery_discharging_df, "battery", "snapshot"
    )
    battery_discharging_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_discharging.csv")
    )

    # Save battery state of charge
    battery_soc_data = [(t, s, soc[s, t].X) for s in S for t in T]
    battery_soc_df = pd.DataFrame(
        battery_soc_data, columns=["snapshot", "battery", "value"]
    )
    battery_soc_reshaped = _reshape_variable(battery_soc_df, "battery", "snapshot")
    battery_soc_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_soc.csv")
    )

    # Save generator build
    generator_build_data = [(i, x[i].X) for i in G_new]
    generator_build_df = pd.DataFrame(
        generator_build_data, columns=["generator", "value"]
    )
    generator_build_df.to_csv(
        os.path.join(decision_variables_folder, "generator_build.csv"), index=False
    )

    # Save branch build
    branch_build_data = [(b, y[b].X) for b in B_new]
    branch_build_df = pd.DataFrame(branch_build_data, columns=["branch", "value"])
    branch_build_df.to_csv(
        os.path.join(decision_variables_folder, "branch_build.csv"), index=False
    )

    # Save battery build
    battery_build_data = [(s, soc_s_max[s].X) for s in S_new]
    battery_build_df = pd.DataFrame(battery_build_data, columns=["battery", "value"])
    battery_build_df.to_csv(
        os.path.join(decision_variables_folder, "battery_build.csv"), index=False
    )

    # Save generator capacities
    generator_capacity_data = [(i, p_i_max[i].X) for i in G_new]
    generator_capacity_df = pd.DataFrame(
        generator_capacity_data, columns=["generator", "value"]
    )
    generator_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "generator_capacity.csv"), index=False
    )

    # Save branch capacities
    branch_capacity_data = [(b, p_b_max[b].X) for b in B_new]
    branch_capacity_df = pd.DataFrame(branch_capacity_data, columns=["branch", "value"])
    branch_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "branch_capacity.csv"), index=False
    )

    # Save generator capacities
    p_i_w = [(i, w, p_i_w[i, w].X) for i in G for w in W]
    p_i_w_df = pd.DataFrame(p_i_w, columns=["generator", "week", "value"])
    p_i_w_df.to_csv(os.path.join(decision_variables_folder, "p_i_w.csv"), index=False)

    # endregion

    return (
        model,
        build_end_time - build_start_time,
        model_optimize_end_time - model_optimize_start_time,
    )


# endregion


# region GTSEP_v3
def GTSEP_v3(config: dict) -> gp.Model:
    """GTSEP model from the specialization project. Modeling battery investments as continous variables (unconstrained investments in batteries)."""
    # region Model setup and running
    must_have_keys = [
        "data_folder_name",
        "VOLL",
        "CC",
        "CO2_price",
        "E_limit",
        "p_max_new_branch",
        "p_min_new_branch",
        "expansion_factor",
        "MS",
        "model_name",
        "MIPGap",
    ]
    for key in must_have_keys:
        if key not in config:
            raise KeyError(
                f"Required key '{key}' not found in config. \nRequired keys: {must_have_keys}\nConfig keys: {config.keys()}"
            )

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

    # Load data
    data_folder_path = os.path.join(PROCESSED_DATA_FOLDER, data_folder_name)
    input_data = load_csv_files_from_folder(data_folder_path)
    batteries = input_data["batteries"]
    branches = input_data["branches"]
    capacity_factors = input_data["capacity_factors"]
    generators = input_data["generators"]
    generator_costs = input_data["generator_costs"]
    hourly_demand = input_data["hourly_demand"]
    nodes = input_data["nodes"]

    # Data processing
    # Create new branches
    # Add a new column 'exists' to the original branches dataframe and set it to 1
    branches["exists"] = 1
    # Create a copy of the dataframe for the "new" branches
    branches_new = branches.copy()
    # Update the index by appending " new" to the original index
    branches_new.index = branches_new.index.astype(str) + " new"
    # Set the 'exists' column to 0 for the new branches
    branches_new["exists"] = 0
    # Concatenate the original dataframe and the new dataframe
    branches = pd.concat([branches, branches_new])
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
    batteries["exists"] = 0

    # Create sets
    N = nodes.index.to_list()
    G_old = generators[generators["exists"] == 1].index.to_list()
    G_new = generators[generators["exists"] == 0].index.to_list()
    G = generators.index.to_list()
    B_old = branches[branches["exists"] == 1].index.to_list()
    B_new = branches[branches["exists"] == 0].index.to_list()
    B = branches.index.to_list()
    S_new = batteries[batteries["exists"] == 0].index.to_list()
    S_old = batteries[batteries["exists"] == 1].index.to_list()
    S = batteries.index.to_list()
    T = hourly_demand.index.to_list()

    # Create mappings
    (
        branches_out_of_node,
        branches_into_node,
        batteries_at_node,
        generators_at_node,
    ) = _create_mappings(nodes, branches, generators, batteries)

    build_start_time = time()
    # Create model
    model_name = model_name if model_name else "GTSEP_v0"
    model = gp.Model(model_name)

    # Decision variables
    g = model.addVars(G, T, name="g", lb=0)  # Power generation dispatch
    f = model.addVars(B, T, name="f", lb=-GRB.INFINITY, ub=GRB.INFINITY)  # Power flow
    sh = model.addVars(N, T, name="sh", lb=0)  # Load shedding
    c = model.addVars(G, T, name="c", lb=0)  # Curtailment
    g_ch = model.addVars(S, T, name="g_ch", lb=0)  # Battery charging
    g_dis = model.addVars(S, T, name="g_dis", lb=0)  # Battery discharging
    soc = model.addVars(S, T, name="soc", lb=0)  # State of charge
    soc_s_max = model.addVars(
        S_new, name="soc_s_max", lb=0
    )  # Max SOC for new batteries
    p_i_max = model.addVars(
        G_new, name="p_i_max", lb=0
    )  # Max capacity of new generators
    p_b_max = model.addVars(B_new, name="p_b_max", lb=0)  # Max capacity of new branches

    # Objective function: Minimize cost
    objective = (
        gp.quicksum(
            (
                generators.loc[i, "marginal_cost"]
                + generators.loc[i, "co2_emissions"] * CO2_price
            )
            * g[i, t]
            for i in G
            for t in T
        )
        # + gp.quicksum(
        #     batteries.loc[s, "MC"] * g_dis[s, t] * batteries.loc[s, "eta_discharge"]
        #     for s in S
        #     for t in T
        # )
        + gp.quicksum(VOLL * sh[n, t] for n in N for t in T)
        + gp.quicksum(CC * c[i, t] for i in G for t in T)
        + gp.quicksum(generators.loc[i, "capital_cost"] * p_i_max[i] for i in G_new)
        + gp.quicksum(branches.loc[b, "capital_cost"] * p_b_max[b] for b in B_new)
        + gp.quicksum(batteries.loc[s, "capital_cost"] * soc_s_max[s] for s in S_new)
    )
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
    # 1. Power balance
    for n in N:
        for t in T:
            model.addConstr(
                gp.quicksum(g[i, t] - c[i, t] for i in generators_at_node[n])
                + gp.quicksum(
                    f[b, t] * (1 - branches.loc[b, "loss_factor"])
                    for b in branches_into_node[n]
                )
                - gp.quicksum(f[b, t] for b in branches_out_of_node[n])
                - gp.quicksum(
                    g_ch[s, t] - batteries.loc[s, "eta_discharge"] * g_dis[s, t]
                    for s in batteries_at_node[n]
                )
                + sh[n, t]
                == hourly_demand.loc[t, n]
            )

    # 2a. Load shedding limits
    for n in N:
        for t in T:
            model.addConstr(sh[n, t] <= MS * hourly_demand.loc[t, n])

    # 2b. Curtailment limits
    for i in G:
        for t in T:
            model.addConstr(c[i, t] <= g[i, t])

    # 3a. Generator output limits (old generators)
    for i in G_old:
        p_max = generators.loc[i, "p_nom"]
        for t in T:
            capacity_factor = capacity_factors.loc[t, i]
            model.addConstr(g[i, t] <= p_max * capacity_factor)
            # Lower bound is 0 by default

    # 3b. Generator output limits (new generators)
    for i in G_new:
        for t in T:
            original_generator_id = " ".join(i.split(" ")[:-1])
            capacity_factor = capacity_factors.loc[t, original_generator_id]
            model.addConstr(g[i, t] <= p_i_max[i] * capacity_factor)
            # Lower bound is 0 by default

    # 3c. New generator capacity limits
    for i in G_new:
        p_max = generators.loc[i, "p_nom"]
        model.addConstr(p_i_max[i] <= expansion_factor * p_max)

    # 4a. Branch flow limits (old branches)
    for b in B_old:
        for t in T:
            model.addConstr(f[b, t] >= -branches.loc[b, "p_max"])
            model.addConstr(f[b, t] <= branches.loc[b, "p_max"])

    # 4b. Branch flow limits (new branches)
    for b in B_new:
        for t in T:
            model.addConstr(f[b, t] >= -p_b_max[b])
            model.addConstr(f[b, t] <= p_b_max[b])

    # 4c. New branch capacity limits
    for b in B_new:
        # model.addConstr(p_b_max[b] >= p_min_new_branch)
        model.addConstr(p_b_max[b] <= p_max_new_branch)

    # # 5. Emission restrictions
    # model.addConstr(
    #     gp.quicksum(g[i, t] * generators.loc[i, "co2_emissions"] for i in G for t in T)
    #     <= E_limit
    # )

    # 6a. Battery charging limits, old batteries
    for s in S_old:
        for t in T:
            model.addConstr(g_ch[s, t] >= batteries.loc[s, "P_charge_min"])
            model.addConstr(g_ch[s, t] <= batteries.loc[s, "P_charge_max"])

    # 6b. Battery charging limits, new batteries
    for s in S_new:
        for t in T:
            model.addConstr(
                g_ch[s, t]
                <= soc_s_max[s]
                / (batteries.loc[s, "hour_capacity"] * batteries.loc[s, "cdrate"])
            )

    # 7a. Battery discharging limits, old batteries
    for s in S_old:
        for t in T:
            model.addConstr(g_dis[s, t] >= batteries.loc[s, "P_discharge_min"])
            model.addConstr(g_dis[s, t] <= batteries.loc[s, "P_discharge_max"])

    # 7b. Battery discharging limits, new batteries
    for s in S_new:
        for t in T:
            model.addConstr(
                g_dis[s, t] <= soc_s_max[s] / (batteries.loc[s, "hour_capacity"])
            )

    # 8. State of charge limits
    for s in S:
        for t in T:
            model.addConstr(soc[s, t] >= batteries.loc[s, "SOC_min"] * soc_s_max[s])
            model.addConstr(soc[s, t] <= batteries.loc[s, "SOC_max"] * soc_s_max[s])

    # 9. Battery state of charge dynamics
    for s in S:
        for t in T[1:]:  # Exclude time t=0
            model.addConstr(
                soc[s, t]
                == soc[s, t - pd.Timedelta("1h")]
                + batteries.loc[s, "eta_charge"] * g_ch[s, t]
                - g_dis[s, t] / batteries.loc[s, "eta_discharge"]
            )

    # 10. Initial state of charge
    for s in S:
        model.addConstr(soc[s, T[0]] == batteries.loc[s, "SOC_min"] * soc_s_max[s])

    # Optimize the model
    model.setParam("MIPGap", MIPGap)
    model.setParam("Timelimit", TIMELIMIT)
    model.setParam("BarConvTol", MIPGap)

    build_end_time = time()

    print(f"Model built in {build_end_time - build_start_time} seconds.")
    model_optimize_start_time = time()
    model.optimize()
    model_optimize_end_time = time()
    # endregion

    # region Post-processing and saving results
    save_folder = config.get("save_folder", None)
    decision_variables_folder = os.path.join(save_folder, "decision_variables")
    if not os.path.exists(decision_variables_folder):
        os.makedirs(decision_variables_folder)
    # Save generation
    generation_data = [(t, i, g[i, t].X) for i in G for t in T]
    generation_df = pd.DataFrame(
        generation_data, columns=["snapshot", "generator", "value"]
    )
    generation_reshaped = _reshape_variable(generation_df, "generator", "snapshot")
    generation_reshaped.to_csv(
        os.path.join(decision_variables_folder, "generation.csv")
    )

    # Save power flow
    power_flow_data = [(t, b, f[b, t].X) for b in B for t in T]
    power_flow_df = pd.DataFrame(
        power_flow_data, columns=["snapshot", "branch", "value"]
    )
    power_flow_reshaped = _reshape_variable(power_flow_df, "branch", "snapshot")
    power_flow_reshaped.to_csv(
        os.path.join(decision_variables_folder, "power_flow.csv")
    )

    # Save load shedding
    load_shedding_data = [(t, n, sh[n, t].X) for n in N for t in T]
    load_shedding_df = pd.DataFrame(
        load_shedding_data, columns=["snapshot", "node", "value"]
    )
    load_shedding_reshaped = _reshape_variable(load_shedding_df, "node", "snapshot")
    load_shedding_reshaped.to_csv(
        os.path.join(decision_variables_folder, "load_shedding.csv")
    )

    # Save curtailment
    curtailment_data = [(t, i, c[i, t].X) for i in G for t in T]
    curtailment_df = pd.DataFrame(
        curtailment_data, columns=["snapshot", "generator", "value"]
    )
    curtailment_reshaped = _reshape_variable(curtailment_df, "generator", "snapshot")
    curtailment_reshaped.to_csv(
        os.path.join(decision_variables_folder, "curtailment.csv")
    )

    # Save battery charging
    battery_charging_data = [(t, s, g_ch[s, t].X) for s in S for t in T]
    battery_charging_df = pd.DataFrame(
        battery_charging_data, columns=["snapshot", "battery", "value"]
    )
    battery_charging_reshaped = _reshape_variable(
        battery_charging_df, "battery", "snapshot"
    )
    battery_charging_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_charging.csv")
    )

    # Save battery discharging
    battery_discharging_data = [(t, s, g_dis[s, t].X) for s in S for t in T]
    battery_discharging_df = pd.DataFrame(
        battery_discharging_data, columns=["snapshot", "battery", "value"]
    )
    battery_discharging_reshaped = _reshape_variable(
        battery_discharging_df, "battery", "snapshot"
    )
    battery_discharging_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_discharging.csv")
    )

    # Save battery state of charge
    battery_soc_data = [(t, s, soc[s, t].X) for s in S for t in T]
    battery_soc_df = pd.DataFrame(
        battery_soc_data, columns=["snapshot", "battery", "value"]
    )
    battery_soc_reshaped = _reshape_variable(battery_soc_df, "battery", "snapshot")
    battery_soc_reshaped.to_csv(
        os.path.join(decision_variables_folder, "battery_soc.csv")
    )

    # Save battery build
    battery_capacity_data = [(s, soc_s_max[s].X) for s in S_new]
    battery_capacity_df = pd.DataFrame(
        battery_capacity_data, columns=["battery", "value"]
    )
    battery_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "battery_capacity.csv"), index=False
    )

    # Save generator capacities
    generator_capacity_data = [(i, p_i_max[i].X) for i in G_new]
    generator_capacity_df = pd.DataFrame(
        generator_capacity_data, columns=["generator", "value"]
    )
    generator_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "generator_capacity.csv"), index=False
    )

    # Save branch capacities
    branch_capacity_data = [(b, p_b_max[b].X) for b in B_new]
    branch_capacity_df = pd.DataFrame(branch_capacity_data, columns=["branch", "value"])
    branch_capacity_df.to_csv(
        os.path.join(decision_variables_folder, "branch_capacity.csv"), index=False
    )

    # endregion

    return (
        model,
        build_end_time - build_start_time,
        model_optimize_end_time - model_optimize_start_time,
    )


# endregion


MODEL_REGISTRY = {
    "GTSEP_v0": GTSEP_v0,
    "GTSEP_v1": GTSEP_v1,
    "GTSEP_v2": GTSEP_v2,
    "GTSEP_v3": GTSEP_v3,
    "GTSEP_v1_multi": GTSEP_v1_multi,
    "GTSEP_v1a_multi": GTSEP_v1a_multi,
    "GTSEP_stochastic_v1": GTSEP_stochastic_v1,
}


def get_model(config: dict) -> gp.Model:
    """Return the appropriate model based on the config."""
    model_name = config.get("model_name", "non_existent_model")
    try:
        model_func = MODEL_REGISTRY[model_name]
    except KeyError:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' is not registered. Available models are: {available}"
        )
    return model_func(config)


if __name__ == "__main__":
    print(f"Data folder: {PROCESSED_DATA_FOLDER}")
    config = load_model_config()
    print(config)
    start = time()
    model = GTSEP_v0(config)
    print(f"Model created in {time() - start} seconds.")

    start = time()
    model.optimize()
    print(f"Model optimized in {time() - start} seconds.")
