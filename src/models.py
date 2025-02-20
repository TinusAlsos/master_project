from time import time
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from src.utils import load_csv_files_from_folder, load_model_config
import os

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
            batteries.loc[s, "capital_cost"] * batteries.loc[s, "P_discharge_max"] * batteries.loc[s, "hour_capacity"] * z[s]
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

#region GTSEP_v1
def GTSEP_v1(config: dict) -> gp.Model:
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
    soc_s_max = model.addVars(S_new, name="soc_s_max", lb=0)  # Max SOC for new batteries
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
            batteries.loc[s, "capital_cost"] * soc_s_max[s]
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
            model.addConstr(g_ch[s, t] >= 0)
            model.addConstr(g_ch[s, t] <= soc_s_max[s] / (batteries.loc[s, "hour_capacity"] * batteries.loc[s, "cdrate"]))

    # 7a. Battery discharging limits, old batteries
    for s in S_old:
        for t in T:
            model.addConstr(g_dis[s, t] >= batteries.loc[s, "P_discharge_min"])
            model.addConstr(g_dis[s, t] <= batteries.loc[s, "P_discharge_max"])

    # 7b. Battery discharging limits, new batteries
    for s in S_new:
        for t in T:
            model.addConstr(g_dis[s, t] >= 0)
            model.addConstr(g_dis[s, t] <= soc_s_max[s] / (batteries.loc[s, "hour_capacity"]))

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

#region GTSEP_v2
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
    time_indexes.loc[(time_indexes["week"] == 1) & (time_indexes["year"] == 2014), "week"] = 53
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
    soc_s_max = model.addVars(S_new, name="soc_s_max", lb=0)  # Max SOC for new batteries
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
        + gp.quicksum(
            batteries.loc[s, "capital_cost"] * soc_s_max[s]
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
            model.addConstr(g[i, t] == p_i_w[i,w] * capacity_factor)
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
            model.addConstr(g_ch[s, t] <= soc_s_max[s] / (batteries.loc[s, "hour_capacity"] * batteries.loc[s, "cdrate"]))

    # 7a. Battery discharging limits, old batteries
    for s in S_old:
        for t in T:
            model.addConstr(g_dis[s, t] >= batteries.loc[s, "P_discharge_min"])
            model.addConstr(g_dis[s, t] <= batteries.loc[s, "P_discharge_max"])

    # 7b. Battery discharging limits, new batteries
    for s in S_new:
        for t in T:
            # model.addConstr(g_dis[s, t] >= 0)
            model.addConstr(g_dis[s, t] <= soc_s_max[s] / (batteries.loc[s, "hour_capacity"]))

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
    p_i_w_df.to_csv(
        os.path.join(decision_variables_folder, "p_i_w.csv"), index=False
    )

    

    # endregion

    return (
        model,
        build_end_time - build_start_time,
        model_optimize_end_time - model_optimize_start_time,
    )
# endregion


MODEL_REGISTRY = {"GTSEP_v0": GTSEP_v0,
                  "GTSEP_v1": GTSEP_v1,
                  "GTSEP_v2": GTSEP_v2}


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
