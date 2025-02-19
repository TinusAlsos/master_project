import calendar
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
import os
import pandas as pd
import seaborn as sns

np.random.seed(3)  # For reproducibility

COUNTIRES_TO_FILTER_TO_MAINLAND = ["France", "Norway"]
node_to_city = {
    "ES1 0": "Murcia",
    "ES1 1": "Bilbao",
    "ES1 2": "Zaragoza",
    "ES1 3": "Seville",
    "ES1 4": "Lugo",
    "ES1 5": "Madrid",
    "ES1 6": "Valencia",
    "ES1 7": "Salamanca",
    "ES1 8": "Barcelona",
    "PT1 0": "Porto",
    "PT1 1": "Lisbon",
}
city_to_node = {v: k for k, v in node_to_city.items()}
GENERATOR_ORDER = [
    "CCGT",
    "coal",
    "offwind-ac",
    "onwind",
    "solar",
    "ror",
    "oil",
    "biomass",
    "nuclear",
    "offwind-dc",
    "lignite",
    "OCGT",
]
country_code_to_country = {
    "AL": "Albania",
    "AT": "Austria",
    "BA": "Bosnia and Herz.",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "CH": "Switzerland",
    "CZ": "Czechia",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "GB": "United Kingdom",
    "GR": "Greece",
    "HR": "Croatia",
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "ME": "Montenegro",
    "MK": "North Macedonia",
    "NL": "Netherlands",
    "NO": "Norway",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "RS": "Serbia",
    "SE": "Sweden",
    "SI": "Slovenia",
    "SK": "Slovakia",
}
country_to_country_code = {v: k for k, v in country_code_to_country.items()}
country_code_to_color = {
    "AL": "lightblue",  # Albania
    "AT": "palegreen",  # Austria (contrasts with CZ, DE, HU)
    "BA": "paleturquoise",  # Bosnia and Herzegovina (contrasts with HR, RS)
    "BE": "lightgoldenrodyellow",  # Belgium (contrasts with NL, FR, DE)
    "BG": "lightsteelblue",  # Bulgaria (contrasts with RO, GR)
    "CH": "rosybrown",  # Switzerland (contrasts with FR, DE, IT)
    "CZ": "skyblue",  # Czech Republic (contrasts with DE, AT, PL)
    "DE": "mistyrose",  # Germany (contrasts with NL, FR, PL, AT, CZ, DK)
    "DK": "lavender",  # Denmark (contrasts with DE, SE)
    "EE": "palegreen",  # Estonia (contrasts with LV, FI)
    "ES": "gainsboro",  # Spain (contrasts with PT, FR)
    "FI": "powderblue",  # Finland (contrasts with SE, NO, EE)
    "FR": "lightsteelblue",  # France (contrasts with DE, BE, ES, IT, CH)
    "GB": "pink",  # United Kingdom (contrasts with IE, FR)
    "GR": "khaki",  # Greece (contrasts with AL, MK, BG)
    "HR": "peachpuff",  # Croatia (contrasts with SI, BA, RS, HU)
    "HU": "goldenrod",  # Hungary (contrasts with AT, SK, RO, HR, SR)
    "IE": "tan",  # Ireland (contrasts with GB)
    "IT": "navajowhite",  # Italy (contrasts with FR, CH, AT, SI)
    "LT": "wheat",  # Lithuania (contrasts with LV, PL)
    "LU": "mediumaquamarine",  # Luxembourg (contrasts with BE, DE, FR)
    "LV": "aquamarine",  # Latvia (contrasts with EE, LT)
    "ME": "burlywood",  # Montenegro (contrasts with RS, AL, BA)
    "MK": "lemonchiffon",  # North Macedonia (contrasts with AL, GR, BG, RS)
    "NL": "goldenrod",  # Netherlands (contrasts with BE, DE)
    "NO": "lightgray",  # Norway (contrasts with SE, FI, DK)
    "PL": "rosybrown",  # Poland (contrasts with DE, CZ, SK, LT)
    "PT": "darkgrey",  # Portugal (contrasts with ES)
    "RO": "lightpink",  # Romania (contrasts with HU, BG, RS)
    "RS": "bisque",  # Serbia (contrasts with HU, RO, BG, MK, ME, HR, BA)
    "SE": "lightseagreen",  # Sweden (contrasts with NO, FI, DK)
    "SI": "lightcyan",  # Slovenia (contrasts with AT, HR, IT)
    "SK": "palegoldenrod",  # Slovakia (contrasts with CZ, AT, HU, PL)
}
coordinates_to_city_or_country = {
    (2.1734, 41.3851): "Barcelona",
    (-3.7038, 40.4168): "Madrid",
    (-2.9349, 43.2630): "Bilbao",
    (-8.6110, 41.1496): "Porto",
    (-9.1393, 38.7223): "Lisbon",
    (-5.6635, 40.9701): "Salamanca",
    (-5.9845, 37.3891): "Sevilla",
    (-1.1307, 37.9922): "Murcia",
    (-0.8891, 41.6488): "Zaragoza",
    (-7.5560, 43.0125): "Lugo",
    (-0.3763, 39.4699): "Valencia",
}
city_or_country_to_coordinates = {
    v: k for k, v in coordinates_to_city_or_country.items()
}


def _filter_out_overseas(world: gpd.GeoDataFrame, countries: str) -> gpd.GeoDataFrame:
    for country_name in countries:
        country = world[world["NAME"] == country_name]
        if isinstance(country["geometry"].values[0], MultiPolygon):
            mainland_country = max(
                country.iloc[0].geometry.geoms, key=lambda p: p.area
            )  # Keep the largest polygon (mainland)
            country.at[country.index[0], "geometry"] = (
                mainland_country  # Overwrite Norway's geometry
            )
            world.loc[world["NAME"] == country_name, "geometry"] = country["geometry"]
    return world


def plot_background(nodes: pd.DataFrame) -> plt.Figure:
    world_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "data", "countries", "ne_110m_admin_0_countries.shp")
    world = gpd.read_file(world_path)

    # Filter out overseas territories
    COUNTIRES_TO_FILTER_TO_MAINLAND = ["France", "Norway"]
    world = _filter_out_overseas(world, COUNTIRES_TO_FILTER_TO_MAINLAND)

    relevant_countries = [
        country_code_to_country[country_code]
        for country_code in nodes["country"].unique()
    ]
    countries = [world[world["NAME"] == country] for country in relevant_countries]

    # Plot the map and initialize the figure
    fig, ax = plt.subplots(figsize=(10, 10))

    for idx, country in enumerate(countries):
        country.plot(
            ax=ax,
            color=country_code_to_color[
                country_to_country_code[relevant_countries[idx]]
            ],
            edgecolor="black",
        )
    return fig, ax


def _get_generator_order(generators: pd.DataFrame) -> list:
    """Returns a list of unique generator types in the order they appear in the DataFrame."""
    order = [
        generator
        for generator in GENERATOR_ORDER
        if generator in generators["carrier"].unique()
    ]
    return order


def plot_buses_and_lines(buses, lines, savefolder=None):
    """Links can be links or lines"""
    # Convert Iberia_buses DataFrame to a GeoDataFrame
    geometry_buses = gpd.points_from_xy(buses["x"], buses["y"])
    gdf_buses = gpd.GeoDataFrame(buses, geometry=geometry_buses, crs="EPSG:4326")

    # Create a dictionary to map bus IDs to their coordinates for easy access
    bus_coords = gdf_buses[["x", "y"]].to_dict("index")

    # Generate LineString geometries based on the coordinates of bus0 and bus1 in the lines DataFrame
    line_geometries = []
    for _, row in lines.iterrows():
        if row["bus0"] in bus_coords and row["bus1"] in bus_coords:
            point0 = (bus_coords[row["bus0"]]["x"], bus_coords[row["bus0"]]["y"])
            point1 = (bus_coords[row["bus1"]]["x"], bus_coords[row["bus1"]]["y"])
            line_geometries.append(LineString([point0, point1]))
        else:
            print(f"Excluded line with missing bus coordinates: {row}")

    # Create a GeoDataFrame for the lines
    gdf_lines = gpd.GeoDataFrame(lines, geometry=line_geometries, crs="EPSG:4326")

    fig, ax = plot_background(buses)

    # Plot buses and lines
    gdf_buses.plot(
        ax=ax, color="black", marker="o", markersize=30, zorder=20
    )  # Plot buses with increased size
    if not gdf_lines.empty:
        gdf_lines.plot(
            ax=ax, color="red", linewidth=2, zorder=15
        )  # Plot transmission lines
    # Create custom legend elements with thicker lines and larger font size
    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            label="Nodes",
            markersize=10,
            linestyle="None",
        ),
        mlines.Line2D([0], [0], color="red", linewidth=4, label="Lines"),
    ]

    # Add the custom legend with increased font size
    ax.legend(handles=legend_elements, loc="upper right", fontsize=18)

    # Customize the plot
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax.set_axis_off()
    plt.grid(True)
    plt.show()
    print(f"Plotting grid simple")

    if savefolder:
        savepath = os.path.join(savefolder, "grid_simple.png")
        fig.savefig(savepath, bbox_inches="tight")


def plot_battery_cases(
    nodes: pd.DataFrame,
    battery_sets: list[list] = None,
    battery_colors: list[str] = None,
    battery_labels: list[str] = None,
    savefolder: str = None,
):
    if battery_sets is None:
        battery_sets = [
            ["ES1 1 bat"],
            ["ES1 1 bat", "ES1 5 bat", "ES1 8 bat", "PT1 0 bat", "PT1 1 bat"],
            [
                "ES1 0 bat",
                "ES1 1 bat",
                "ES1 2 bat",
                "ES1 3 bat",
                "ES1 4 bat",
                "ES1 8 bat",
                "PT1 0 bat",
                "PT1 1 bat",
            ],
        ]
    if battery_colors is None:
        battery_colors = ["red", "blue", "green"]
    if battery_labels is None:
        battery_labels = ["Exogenous Battery", "5 Batteries", "All Batteries"]
    # Convert nodes DataFrame to a GeoDataFrame
    geometry_buses = gpd.points_from_xy(nodes["x"], nodes["y"])
    gdf_buses = gpd.GeoDataFrame(nodes, geometry=geometry_buses, crs="EPSG:4326")

    fig, ax = plot_background(nodes)
    # Define colors for the three sets
    battery_colors = ["red", "blue", "green"]
    battery_labels = ["Exogenous Battery", "5 Batteries", "All Batteries"]

    # Plot the batteries in different colors for each set
    counter = 0
    for battery_set, color, label in zip(battery_sets, battery_colors, battery_labels):
        counter += 1
        battery_nodes = [" ".join(bat.split()[:-1]) for bat in battery_set]
        battery_points = gdf_buses[gdf_buses.index.isin(battery_nodes)]
        offset_x = 0
        offset_y = -0.15
        if counter == 3:
            offset_x = 0.1
            offset_y = 0
        if counter == 2:
            offset_x = -0.1
            offset_y = 0
        ax.scatter(
            battery_points["x"] + offset_x,
            battery_points["y"] + offset_y,
            color=color,
            s=100,
            label=label,
            zorder=10 - counter,
        )

    # Add text labels for major cities
    for node, city in node_to_city.items():
        if node in nodes.index:
            plt.text(
                nodes.loc[node, "x"],
                nodes.loc[node, "y"] + 0.2,
                city,
                fontsize=14,
                ha="center",
                va="center",
                weight="bold",
                color="black",
            )
        else:
            print(f"Excluded city with missing coordinates: {city}")

    # Add legend for battery sets
    ax.legend(loc="upper right", fontsize=14)

    # Customize the plot
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax.set_axis_off()  # Remove axis lines

    # Tighten the layout to reduce whitespace
    plt.tight_layout()
    if savefolder:
        savepath = os.path.join(savefolder, "grid_battery_cases.png")
        plt.savefig(savepath, bbox_inches="tight")
    # Show the plot
    plt.show()


def plot_sized_generators_and_lines(
    buses: pd.DataFrame,
    branches: pd.DataFrame,
    generators: pd.DataFrame,
    savefolder=None,
    new_only=False,
):
    """Links can be links or branches"""

    # Convert buses DataFrame to a GeoDataFrame
    geometry_buses = gpd.points_from_xy(buses["x"], buses["y"])
    gdf_buses = gpd.GeoDataFrame(buses, geometry=geometry_buses, crs="EPSG:4326")

    # Create a dictionary to map bus IDs to their coordinates for easy access
    bus_coords = gdf_buses[["x", "y"]].to_dict("index")

    # Generate list of line segments for LineCollection based on bus coordinates
    lines = []
    line_widths = []
    max_capacity = branches["p_max"].max()
    min_capacity = branches["p_max"].min()
    min_line_width = 1
    max_line_width = 5

    for _, row in branches.iterrows():
        if row["bus0"] in bus_coords and row["bus1"] in bus_coords:
            point0 = (bus_coords[row["bus0"]]["x"], bus_coords[row["bus0"]]["y"])
            point1 = (bus_coords[row["bus1"]]["x"], bus_coords[row["bus1"]]["y"])
            lines.append([point0, point1])

            # Normalize p_max between min_line_width and max_line_width
            divide_by = (max_capacity - min_capacity) * (
                max_line_width - min_line_width
            )
            divide_by = divide_by if divide_by != 0 else 1
            linewidth = min_line_width + (row["p_max"] - min_capacity) / divide_by
            line_widths.append(linewidth)
        else:
            print(f"Excluded line with missing bus coordinates: {row}")

    # Load the Iberian map shapefile
    fig, ax = plot_background(buses)

    # Create and plot the LineCollection for transmission lines
    lc = LineCollection(lines, colors="red", linewidths=line_widths, zorder=15)
    ax.add_collection(lc)

    # Plot buses
    gdf_buses.plot(
        ax=ax, color="black", marker="o", markersize=10
    )  # No legend entry for buses

    # Prepare to plot generators with offsets based on carrier grouping
    max_power = generators["p_nom"].max()
    min_power = generators["p_nom"].min()
    min_marker_size = 50  # Define minimum marker size
    max_marker_size = 300  # Define maximum marker size
    offset_radius = 0.1  # Base offset distance around the node

    for bus, group in generators.groupby("bus"):
        if bus in bus_coords:
            x, y = bus_coords[bus]["x"], bus_coords[bus]["y"]

            # Group by carrier and calculate cumulative capacity
            carrier_group = (
                group.groupby("carrier")
                .agg({"p_nom": "sum", "color": "first"})
                .reset_index()
            )
            num_carriers = len(carrier_group)

            # Determine the offsets for each carrier count configuration
            if num_carriers == 1:
                offsets = [(0, 0)]  # Centered on the node
            elif num_carriers == 2:
                effective_offset = offset_radius * np.sqrt(2) / 2
                offsets = [
                    (0, effective_offset),
                    (0, -effective_offset),
                ]  # North and South
            elif num_carriers == 3:
                effective_offset = offset_radius * np.sqrt(2) / 2
                offsets = [
                    (0, effective_offset),
                    (-effective_offset, -effective_offset),
                    (effective_offset, -effective_offset),
                ]  # Triangle
            elif num_carriers == 4:
                offsets = [
                    (0, offset_radius),
                    (0, -offset_radius),
                    (-offset_radius, 0),
                    (offset_radius, 0),
                ]  # Square
            elif num_carriers == 5:
                angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
                offsets = [
                    (offset_radius * np.cos(a), offset_radius * np.sin(a))
                    for a in angles
                ]  # Pentagon
            else:
                angles = np.linspace(0, 2 * np.pi, num_carriers, endpoint=False)
                offsets = [
                    (offset_radius * np.cos(a), offset_radius * np.sin(a))
                    for a in angles
                ]

            # Plot each carrier group with offset and cumulative size
            for (dx, dy), (_, carrier) in zip(offsets, carrier_group.iterrows()):
                cum_capacity = carrier["p_nom"]
                marker_size = min_marker_size + (cum_capacity - min_power) / (
                    max_power - min_power
                ) * (max_marker_size - min_marker_size)
                ax.scatter(
                    x + dx, y + dy, color=carrier["color"], s=marker_size, zorder=20
                )

    # Create legend entries for generator types and capacities
    unique_carriers = generators[["carrier", "color", "nice_name"]].drop_duplicates()
    legend_elements = [
        mlines.Line2D(
            [],
            [],
            color=row["color"],
            marker="o",
            linestyle="None",
            markersize=10,
            label=row["nice_name"],
        )
        for _, row in unique_carriers.iterrows()
    ]

    # Format capacities to show 0 decimal places and add "MW"
    max_capacity_label = f"{max_capacity:.0f} MW"
    min_capacity_label = f"{min_capacity:.0f} MW"
    max_power_label = f"{max_power:.0f} MW"
    min_power_label = f"{min_power:.0f} MW"

    # Create custom legend items for max and min capacities with blue color for generators
    max_line = mlines.Line2D(
        [], [], color="red", linewidth=max_line_width, label=max_capacity_label
    )
    min_line = mlines.Line2D(
        [], [], color="red", linewidth=min_line_width, label=min_capacity_label
    )
    max_circle = mlines.Line2D(
        [],
        [],
        color="blue",
        marker="o",
        markersize=np.sqrt(max_marker_size),
        linestyle="None",
        label=max_power_label,
    )
    min_circle = mlines.Line2D(
        [],
        [],
        color="blue",
        marker="o",
        markersize=np.sqrt(min_marker_size),
        linestyle="None",
        label=min_power_label,
    )

    # Combine all legend elements into one legend and place in the lower right
    legend_elements += [max_line, min_line, max_circle, min_circle]
    combined_legend = ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=12,
    )
    ax.add_artist(combined_legend)

    # Customize the plot
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax.set_axis_off()
    plt.grid(True)
    plt.show()
    if new_only:
        print("Plotting new grid sized generators and lines")
    else:
        print("Plotting grid sized generators and lines")

    if savefolder:
        savepath = os.path.join(savefolder, "grid_sized_generators_and_lines.png")
        fig.savefig(savepath, bbox_inches="tight")


def plot_sized_generators(
    buses: pd.DataFrame,
    branches: pd.DataFrame,
    generators: pd.DataFrame,
    savefolder=None,
):
    """Plots the network with transmission lines and new generators, including separate legends for generator capacities."""

    # Convert buses DataFrame to a GeoDataFrame
    geometry_buses = gpd.points_from_xy(buses["x"], buses["y"])
    gdf_buses = gpd.GeoDataFrame(buses, geometry=geometry_buses, crs="EPSG:4326")

    # Create a dictionary to map bus IDs to their coordinates for easy access
    bus_coords = gdf_buses[["x", "y"]].to_dict("index")

    # Generate list of line segments for LineCollection based on bus coordinates
    lines = []
    linewidth = 2
    for _, row in branches.iterrows():
        if row["bus0"] in bus_coords and row["bus1"] in bus_coords:
            point0 = (bus_coords[row["bus0"]]["x"], bus_coords[row["bus0"]]["y"])
            point1 = (bus_coords[row["bus1"]]["x"], bus_coords[row["bus1"]]["y"])
            lines.append([point0, point1])

    fig, ax = plot_background(buses)

    # Plot buses
    gdf_buses.plot(ax=ax, color="black", marker="o", markersize=10, zorder=20)

    # Create and plot the LineCollection for transmission lines with a constant linewidth
    lc = LineCollection(lines, colors="red", linewidths=linewidth, zorder=15)
    ax.add_collection(lc)

    # Plot generators with offsets
    max_power = generators["p_nom"].max()
    min_power = generators["p_nom"].min()
    mid_power = (max_power + min_power) / 2
    min_marker_size = 50  # Minimum marker size
    max_marker_size = 300  # Maximum marker size
    mid_marker_size = (min_marker_size + max_marker_size) / 2
    offset_radius = 0.1  # Base offset distance around the node

    for bus, group in generators.groupby("bus"):
        if bus in bus_coords:
            x, y = bus_coords[bus]["x"], bus_coords[bus]["y"]

            # Group by carrier and calculate cumulative capacity
            carrier_group = (
                group.groupby("carrier")
                .agg({"p_nom": "sum", "color": "first"})
                .reset_index()
            )
            num_carriers = len(carrier_group)

            # Determine offsets for each carrier count configuration
            if num_carriers == 1:
                offsets = [(0, 0)]
            elif num_carriers == 2:
                offsets = [(0, offset_radius), (0, -offset_radius)]
            elif num_carriers == 3:
                offsets = [
                    (0, offset_radius),
                    (-offset_radius / 2, -offset_radius / 2),
                    (offset_radius / 2, -offset_radius / 2),
                ]
            else:
                angles = np.linspace(0, 2 * np.pi, num_carriers, endpoint=False)
                offsets = [
                    (offset_radius * np.cos(a), offset_radius * np.sin(a))
                    for a in angles
                ]

            # Plot each carrier group with offset and cumulative size
            for (dx, dy), (_, carrier) in zip(offsets, carrier_group.iterrows()):
                cum_capacity = carrier["p_nom"]
                marker_size = min_marker_size + (cum_capacity - min_power) / (
                    max_power - min_power
                ) * (max_marker_size - min_marker_size)
                ax.scatter(
                    x + dx, y + dy, color=carrier["color"], s=marker_size, zorder=25
                )

    # Create legend for generator types and transmission lines
    unique_carriers = generators[["carrier", "color", "nice_name"]].drop_duplicates()
    generator_legend = [
        mlines.Line2D(
            [],
            [],
            color=row["color"],
            marker="o",
            linestyle="None",
            markersize=10,
            label=row["nice_name"],
        )
        for _, row in unique_carriers.iterrows()
    ]

    line_legend = [
        mlines.Line2D(
            [], [], color="red", linewidth=linewidth, label="Transmission Lines"
        )
    ]

    combined_legend = generator_legend + line_legend
    legend1 = ax.legend(handles=combined_legend, loc="lower right", fontsize=12)
    ax.add_artist(legend1)

    capacity_legend = [
        mlines.Line2D(
            [],
            [],
            color="gray",
            marker="o",
            markerfacecolor="lightgray",
            markeredgecolor="black",
            markersize=np.sqrt(min_marker_size),
            linestyle="None",
            label=f"{min_power:.0f} MW",
        ),
        mlines.Line2D(
            [],
            [],
            color="gray",
            marker="o",
            markerfacecolor="lightgray",
            markeredgecolor="black",
            markersize=np.sqrt(mid_marker_size),
            linestyle="None",
            label=f"{mid_power:.0f} MW",
        ),
        mlines.Line2D(
            [],
            [],
            color="gray",
            marker="o",
            markerfacecolor="lightgray",
            markeredgecolor="black",
            markersize=np.sqrt(max_marker_size),
            linestyle="None",
            label=f"{max_power:.0f} MW",
        ),
    ]

    legend2 = ax.legend(
        handles=capacity_legend,
        loc="upper right",
        fontsize=12,
        title="Generator Capacities",
    )
    ax.add_artist(legend2)

    # Add text labels for major cities
    for node, city in node_to_city.items():
        if node in buses.index:
            plt.text(
                buses.loc[node, "x"],
                buses.loc[node, "y"] + 0.3,
                city,
                fontsize=14,
                ha="center",
                va="center",
                weight="bold",
                color="black",
                zorder=25,
            )
        else:
            print(f"Excluded city with missing coordinates: {city}")

    # Customize the plot
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax.set_axis_off()
    plt.grid(True)
    plt.show()
    print("Plotting grid sized generators")
    if savefolder:
        savepath = os.path.join(savefolder, "grid_sized_generators.png")
        fig.savefig(savepath, bbox_inches="tight")


def plot_sized_branches(buses: pd.DataFrame, branches: pd.DataFrame, savefolder=None):
    """Plots the base network with buses and transmission lines, without generators."""

    # Convert buses DataFrame to a GeoDataFrame
    geometry_buses = gpd.points_from_xy(buses["x"], buses["y"])
    gdf_buses = gpd.GeoDataFrame(buses, geometry=geometry_buses, crs="EPSG:4326")

    # Create a dictionary to map bus IDs to their coordinates for easy access
    bus_coords = gdf_buses[["x", "y"]].to_dict("index")

    # Generate list of line segments for LineCollection based on bus coordinates
    lines = []
    line_widths = []
    max_capacity = branches["p_max"].max()
    min_capacity = branches["p_max"].min()
    min_line_width = 1
    max_line_width = 5

    for _, row in branches.iterrows():
        if row["bus0"] in bus_coords and row["bus1"] in bus_coords:
            point0 = (bus_coords[row["bus0"]]["x"], bus_coords[row["bus0"]]["y"])
            point1 = (bus_coords[row["bus1"]]["x"], bus_coords[row["bus1"]]["y"])
            lines.append([point0, point1])

            # Normalize p_max between min_line_width and max_line_width
            divide_by = (max_capacity - min_capacity) * (
                max_line_width - min_line_width
            )
            divide_by = divide_by if divide_by != 0 else 1
            linewidth = min_line_width + (row["p_max"] - min_capacity) / divide_by
            line_widths.append(linewidth)
        else:
            print(f"Excluded line with missing bus coordinates: {row}")

    fig, ax = plot_background(buses)

    # Create and plot the LineCollection for transmission lines
    lc = LineCollection(lines, colors="red", linewidths=line_widths, zorder=15)
    ax.add_collection(lc)

    # Plot buses
    gdf_buses.plot(
        ax=ax, color="black", marker="o", markersize=50, zorder=20
    )  # No legend entry for buses

    # Create legend entries for line capacities
    max_capacity_label = f"{max_capacity:.0f} MW"
    min_capacity_label = f"{min_capacity:.0f} MW"

    # Custom legend items for max and min capacities
    max_line = mlines.Line2D(
        [], [], color="red", linewidth=max_line_width, label=max_capacity_label
    )
    min_line = mlines.Line2D(
        [], [], color="red", linewidth=min_line_width, label=min_capacity_label
    )
    bus_legend = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        markersize=10,
        label="Bus",
    )

    # Combine all legend elements into one legend and place in the lower right
    legend_elements = [max_line, min_line, bus_legend]
    combined_legend = ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=12,
    )
    ax.add_artist(combined_legend)

    # Customize the plot
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax.set_axis_off()
    plt.grid(True)
    plt.show()

    print("Plotting grid sized branches")

    if savefolder:
        savepath = os.path.join(savefolder, "grid_lines_with_sizes.png")
        fig.savefig(savepath, bbox_inches="tight")


def plot_sized_lines_with_extensions(
    buses, branches, new_capacity=5000, savefolder=None, divider=48
):
    """Plots the base network with buses and transmission lines, without generators."""

    # Convert buses DataFrame to a GeoDataFrame
    geometry_buses = gpd.points_from_xy(buses["x"], buses["y"])
    gdf_buses = gpd.GeoDataFrame(buses, geometry=geometry_buses, crs="EPSG:4326")

    # Create a dictionary to map bus IDs to their coordinates for easy access
    bus_coords = gdf_buses[["x", "y"]].to_dict("index")

    # Generate list of line segments for LineCollection based on bus coordinates
    lines = []
    line_widths = []
    max_capacity = branches["p_max"].max()
    min_capacity = branches["p_max"].min()
    min_line_width = 1
    max_line_width = 5
    divide_by = (max_capacity - min_capacity) * (max_line_width - min_line_width)
    divide_by = divide_by if divide_by != 0 else 1
    new_line_width = min_line_width + new_capacity / divide_by
    new_lines = []
    new_line_widths = []

    for _, row in branches.iterrows():
        if row["bus0"] in bus_coords and row["bus1"] in bus_coords:
            point0 = (bus_coords[row["bus0"]]["x"], bus_coords[row["bus0"]]["y"])
            point1 = (bus_coords[row["bus1"]]["x"], bus_coords[row["bus1"]]["y"])
            lines.append([point0, point1])

            # Normalize p_max between min_line_width and max_line_width
            divide_by = (max_capacity - min_capacity) * (
                max_line_width - min_line_width
            )
            divide_by = divide_by if divide_by != 0 else 1
            linewidth = min_line_width + (row["p_max"] - min_capacity) / divide_by
            line_widths.append(linewidth)

            perpendicular_vector = np.array(
                [point1[1] - point0[1], point0[0] - point1[0]]
            )
            perpendicular_vector = perpendicular_vector / np.linalg.norm(
                perpendicular_vector
            )
            # Check if the line is going from left to right
            if perpendicular_vector[0] > 0:
                perpendicular_vector = (
                    -perpendicular_vector
                )  # We want it pointing to the left
            # Check if the line is pointing up
            if perpendicular_vector[1] > 0:
                perpendicular_vector = -perpendicular_vector  # We want it pointing down
            new_point0 = (
                np.array(point0)
                + (new_line_width / 2 + linewidth / 2) * perpendicular_vector / divider
            )
            new_point1 = (
                np.array(point1)
                + (new_line_width / 2 + linewidth / 2) * perpendicular_vector / divider
            )
            new_lines.append([new_point0, new_point1])
            new_line_widths.append(new_line_width)
        else:
            print(f"Excluded line with missing bus coordinates: {row}")

    fig, ax = plot_background(buses)

    # Create and plot the LineCollection for transmission lines
    lc = LineCollection(lines, colors="red", linewidths=line_widths, zorder=15)
    ax.add_collection(lc)

    # Create and plot the LineCollection for new lines with additional capacity
    lc_new = LineCollection(
        new_lines, colors="indigo", linewidths=new_line_widths, zorder=18
    )
    ax.add_collection(lc_new)

    # Plot buses with larger marker size
    bus_marker_size = 50  # Increased marker size for buses
    gdf_buses.plot(
        ax=ax, color="black", marker="o", markersize=bus_marker_size, zorder=20
    )

    # Create legend entries for line capacities
    max_capacity_label = f"{max_capacity:.0f} MW"
    min_capacity_label = f"{min_capacity:.0f} MW"
    new_capacity_label = f"{new_capacity} MW (New)"

    # Custom legend items for actual and new line capacities
    max_line = mlines.Line2D(
        [], [], color="red", linewidth=max_line_width, label=max_capacity_label
    )
    min_line = mlines.Line2D(
        [], [], color="red", linewidth=min_line_width, label=min_capacity_label
    )
    new_line_legend = mlines.Line2D(
        [], [], color="indigo", linewidth=new_line_width, label=new_capacity_label
    )
    bus_legend = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        markersize=10,
        label="Bus",
    )

    # Combine all legend elements into one legend and place in the lower right
    legend_elements = [max_line, min_line, new_line_legend, bus_legend]
    combined_legend = ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=12,
    )
    ax.add_artist(combined_legend)

    # Customize the plot
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax.set_axis_off()
    plt.grid(True)
    plt.show()
    print(f"Transmission lines with additional capacity potential of {new_capacity} MW")
    if savefolder:
        savepath = os.path.join(savefolder, "grid_with_line_sizes_and_extensions.png")
        fig.savefig(savepath, bbox_inches="tight")


def plot_installed_capacity_bar_chart(
    nodes: pd.DataFrame, generators: pd.DataFrame, savefolder=None
):
    # Add a 'city' column to the generators DataFrame based on the 'bus' mapping
    generators = generators.copy()
    generators[["x", "y"]] = nodes.loc[generators["bus"], ["x", "y"]].values
    generators["city_or_country"] = nodes.loc[generators["bus"], "country"].values

    # Create a mask for rows where (x, y) exists in the dictionary
    mask = generators.apply(
        lambda row: (row["x"], row["y"]) in coordinates_to_city_or_country, axis=1
    )

    # Overwrite only the matching rows
    generators.loc[mask, "city_or_country"] = generators.loc[mask].apply(
        lambda row: coordinates_to_city_or_country[(row["x"], row["y"])], axis=1
    )
    # Convert p_nom from MW to GW
    generators["p_nom_gw"] = generators["p_nom"] / 1000

    # Define the desired order for generator types
    generator_order = _get_generator_order(generators)

    # Create a mapping of carriers to their colors and nice names
    carrier_colors = (
        generators.drop_duplicates("carrier").set_index("carrier")["color"].to_dict()
    )
    carrier_nice_names = (
        generators.drop_duplicates("carrier")
        .set_index("carrier")["nice_name"]
        .to_dict()
    )

    # Group by city and carrier, summing the installed capacity (p_nom_gw)
    city_generator_capacity = (
        generators.groupby(["city_or_country", "carrier"])["p_nom_gw"]
        .sum()
        .unstack(fill_value=0)
    )
    city_generator_capacity = city_generator_capacity[generator_order]

    # Plot the stacked bar chart
    plt.figure(figsize=(18, 9))
    city_generator_capacity.plot(
        kind="bar",
        stacked=True,
        color=[carrier_colors[carrier] for carrier in city_generator_capacity.columns],
        figsize=(18, 9),
    )

    # Customize the plot appearance
    plt.ylabel("Installed Capacity (GW)", fontsize=20)
    plt.xticks(rotation=45, fontsize=18)
    if len(city_generator_capacity) > 15:
        plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=16)
    ncol = len(generator_order) // 2
    ncol = ncol if ncol > 1 else 1
    plt.legend(
        [carrier_nice_names[carrier] for carrier in city_generator_capacity.columns],
        loc="upper center",
        fontsize=18,
        framealpha=1,
        bbox_to_anchor=(0.5, -0.16),
        ncol=ncol,
    )
    plt.tight_layout()
    print("Generators by city and type installed capacity (nominal) in GW")
    if savefolder:
        savepath = os.path.join(savefolder, "bar_generators_installed_capacity.png")
        plt.savefig(savepath, bbox_inches="tight")

    # Show the plot
    plt.show()


def plot_effective_capacity_generators_bar_chart(
    nodes: pd.DataFrame,
    generators: pd.DataFrame,
    capacity_factors: pd.DataFrame,
    savefolder=None,
):
    generators = generators.copy()
    generators["capacity_factor"] = np.zeros(generators.shape[0])
    for generator in generators.index:
        if generator in capacity_factors.columns:
            generators.loc[generator, "capacity_factor"] = capacity_factors[
                generator
            ].mean()
        else:
            generators.loc[generator, "capacity_factor"] = 1
    generators["effective_capacity"] = (
        generators["p_nom"] * generators["capacity_factor"]
    )
    weighted_capacity_factor = (
        generators.groupby("carrier")["effective_capacity"].sum()
        / generators.groupby("carrier")["p_nom"].sum()
    )

    generators[["x", "y"]] = nodes.loc[generators["bus"], ["x", "y"]].values
    generators["city_or_country"] = nodes.loc[generators["bus"], "country"].values

    # Create a mask for rows where (x, y) exists in the dictionary
    mask = generators.apply(
        lambda row: (row["x"], row["y"]) in coordinates_to_city_or_country, axis=1
    )

    # Overwrite only the matching rows
    generators.loc[mask, "city_or_country"] = generators.loc[mask].apply(
        lambda row: coordinates_to_city_or_country[(row["x"], row["y"])], axis=1
    )

    # Convert effective_capacity from MW to GW
    generators["effective_capacity_gw"] = generators["effective_capacity"] / 1000

    # Define the desired order for generator types
    generator_order = _get_generator_order(generators)

    # Create a mapping of carriers to their colors and nice names
    carrier_colors = (
        generators.drop_duplicates("carrier").set_index("carrier")["color"].to_dict()
    )
    carrier_nice_names = (
        generators.drop_duplicates("carrier")
        .set_index("carrier")["nice_name"]
        .to_dict()
    )

    # Group by city and carrier, summing the effective capacity (effective_capacity_gw)
    city_generator_effective_capacity = (
        generators.groupby(["city_or_country", "carrier"])["effective_capacity_gw"]
        .sum()
        .unstack(fill_value=0)
    )

    # Reorder the columns based on the specified generator order
    city_generator_effective_capacity = city_generator_effective_capacity[
        generator_order
    ]

    # Plot the stacked bar chart
    plt.figure(figsize=(18, 9))
    city_generator_effective_capacity.plot(
        kind="bar",
        stacked=True,
        color=[
            carrier_colors[carrier]
            for carrier in city_generator_effective_capacity.columns
        ],
        figsize=(18, 9),
    )

    # Customize the plot appearance
    plt.ylabel("Effective Capacity (GW)", fontsize=20)
    if len(city_generator_effective_capacity) > 15:
        plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=16)
    ncol = len(generator_order) // 2
    ncol = ncol if ncol > 1 else 1
    plt.legend(
        [
            carrier_nice_names[carrier]
            for carrier in city_generator_effective_capacity.columns
        ],
        loc="upper center",
        fontsize=18,
        framealpha=1,
        bbox_to_anchor=(0.5, -0.16),
        ncol=ncol,
    )
    plt.tight_layout()
    print("Generators by city and type effective capacity in GW")
    if savefolder:
        savepath = os.path.join(savefolder, "bar_effective_capacity_generators.png")
        plt.savefig(savepath, bbox_inches="tight")

    # Show the plot
    plt.show()


def plot_demand_network_hourly(
    nodes: pd.DataFrame,
    hourly_demand: pd.DataFrame,
    n_largest: int = 5,
    savefolder=None,
):
    """Plot the network with buses colored and sized depending on the demand."""
    # Calculate the average daily demand for each area
    average_hourly_demand = hourly_demand.mean()

    # Create a GeoDataFrame for plotting
    buses = nodes
    buses["average_demand"] = (
        average_hourly_demand  # Add average demand data to buses DataFrame
    )
    geometry_buses = gpd.points_from_xy(buses["x"], buses["y"])
    gdf_buses = gpd.GeoDataFrame(buses, geometry=geometry_buses, crs="EPSG:4326")

    fig, ax = plot_background(buses)

    # Plot buses with size and color based on average demand, without automatic legend
    gdf_buses.plot(
        ax=ax,
        marker="o",
        column="average_demand",
        cmap="bwr",
        markersize=gdf_buses["average_demand"]
        / gdf_buses["average_demand"].max()
        * 600,  # Scale marker size
    )

    # Get the indices of the 5 nodes with the highest average demand
    five_biggest_demand_nodes = (
        buses["average_demand"].nlargest(n_largest).index.tolist()
    )

    # Add text labels for major cities
    for node in five_biggest_demand_nodes:
        lon = buses.loc[node, "x"]
        y_offset = 0.3
        lat = buses.loc[node, "y"] + y_offset
        if node in node_to_city:
            city = node_to_city[node]
            plt.text(
                lon,
                lat,
                city,
                fontsize=12,
                ha="center",
                va="center",
                weight="bold",
                color="black",
            )

    # Customize the plot
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax.set_axis_off()  # Remove axis lines

    # Adding a color bar to represent the demand scale
    sm = plt.cm.ScalarMappable(
        cmap="bwr",
        norm=plt.Normalize(
            vmin=gdf_buses["average_demand"].min(),
            vmax=gdf_buses["average_demand"].max(),
        ),
    )
    sm.set_array([])

    # Position the color bar on the right side with larger font size
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, orientation="vertical")
    cbar.set_label(
        "Average Hourly Demand (MW)", fontsize=14
    )  # Increase label font size
    cbar.ax.tick_params(labelsize=12)  # Increase tick font size

    # Tighten the layout to reduce whitespace
    plt.tight_layout()
    print("Average hourly demand [MWh/h] per area")
    plt.show()

    # Save the figure if a save path is provided
    if savefolder:
        savepath = os.path.join(savefolder, "grid_demand_network_hourly.png")
        fig.savefig(savepath, bbox_inches="tight")


def plot_base_network_with_lineIDs_and_city_text(
    buses, branches, savefolder=None, divider=48, text_offset=15
):
    """Plots the base network with buses and transmission lines, without generators."""

    # Convert buses DataFrame to a GeoDataFrame
    geometry_buses = gpd.points_from_xy(buses["x"], buses["y"])
    gdf_buses = gpd.GeoDataFrame(buses, geometry=geometry_buses, crs="EPSG:4326")

    # Create a dictionary to map bus IDs to their coordinates for easy access
    bus_coords = gdf_buses[["x", "y"]].to_dict("index")

    # Generate list of line segments for LineCollection based on bus coordinates
    lines = []
    line_widths = []
    max_capacity = branches["p_max"].max()
    min_capacity = branches["p_max"].min()
    min_line_width = 1
    max_line_width = 5
    line_midpoints = []

    for index, row in branches.iterrows():
        if row["bus0"] in bus_coords and row["bus1"] in bus_coords:
            point0 = (bus_coords[row["bus0"]]["x"], bus_coords[row["bus0"]]["y"])
            point1 = (bus_coords[row["bus1"]]["x"], bus_coords[row["bus1"]]["y"])
            lines.append([point0, point1])

            # Calculate the midpoint for adding line numbers later
            midpoint = (
                (point0[0] + point1[0]) / 2,
                (point0[1] + point1[1]) / 2,
            )

            # Calculate the perpendicular vector for the line
            perpendicular_vector = np.array(
                [point1[1] - point0[1], point0[0] - point1[0]]
            )
            perpendicular_vector = perpendicular_vector / np.linalg.norm(
                perpendicular_vector
            )

            # Adjust perpendicular vector to point consistently
            if perpendicular_vector[0] > 0:
                perpendicular_vector = -perpendicular_vector
            if perpendicular_vector[1] > 0:
                perpendicular_vector = -perpendicular_vector

            # Offset the midpoint by the perpendicular vector for text placement
            offset_midpoint = (
                midpoint[0] + text_offset * perpendicular_vector[0] / divider,
                midpoint[1] + text_offset * perpendicular_vector[1] / divider,
            )
            line_midpoints.append(
                (offset_midpoint, index)
            )  # Store offset midpoint and line number

            # Normalize p_max between min_line_width and max_line_width
            divide_by = (max_capacity - min_capacity) * (
                max_line_width - min_line_width
            )
            divide_by = divide_by if divide_by != 0 else 1
            linewidth = min_line_width + (row["p_max"] - min_capacity) / divide_by
            line_widths.append(linewidth)
        else:
            print(f"Excluded line with missing bus coordinates: {row}")

    fig, ax = plot_background(buses)

    # Create and plot the LineCollection for transmission lines
    lc = LineCollection(lines, colors="red", linewidths=line_widths, zorder=15)
    ax.add_collection(lc)

    # Plot buses with larger marker size
    bus_marker_size = 50  # Increased marker size for buses
    gdf_buses.plot(
        ax=ax, color="black", marker="o", markersize=bus_marker_size, zorder=20
    )

    # Add text labels for major cities
    for node, city in node_to_city.items():
        if node in buses:
            plt.text(
                buses.loc[node, "x"],
                buses.loc[node, "y"] + 0.2,
                city,
                fontsize=14,
                ha="center",
                va="center",
                weight="bold",
                color="black",
                zorder=25,
            )
        else:
            print(f"Excluded city with missing bus coordinates: {city}")

    # Add text labels for line numbers at the offset positions
    for offset_midpoint, line_number in line_midpoints:
        plt.text(
            offset_midpoint[0],
            offset_midpoint[1],
            str(line_number),
            fontsize=10,
            ha="center",
            va="center",
            color="blue",
            zorder=30,
        )

    # Create legend entries for line capacities
    max_capacity_label = f"{max_capacity:.0f} MW"
    min_capacity_label = f"{min_capacity:.0f} MW"

    # Custom legend items for actual and new line capacities
    max_line = mlines.Line2D(
        [], [], color="red", linewidth=max_line_width, label=max_capacity_label
    )
    min_line = mlines.Line2D(
        [], [], color="red", linewidth=min_line_width, label=min_capacity_label
    )
    bus_legend = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        markersize=10,
        label="Nodes",
    )

    # Combine all legend elements into one legend and place in the lower right
    legend_elements = [max_line, min_line, bus_legend]
    combined_legend = ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=12,
    )
    ax.add_artist(combined_legend)

    # Customize the plot
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax.set_axis_off()
    plt.grid(True)
    print("Plotting grid with line IDs and city text")
    plt.show()

    if savefolder:
        savepath = os.path.join(savefolder, "grid_with_ids.png")
        fig.savefig(savepath, bbox_inches="tight")


def get_closest_node_to_city(city: str, nodes: pd.DataFrame) -> str:
    """Find the node closest to the specified city."""
    # Get the coordinates of the specified city
    city_coords = city_or_country_to_coordinates[city]

    return get_closest_node_to_coordinates(city_coords, nodes)


def get_closest_node_to_coordinates(coords: tuple, nodes: pd.DataFrame) -> str:
    """Find the node closest to the specified coordinates."""
    # Calculate the Euclidean distance between the coordinates and each node
    nodes["distance"] = np.sqrt(
        (nodes["x"] - coords[0]) ** 2 + (nodes["y"] - coords[1]) ** 2
    )

    # Find the node with the smallest distance to the coordinates
    closest_node = nodes["distance"].idxmin()

    return closest_node


def get_closest_city_to_node(node: str, nodes: pd.DataFrame) -> str:
    """Find the city closest to the specified node."""
    # Get the coordinates of the specified node
    node_coords = (nodes.loc[node, "x"], nodes.loc[node, "y"])

    nodes = nodes.copy()
    nodes["distance"] = np.sqrt(
        (nodes["x"] - node_coords[0]) ** 2 + (nodes["y"] - node_coords[1]) ** 2
    )
    closest_node = nodes["distance"].idxmin()
    return node_to_city[closest_node]


def plot_average_hourly_demand_each_month_at_node(
    hourly_demand: pd.DataFrame, node: str, savefolder: str = None
):
    """Plot the average hourly demand for each month for a specific node."""

    node_timeseries = hourly_demand[node]

    # Calculate monthly average for each hour
    monthly_hourly_avg = (
        node_timeseries.groupby(
            [node_timeseries.index.month, node_timeseries.index.hour]
        )
        .mean()
        .unstack(level=0)
    )

    # Create month names for the legend
    month_names = [calendar.month_name[i] for i in range(1, 13)]

    # Plot for average hourly load per month with improved labels and font sizes
    plt.figure(figsize=(12, 6))

    # Plot each month's average hourly load
    for month in range(1, 13):
        plt.plot(
            monthly_hourly_avg.index,
            monthly_hourly_avg[month],
            label=month_names[month - 1],
        )

    # Customize the plot appearance
    plt.xlabel("Hour of Day", fontsize=14)
    plt.ylabel("Average Load (MW)", fontsize=14)
    plt.xticks(
        range(0, 24), fontsize=12
    )  # x-axis only from 0 to 23 without extra space
    plt.yticks(fontsize=12)  # Increase y-axis font size
    plt.xlim(0, 23)  # Set x-axis limits to remove whitespace
    plt.legend(
        title="Month", loc="upper left", fontsize=12, title_fontsize=14
    )  # Adjust legend position and font sizes
    plt.grid(True, linestyle="--", alpha=0.5)
    if node in node_to_city:
        city_name = node_to_city[node]
    else:
        city_name = "Unknown"
    print(f"Average hourly load for {city_name} aka {node}:")

    if savefolder:
        savepath = os.path.join(
            savefolder, f"plot_average_hourly_demand_each_month_{node}.png"
        )
        plt.savefig(savepath, bbox_inches="tight")

    plt.show()


def plot_average_hourly_demand_each_month_aggragated(
    hourly_demand: pd.DataFrame, savefolder: str = None
) -> None:
    """Plot the average hourly demand for each month for all demand nodes aggregated."""
    total_load_timeseries = hourly_demand.sum(axis=1)

    # Calculate monthly average for each hour
    monthly_hourly_avg_GWh = (
        total_load_timeseries.groupby(
            [total_load_timeseries.index.month, total_load_timeseries.index.hour]
        )
        .mean()
        .unstack(level=0)
    ) / 1e3  # Convert from MW to GW

    # Create month names for the legend
    month_names = [calendar.month_name[i] for i in range(1, 13)]

    # Plot for average hourly load per month with improved labels and font sizes
    plt.figure(figsize=(12, 6))

    # Plot each month's average hourly load
    for month in range(1, 13):
        plt.plot(
            monthly_hourly_avg_GWh.index,
            monthly_hourly_avg_GWh[month],
            label=month_names[month - 1],
        )

    # Customize the plot appearance
    plt.xlabel("Hour of Day", fontsize=14)
    plt.ylabel("Average Load (GWh)", fontsize=14)
    plt.xticks(
        range(0, 24), fontsize=12
    )  # x-axis only from 0 to 23 without extra space
    plt.yticks(fontsize=12)  # Increase y-axis font size
    plt.xlim(0, 23)  # Set x-axis limits to remove whitespace
    plt.legend(
        title="Month", loc="upper left", fontsize=12, title_fontsize=14
    )  # Adjust legend position and font sizes
    plt.grid(True, linestyle="--", alpha=0.5)

    print("Average hourly load per month all demand nodes")

    if savefolder:
        savepath = os.path.join(
            savefolder, "plot_average_hourly_demand_each_month_aggregated.png"
        )
        plt.savefig(savepath, bbox_inches="tight")

    plt.show()


def plot_average_hourly_demand_each_season_aggragated(
    hourly_demand: pd.DataFrame, savefolder: str = None
):
    """Plot the average hourly demand for each season for all demand nodes aggregated."""
    seasons = {
        "Winter": [12, 1, 2],  # December, January, February
        "Spring": [3, 4, 5],  # March, April, May
        "Summer": [6, 7, 8],  # June, July, August
        "Autumn": [9, 10, 11],  # September, October, November
    }

    # Sum the load across all nodes for each timestamp
    total_load_timeseries_GWh = hourly_demand.sum(axis=1) / 1e3

    # Create a DataFrame to store the seasonal hourly average
    seasonal_hourly_avg = pd.DataFrame(index=range(24), columns=seasons.keys())

    # Calculate the average load for each season for each hour
    for season, months in seasons.items():
        # Filter for the relevant months, group by hour within each season, and calculate the mean
        season_data = total_load_timeseries_GWh[
            total_load_timeseries_GWh.index.month.isin(months)
        ]
        seasonal_hourly_avg[season] = season_data.groupby(season_data.index.hour).mean()

    # Plot the average hourly load by season
    plt.figure(figsize=(12, 6))

    # Plot each season's average hourly load
    for season in seasons.keys():
        plt.plot(seasonal_hourly_avg.index, seasonal_hourly_avg[season], label=season)

    # Customize the plot appearance
    plt.xlabel("Hour of Day", fontsize=14)
    plt.ylabel("Average Load (GWh)", fontsize=14)
    plt.xticks(
        range(0, 24), fontsize=12
    )  # x-axis only from 0 to 23 without extra space
    plt.yticks(fontsize=12)  # Increase y-axis font size
    plt.xlim(0, 23)  # Set x-axis limits to remove whitespace
    plt.legend(title="Season", loc="upper left", fontsize=12, title_fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)

    print("Average hourly load per season all demand nodes")
    if savefolder:
        savepath = os.path.join(
            savefolder, "plot_average_hourly_demand_each_season_aggregated.png"
        )
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_demand_network_daily(
    nodes: pd.DataFrame, hourly_demand: pd.DataFrame, savefolder: str = None
) -> None:
    """Plot the network with buses colored and sized depending on the demand. Aggregated to daily demand in GWh."""
    # Calculate the average daily demand for each area
    average_daily_demand_in_GWh = (
        hourly_demand.mean() * 24 / 1e3
    )  # Average daily demand in GWh

    # Create a GeoDataFrame for plotting
    buses = nodes.copy()
    buses["average_demand"] = (
        average_daily_demand_in_GWh  # Add average demand data to buses DataFrame
    )
    geometry_buses = gpd.points_from_xy(buses["x"], buses["y"])
    gdf_buses = gpd.GeoDataFrame(buses, geometry=geometry_buses, crs="EPSG:4326")
    fig, ax = plot_background(buses)

    # Add text labels for major cities
    for node, city in node_to_city.items():
        if node in nodes.index:
            plt.text(
                nodes.loc[node, "x"],
                nodes.loc[node, "y"] + 0.3,
                city,
                fontsize=12,
                ha="center",
                va="center",
                weight="bold",
                color="black",
            )
        else:
            print(f"Excluded city with missing bus coordinates: {city}")

    # Plot buses with size and color based on average demand, without automatic legend
    gdf_buses.plot(
        ax=ax,
        marker="o",
        column="average_demand",
        cmap="bwr",
        markersize=gdf_buses["average_demand"]
        / gdf_buses["average_demand"].max()
        * 600,  # Increase scale for marker size
    )

    # Add text labels for each bus/area at their respective coordinates
    # for idx, row in gdf_buses.iterrows():
    #     plt.text(
    #         row.geometry.x, row.geometry.y, idx, fontsize=8, ha="center", color="black"
    #     )

    # Customize the plot
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax.set_axis_off()  # Remove axis lines

    # Adding color bar to represent demand scale on the right with larger label font size
    sm = plt.cm.ScalarMappable(
        cmap="bwr",
        norm=plt.Normalize(
            vmin=gdf_buses["average_demand"].min(),
            vmax=gdf_buses["average_demand"].max(),
        ),
    )
    sm.set_array([])
    # Position the color bar on the right side with larger font size
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, orientation="vertical")
    cbar.set_label(
        "Average Daily Demand (GWh)", fontsize=14
    )  # Increase label font size
    cbar.ax.tick_params(labelsize=12)  # Increase tick font size

    print("Average daily demand [GWh per day] per area")

    if savefolder:
        savepath = os.path.join(savefolder, "grid_demand_daily.png")
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_aggregated_average_hourly_demand_with_stds(
    hourly_demand: pd.DataFrame, savefolder: str = None
) -> None:
    """Plot the aggregated average hourly demand with standard deviations."""
    # Calculate total load across all nodes for each timestamp
    total_load_timeseries = hourly_demand.sum(axis=1) / 1e3  # Convert to GWh

    # Group by hour of the day and calculate the mean and standard deviation
    hourly_stats = total_load_timeseries.groupby(total_load_timeseries.index.hour).agg(
        ["mean", "std"]
    )

    # Extract the mean and standard deviation
    mean_demand = hourly_stats["mean"]
    std_demand = hourly_stats["std"]

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot the mean demand
    plt.plot(
        mean_demand.index,
        mean_demand,
        label="Average Hourly Demand",
        color="blue",
        linewidth=2,
    )

    # Plot the shaded areas for ±1 and ±2 standard deviations
    plt.fill_between(
        mean_demand.index,
        mean_demand - std_demand,
        mean_demand + std_demand,
        color="blue",
        alpha=0.2,
        label="±1 Std Dev",
    )
    plt.fill_between(
        mean_demand.index,
        mean_demand - 2 * std_demand,
        mean_demand + 2 * std_demand,
        color="blue",
        alpha=0.1,
        label="±2 Std Dev",
    )

    # Customize the plot appearance
    plt.xlabel("Hour of Day", fontsize=16)
    plt.ylabel("Total Load (GWh)", fontsize=16)
    plt.xticks(range(0, 24), fontsize=14)  # x-axis only from 0 to 23
    plt.yticks(fontsize=14)  # Increase y-axis font size
    plt.xlim(0, 23)  # Set x-axis limits to remove whitespace
    print("Average Hourly Demand with Standard Deviation")
    plt.legend(fontsize=14, loc="upper left")  # Adjust legend position and font sizes
    plt.grid(True, linestyle="--", alpha=0.5)

    if savefolder:
        savepath = os.path.join(
            savefolder, "plot_aggregated_average_hourly_demand_with_stds.png"
        )
        plt.savefig(savepath, bbox_inches="tight")

    # Show the plot
    plt.show()


def plot_normalized_hourly_load_by_country(
    buses: pd.DataFrame, hourly_demand: pd.DataFrame, savefolder: str = None
) -> None:
    # Filter Spain (ES) and Portugal (PT) columns
    spain_columns = [col for col in hourly_demand.columns if col.startswith("ES")]
    portugal_columns = [col for col in hourly_demand.columns if col.startswith("PT")]
    country_codes = buses["country"].unique()
    country_columns = {
        country_code: [
            col for col in hourly_demand.columns if col.startswith(country_code)
        ]
        for country_code in country_codes
    }
    # Calculate the total load for Spain and Portugal for each timestamp
    country_load = {
        country_code: hourly_demand[columns].sum(axis=1)
        for country_code, columns in country_columns.items()
    }
    country_load
    # Create a DataFrame with the total load for Spain and Portugal
    total_load = pd.DataFrame(country_load)
    # Group by the hour of the day and calculate the average load
    average_hourly_load = total_load.groupby(total_load.index.hour).mean()

    # Normalize the average hourly loads (min-max normalization)
    normalized_hourly_load = (average_hourly_load - average_hourly_load.min()) / (
        average_hourly_load.max() - average_hourly_load.min()
    )

    # Plot the normalized average hourly load for Spain and Portugal
    plt.figure(figsize=(12, 6))
    for country in normalized_hourly_load.columns:
        plt.plot(
            normalized_hourly_load.index,
            normalized_hourly_load[country],
            label=country,
            linewidth=2,
        )

    # Customize the plot
    plt.xlabel("Hour of Day", fontsize=16)
    plt.ylabel("Normalized Load", fontsize=16)
    plt.xticks(range(0, 24), fontsize=14)  # x-axis only from 0 to 23
    plt.yticks(fontsize=14)  # Increase y-axis font size
    print("Normalized Average Hourly Load for per country")
    plt.legend(fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)

    if savefolder:
        savepath = os.path.join(savefolder, "plot_normalized_demand_hourly_country.png")
        plt.savefig(savepath, bbox_inches="tight")
    # Show the plot
    plt.show()


### Battery Plots ###


def plot_battery_average_hourly_soc_per_battery(
    battery_soc: pd.DataFrame, savefolder: str = None
) -> None:
    if isinstance(battery_soc, pd.DataFrame):
        # Extract the hour of the day from the index and group by hour
        battery_soc["hour"] = battery_soc.index.hour
        average_hourly_soc = battery_soc.groupby("hour").mean()

        # Plot the average hourly SOC for each battery
        plt.figure(figsize=(12, 6))
        for column in battery_soc.columns[:-1]:  # Exclude the 'hour' column
            plt.plot(average_hourly_soc.index, average_hourly_soc[column], label=column)

        print("Average Hourly In-Use Pattern (State of Charge) for Each Battery")
        plt.xlabel("Hour of the Day", fontsize=16)
        plt.xticks(range(24), fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel("Average State of Charge (MWh)", fontsize=16)
        plt.legend(title="Battery", fontsize=12, title_fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        if savefolder:
            savepath = os.path.join(
                savefolder, "plot_battery_average_hourly_soc_per_battery.png"
            )
            plt.savefig(savepath, bbox_inches="tight")
        plt.show()


def plot_battery_average_hourly_soc_per_month(
    battery_soc: pd.DataFrame, savefolder: str = None
) -> None:
    if isinstance(battery_soc, pd.DataFrame):

        # Ensure the 'time' index is a datetime if it's not already
        # If `time` is not already a datetime index, uncomment the line below:
        # battery_soc.index = pd.to_datetime(battery_soc.index)

        # Add 'month' and 'hour' columns
        battery_soc["month"] = pd.to_datetime(battery_soc.index).month
        battery_soc["hour"] = pd.to_datetime(battery_soc.index).hour

        # Group by hour and hour, then calculate the mean
        monthly_hourly_soc = battery_soc.groupby(["month", "hour"]).mean()

        # Create 12 plots, one for each month
        for month in range(1, 13):
            plt.figure(figsize=(12, 6))
            subset = monthly_hourly_soc.loc[month]  # Select data for the current month
            for column in battery_soc.columns[
                :-2
            ]:  # Exclude 'month' and 'hour' columns
                plt.plot(subset.index, subset[column], label=column)

            print(f"Average Hourly soc - Month {month}")
            plt.xlabel("Hour of the Day", fontsize=16)
            plt.ylabel("Average State of Charge (MWh)", fontsize=16)
            plt.xticks(range(24), fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(title="Battery", fontsize=12, title_fontsize=14)
            plt.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            if savefolder:
                savepath = os.path.join(
                    savefolder, f"{month}_plot_battery_average_hourly_soc_month_.png"
                )
                plt.savefig(savepath, bbox_inches="tight")
            plt.show()


def plot_battery_average_hourly_soc_by_month_per_battery(
    battery_soc: pd.DataFrame, savefolder: str = None
) -> None:

    if isinstance(battery_soc, pd.DataFrame):
        # Ensure the 'time' index is a datetime if it's not already
        # If `time` is not already a datetime index, uncomment the line below:
        # battery_soc.index = pd.to_datetime(battery_soc.index)

        # Add 'month' and 'hour' columns
        battery_soc["month"] = pd.to_datetime(battery_soc.index).month
        battery_soc["hour"] = pd.to_datetime(battery_soc.index).hour

        # Group by month and hour, then calculate the mean
        monthly_hourly_soc = battery_soc.groupby(["month", "hour"]).mean()

        # Mapping month numbers to names
        month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        # Create a plot for each battery
        for battery in battery_soc.columns[:-2]:  # Exclude 'month' and 'hour' columns
            plt.figure(figsize=(12, 6))
            for month in range(1, 13):
                # Select data for the current month
                subset = monthly_hourly_soc.loc[month]
                plt.plot(subset.index, subset[battery], label=month_names[month - 1])

            print(f"Average Hourly In-Use Pattern for {battery}")
            plt.xlabel("Hour of the Day", fontsize=16)
            plt.xticks(range(24), fontsize=14)
            plt.ylabel("Average State of Charge (MWh)", fontsize=16)
            plt.yticks(fontsize=14)
            plt.legend(title="Month", loc="upper right", fontsize=12, title_fontsize=14)
            plt.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            if savefolder:
                savepath = os.path.join(
                    savefolder,
                    f"{battery}_plot_battery_average_hourly_soc_by_month_.png",
                )
                plt.savefig(savepath, bbox_inches="tight")
            plt.show()


def plot_num_cycles_per_month(
    batteries: pd.DataFrame, battery_discharging: pd.DataFrame, savefolder: str = ""
) -> None:
    if isinstance(battery_discharging, pd.DataFrame):

        # Example: If the `batteries` DataFrame is not indexed by the battery names, ensure it is:
        # batteries.set_index('Battery', inplace=True)

        # Align columns in `battery_discharging` with `batteries` index
        soc_max = batteries["SOC_max"]

        # Add 'month' column to `battery_discharging`
        battery_discharging["month"] = pd.to_datetime(battery_discharging.index).month

        # Calculate cumulative discharge and cycles
        discharge_per_month = battery_discharging.groupby("month").sum()
        monthly_cycles = discharge_per_month.div(soc_max, axis=1)

        month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        monthly_cycles.index = month_names  # Replace month numbers with names

        # Plot the stacked bar chart
        monthly_cycles.plot(kind="bar", stacked=True, figsize=(12, 6))

        print("Number of Cycles per Battery per Month")
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel("Number of Cycles", fontsize=16)
        ncol = len(monthly_cycles.columns) // 3
        ncol = 1 if ncol == 0 else ncol
        plt.legend(
            title="Battery",
            title_fontsize=16,
            loc="upper center",
            fontsize=16,
            ncol=ncol,
        )
        plt.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.5, axis="y")

        if savefolder:
            savepath = os.path.join(savefolder, "plot_num_cycles_per_month.png")
            plt.savefig(savepath, bbox_inches="tight")
        plt.show()


def cake_battery_usage_per_battery_per_month(
    batteries: pd.DataFrame, battery_discharging: pd.DataFrame, savefolder: str = ""
) -> None:

    if isinstance(battery_discharging, pd.DataFrame):

        # Ensure `batteries` is indexed by battery names
        soc_max = batteries["SOC_max"]

        # Add 'month' column to `battery_discharging`
        battery_discharging["month"] = battery_discharging.index.month

        # Calculate cumulative discharge and cycles
        cumulative_discharge = battery_discharging.cumsum()
        cycles = cumulative_discharge / soc_max

        # Calculate the number of cycles per month
        monthly_cycles = cycles.groupby(battery_discharging["month"]).max()
        monthly_cycles.drop(columns=["month"], inplace=True)
        battery_discharging.drop(columns=["month"], inplace=True)

        # Map month numbers to month names
        month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        monthly_cycles.index = month_names  # Replace month numbers with names

        # Create a 3x4 subplot for the pie charts
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        axes = axes.flatten()

        # Shared legend elements
        all_labels = monthly_cycles.columns
        shared_patches = None

        for i, month in enumerate(month_names):
            ax = axes[i]
            data = monthly_cycles.loc[month]

            # Drop NaN values or replace them with 0
            data = data.fillna(0)

            # Skip if all values are zero
            if data.sum() == 0:
                ax.text(0.5, 0.5, "No Data", fontsize=16, ha="center", va="center")
                ax.axis("off")
                continue

            # Create pie chart for the current subplot
            wedges, texts, autotexts = ax.pie(
                data,
                labels=None,  # Disable labels for individual plots
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 10},
            )
            ax.set_title(month, fontsize=14)

            # Store patches for shared legend
            if shared_patches is None:
                shared_patches = wedges

                # Add a shared legend
                ncol = len(all_labels) // 2 + 1
                ncol = 1 if ncol == 0 else ncol
                fig.legend(
                    shared_patches,
                    labels=all_labels,
                    title="Batteries",
                    title_fontsize=16,
                    loc="lower center",
                    fontsize=16,
                    ncol=ncol,
                    bbox_to_anchor=(0.5, -0.05),
                )

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for legend
        if savefolder:
            savepath = os.path.join(
                savefolder, "cake_battery_usage_per_battery_per_month.png"
            )
            plt.savefig(savepath, bbox_inches="tight")
        plt.show()


### Capacity Factors ###


def plot_correlation_matrix_carrier_by_carrier(
    capacity_factors: pd.DataFrame, savefolder: str = None
):

    # Extract unique carrier types from the column names
    carrier_types = set(col.split()[-1] for col in capacity_factors.columns)
    # Loop through each carrier type and compute the correlation matrix
    for carrier in carrier_types:
        # Filter columns that contain the current carrier type
        carrier_columns = [
            col for col in capacity_factors.columns if col.endswith(carrier)
        ]
        if carrier in ["oil", "CCGT", "coal", "OCGT", "lignite", "nuclear", "biomass"]:
            continue

        # Check if there are any columns for this carrier
        if carrier_columns:
            carrier_df = capacity_factors[carrier_columns]
            node_names = [" ".join(col.split()[:]) for col in carrier_columns]
            if all(node in node_to_city for node in node_names):
                # Create a mapping from node to city + carrier type for renaming
                renamed_columns = {
                    col: f"{node_to_city[' '.join(col.split()[:2])]} {carrier}"
                    for col in carrier_columns
                }

                # Rename the columns in the DataFrame for the current carrier
                carrier_df = carrier_df.rename(columns=renamed_columns)

            # Compute the correlation matrix for the current carrier
            carrier_corr_matrix = carrier_df.corr()

            # Print the title for the current carrier
            print(f"Correlation Matrix for {carrier.capitalize()} Generators\n")

            # Display the heatmap for the current carrier with a fixed scale from 0 to 1
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                carrier_corr_matrix,
                annot=False,
                cmap="coolwarm",
                linewidths=0.5,
                vmin=0,
                vmax=1,
            )
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            if savefolder:
                savepath = os.path.join(savefolder, f"correlation_matrix_{carrier}.png")
                plt.savefig(savepath, bbox_inches="tight")

            # Show the plot
            plt.show()


def get_carrier_maps(generators: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns two dictionaries extracted from the generators DataFrame:
      - carrier_colors: mapping of carrier to its color.
      - carrier_nice_names: mapping of carrier to its nicer display name.

    The generators DataFrame must have the columns: "carrier", "color", and "nice_name".
    """
    carrier_colors: dict[str, str] = (
        generators.drop_duplicates("carrier").set_index("carrier")["color"].to_dict()
    )
    carrier_nice_names: dict[str, str] = (
        generators.drop_duplicates("carrier")
        .set_index("carrier")["nice_name"]
        .to_dict()
    )
    return carrier_colors, carrier_nice_names


def _get_global_y_limits(
    capacity_factors: pd.DataFrame,
    generator_types: list[str] = ["solar", "onwind", "offwind-ac", "ror"],
) -> tuple[float, float]:
    """
    Computes the global minimum and maximum y-values based on the hourly and monthly averages.
    These values can be used to keep the y-axis limits consistent across plots.
    """
    hourly_avg = capacity_factors.groupby(capacity_factors.index.hour).mean()
    monthly_avg = capacity_factors.groupby(capacity_factors.index.month).mean()
    global_min: float = float(min(hourly_avg.min().min(), monthly_avg.min().min()))
    global_max: float = float(max(hourly_avg.max().max(), monthly_avg.max().max()))
    return global_min, global_max


def plot_avg_hourly_capacity_factors(
    capacity_factors: pd.DataFrame,
    generators: pd.DataFrame,
    generator_types: list[str] = ["solar", "onwind", "offwind-ac", "ror"],
    output_folder: str = None,
) -> None:
    """
    Plots the average capacity factors by hour of the day (with ±1 standard deviation)
    for all generator types in a single figure.
    """
    carrier_colors, carrier_nice_names = get_carrier_maps(generators)
    hourly_avg = capacity_factors.groupby(capacity_factors.index.hour).mean()
    hourly_std = capacity_factors.groupby(capacity_factors.index.hour).std()

    plt.figure(figsize=(12, 6))
    for gen in generator_types:
        gen_columns = [col for col in capacity_factors.columns if col.endswith(gen)]
        if gen_columns:
            mean = hourly_avg[gen_columns].mean(axis=1)
            std = hourly_std[gen_columns].mean(axis=1)
            color: str = carrier_colors.get(gen, "black")
            nice_name: str = carrier_nice_names.get(gen, gen.capitalize())
            plt.plot(mean.index, mean, label=nice_name, color=color)
            plt.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.3)

    plt.xlabel("Hour of Day", fontsize=16)
    plt.ylabel("Average Capacity Factor", fontsize=16)
    plt.xticks(range(0, 24), fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)

    if output_folder:
        output_path = os.path.join(
            output_folder, "average_hourly_capacity_factors_with_std.png"
        )
        plt.savefig(output_path, bbox_inches="tight")
    print("Average Hourly Capacity Factors with ±1 standard deviation")
    plt.show()


def plot_avg_monthly_capacity_factors(
    capacity_factors: pd.DataFrame,
    generators: pd.DataFrame,
    generator_types: list[str] = ["solar", "onwind", "offwind-ac", "ror"],
    output_folder: str = None,
) -> None:
    """
    Plots the average capacity factors by month (with ±1 standard deviation)
    for all generator types in a single figure.
    """
    carrier_colors, carrier_nice_names = get_carrier_maps(generators)
    monthly_avg = capacity_factors.groupby(capacity_factors.index.month).mean()
    monthly_std = capacity_factors.groupby(capacity_factors.index.month).std()
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    plt.figure(figsize=(12, 6))
    for gen in generator_types:
        gen_columns = [col for col in capacity_factors.columns if col.endswith(gen)]
        if gen_columns:
            mean = monthly_avg[gen_columns].mean(axis=1)
            std = monthly_std[gen_columns].mean(axis=1)
            color: str = carrier_colors.get(gen, "black")
            nice_name: str = carrier_nice_names.get(gen, gen.capitalize())
            plt.plot(months, mean, label=nice_name, color=color)
            plt.fill_between(months, mean - std, mean + std, color=color, alpha=0.3)

    plt.xlabel("Month", fontsize=16)
    plt.ylabel("Average Capacity Factor", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)

    if output_folder:
        output_path = os.path.join(
            output_folder, "average_monthly_capacity_factors_with_std.png"
        )
        plt.savefig(output_path, bbox_inches="tight")
    print("Average Monthly Capacity Factors with ±1 standard deviation")
    plt.show()


def plot_avg_hourly_capacity_factors_sep(
    capacity_factors: pd.DataFrame,
    generators: pd.DataFrame,
    generator_types: list[str] = ["solar", "onwind", "offwind-ac", "ror"],
    output_folder: str = None,
) -> None:
    """
    Creates separate hourly plots (one per generator type) showing average capacity factors
    by hour of the day with ±1 standard deviation. Global y-axis limits are computed within the function.
    """
    carrier_colors, carrier_nice_names = get_carrier_maps(generators)
    hourly_avg = capacity_factors.groupby(capacity_factors.index.hour).mean()
    hourly_std = capacity_factors.groupby(capacity_factors.index.hour).std()

    # Compute global y-axis limits internally.
    global_min, global_max = _get_global_y_limits(capacity_factors, generator_types)

    for gen in generator_types:
        gen_columns = [col for col in capacity_factors.columns if col.endswith(gen)]
        if gen_columns:
            mean = hourly_avg[gen_columns].mean(axis=1)
            std = hourly_std[gen_columns].mean(axis=1)
            color: str = carrier_colors.get(gen, "black")
            nice_name: str = carrier_nice_names.get(gen, gen.capitalize())

            plt.figure(figsize=(12, 6))
            plt.plot(mean.index, mean, label=nice_name, color=color)
            plt.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.3)
            plt.xlabel("Hour of Day", fontsize=16)
            plt.ylabel("Average Capacity Factor", fontsize=16)
            plt.xticks(range(0, 24), fontsize=14)
            plt.yticks(fontsize=14)
            plt.ylim(global_min, global_max)
            plt.legend(fontsize=16)
            plt.grid(True, linestyle="--", alpha=0.5)
            if output_folder:
                output_path = os.path.join(
                    output_folder, f"{gen}_average_hourly_capacity_factors_with_std.png"
                )
                plt.savefig(output_path, bbox_inches="tight")
            print(
                f"Average Hourly Capacity Factors for {gen} with ±1 standard deviation"
            )
            plt.show()


def plot_avg_monthly_capacity_factors_sep(
    capacity_factors: pd.DataFrame,
    generators: pd.DataFrame,
    generator_types: list[str] = ["solar", "onwind", "offwind-ac", "ror"],
    output_folder: str = None,
) -> None:
    """
    Creates separate monthly plots (one per generator type) showing average capacity factors
    by month with ±1 standard deviation. Global y-axis limits are computed within the function.
    """
    carrier_colors, carrier_nice_names = get_carrier_maps(generators)
    monthly_avg = capacity_factors.groupby(capacity_factors.index.month).mean()
    monthly_std = capacity_factors.groupby(capacity_factors.index.month).std()
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Compute global y-axis limits internally.
    global_min, global_max = _get_global_y_limits(capacity_factors, generator_types)

    for gen in generator_types:
        gen_columns = [col for col in capacity_factors.columns if col.endswith(gen)]
        if gen_columns:
            mean = monthly_avg[gen_columns].mean(axis=1)
            std = monthly_std[gen_columns].mean(axis=1)
            color: str = carrier_colors.get(gen, "black")
            nice_name: str = carrier_nice_names.get(gen, gen.capitalize())

            plt.figure(figsize=(12, 6))
            plt.plot(months, mean, label=nice_name, color=color)
            plt.fill_between(months, mean - std, mean + std, color=color, alpha=0.3)
            plt.xlabel("Month", fontsize=16)
            plt.ylabel("Average Capacity Factor", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.ylim(global_min, global_max)
            plt.legend(fontsize=16)
            plt.grid(True, linestyle="--", alpha=0.5)
            if output_folder:
                output_path = os.path.join(
                    output_folder,
                    f"{gen}_average_monthly_capacity_factors_with_std.png",
                )
                plt.savefig(output_path, bbox_inches="tight")
            print(
                f"Average Monthly Capacity Factors for {gen} with ±1 standard deviation"
            )
            plt.show()


def plot_heatmap_capacity_factors(
    capacity_factors: pd.DataFrame,
    generator_types: list[str] = ["solar", "onwind", "offwind-ac", "ror"],
    output_folder: str = None,
    use_global_limits: bool = True,
) -> None:
    """
    For each generator type, creates a heatmap of capacity factors by hour of day and month.

    If use_global_limits is True, the color scale limits (vmin, vmax) are computed using the
    global limits function; otherwise, defaults of 0 and 1 are used.
    """
    for gen in generator_types:
        gen_columns = [col for col in capacity_factors.columns if col.endswith(gen)]
        if gen_columns:
            mean_series = capacity_factors[gen_columns].mean(axis=1)
            heatmap_data = (
                mean_series.groupby([mean_series.index.hour, mean_series.index.month])
                .mean()
                .unstack()
            )

            plt.figure(figsize=(12, 6))
            if use_global_limits:
                global_min, global_max = _get_global_y_limits(
                    capacity_factors, generator_types
                )
                vmin, vmax = global_min, global_max
                sns.heatmap(
                    heatmap_data, cmap="coolwarm", linewidths=0.5, vmin=vmin, vmax=vmax
                )
            else:
                sns.heatmap(heatmap_data, cmap="coolwarm", linewidths=0.5)
            plt.xlabel("Month", fontsize=16)
            plt.ylabel("Hour of Day", fontsize=16)
            plt.yticks(
                ticks=range(0, 24, 2),
                labels=[str(h) for h in range(0, 24, 2)],
                fontsize=14,
            )
            plt.grid(True, linestyle="--", alpha=0.5)
            print(
                f"Heatmap of {gen.capitalize()} capacity factors by hour of day and month"
            )
            if output_folder:
                output_path = os.path.join(
                    output_folder, f"{gen}_heatmap_capacity_factors.png"
                )
                plt.savefig(output_path, bbox_inches="tight")
            plt.show()


def plot_avg_weekly_capacity_factors(
    capacity_factors: pd.DataFrame,
    generator_types: list[str] = ["solar", "onwind", "offwind-ac", "ror"],
    output_folder: str = None,
    use_global_limits: bool = True,
) -> None:
    """
    Plots the average capacity factors by day of the week (aggregated for all generator types)
    in a single figure.

    If use_global_limits is True, the y-axis limits are computed using the global limits function;
    otherwise, default axis scaling is used.
    """
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    plt.figure(figsize=(12, 6))
    for gen in generator_types:
        gen_columns = [col for col in capacity_factors.columns if col.endswith(gen)]
        if gen_columns:
            # Group by day of week and take the mean across the selected columns.
            daily_avg = (
                capacity_factors[gen_columns]
                .groupby(capacity_factors.index.dayofweek)
                .mean()
                .mean(axis=1)
            )
            plt.plot(days, daily_avg, label=gen.capitalize())
    plt.xlabel("Day of the Week", fontsize=16)
    plt.ylabel("Average Capacity Factor", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)

    if use_global_limits:
        global_min, global_max = _get_global_y_limits(capacity_factors, generator_types)
        plt.ylim(global_min, global_max)

    print(
        "Average capacity factors by day of the week (aggregated over all generator types)"
    )
    if output_folder:
        output_path = os.path.join(output_folder, "average_weekly_capacity_factors.png")
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_timeseries_capacity_factors(
    capacity_factors: pd.DataFrame,
    generators: pd.DataFrame,
    generator_types: list[str] = ["solar", "onwind", "offwind-ac", "ror"],
    output_folder: str = None,
    use_global_limits: bool = True,
) -> None:
    """
    Plots the time series of capacity factors for all generator types in one figure.

    If use_global_limits is True, global y-axis limits are computed using get_global_y_limits;
    otherwise, default axis scaling is used.
    """
    carrier_colors, carrier_nice_names = get_carrier_maps(generators)

    if use_global_limits:
        global_min, global_max = _get_global_y_limits(capacity_factors, generator_types)

    plt.figure(figsize=(12, 6))

    for gen in generator_types:
        gen_columns = [col for col in capacity_factors.columns if col.endswith(gen)]
        if gen_columns:
            mean_series = capacity_factors[gen_columns].mean(axis=1)
            color: str = carrier_colors.get(gen, "black")
            nice_name: str = carrier_nice_names.get(gen, gen.capitalize())
            plt.plot(mean_series.index, mean_series, label=nice_name, color=color)

    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Capacity Factor", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if use_global_limits:
        plt.ylim(global_min, global_max)

    plt.legend(fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)

    print(
        "Time series of capacity factors for all generator types "
        + (
            "using global y-axis limits."
            if use_global_limits
            else "with default axis scaling."
        )
    )

    if output_folder:
        output_path = os.path.join(output_folder, "time_series_capacity_factors.png")
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()


def plot_timeseries_capacity_factors_sep(
    capacity_factors: pd.DataFrame,
    generators: pd.DataFrame,
    generator_types: list[str] = ["solar", "onwind", "offwind-ac", "ror"],
    output_folder: str = None,
    use_global_limits: bool = True,
) -> None:
    """
    Creates separate time series plots (one per generator type) with consistent y-axis limits.

    If use_global_limits is True, global y-axis limits are computed using get_global_y_limits;
    otherwise, default axis scaling is used.
    """
    carrier_colors, carrier_nice_names = get_carrier_maps(generators)

    if use_global_limits:
        global_min, global_max = _get_global_y_limits(capacity_factors, generator_types)

    for gen in generator_types:
        gen_columns = [col for col in capacity_factors.columns if col.endswith(gen)]
        if gen_columns:
            mean_series = capacity_factors[gen_columns].mean(axis=1)
            color: str = carrier_colors.get(gen, "black")
            nice_name: str = carrier_nice_names.get(gen, gen.capitalize())

            plt.figure(figsize=(12, 6))
            plt.plot(mean_series.index, mean_series, label=nice_name, color=color)
            plt.xlabel("Time", fontsize=16)
            plt.ylabel("Capacity Factor", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            if use_global_limits:
                plt.ylim(global_min, global_max)

            plt.legend(fontsize=16)
            plt.grid(True, linestyle="--", alpha=0.5)

            print(
                f"Time series of {nice_name} capacity factors "
                + (
                    "using global y-axis limits."
                    if use_global_limits
                    else "with default axis scaling."
                )
            )

            if output_folder:
                output_path = os.path.join(
                    output_folder, f"{gen}_time_series_capacity_factors.png"
                )
                plt.savefig(output_path, bbox_inches="tight")

            plt.show()


def plot_monthly_production(
    generators: pd.DataFrame, generation: pd.DataFrame, savefolder: str = None
) -> None:
    # Extract unique carrier types, their colors, and nice names
    carrier_colors = generators.set_index("carrier")["color"].to_dict()
    carrier_nice_names = generators.set_index("carrier")["nice_name"].to_dict()

    # Extract month from the index of the generation DataFrame
    generation["month"] = pd.to_datetime(generation.index).month

    # Initialize a DataFrame to store monthly totals for each carrier
    monthly_totals = pd.DataFrame()

    # Calculate total production per carrier per month
    for carrier in generators["carrier"].unique():
        carrier_columns = [
            col
            for col in generation.columns
            if col.endswith(carrier) or col.endswith(f"{carrier} new")
        ]
        monthly_totals[carrier] = (
            generation[carrier_columns].sum(axis=1).groupby(generation["month"]).sum()
        ) / 1e6  # Convert MWh to TWh

    # Plot the stacked bar chart
    monthly_totals.index = monthly_totals.index.map(
        lambda x: [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ][x - 1]
    )
    monthly_totals.plot(
        kind="bar",
        stacked=True,
        color=[carrier_colors[carrier] for carrier in monthly_totals.columns],
        figsize=(18, 9),
    )

    # Customize the plot
    plt.ylabel("Total Production (TWh)", fontsize=20)
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=16)
    plt.legend(
        [carrier_nice_names[carrier] for carrier in monthly_totals.columns],
        loc="upper center",
        fontsize=18,
        framealpha=1,
        bbox_to_anchor=(0.5, -0.16),
        ncol=len(monthly_totals.columns),
    )
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.5)
    print("Monthly Production by Carrier")
    # # Save the plot
    if savefolder:
        savepath = os.path.join(savefolder, "bar_monthly_production.png")
        plt.savefig(savepath, bbox_inches="tight")

    # Show the plot
    plt.show()


def plot_congestion_network(
    buses,
    branches,
    congestion_rate,
    savefolder=None,
    plot_line_numbers=False,
    text_offset=15,
    divider=48,
    type: str = "",
):
    """Plots the network with buses and transmission lines, with branch color representing congestion rate."""

    # Convert buses DataFrame to a GeoDataFrame

    geometry_buses = gpd.points_from_xy(buses["x"], buses["y"])

    gdf_buses = gpd.GeoDataFrame(buses, geometry=geometry_buses, crs="EPSG:4326")

    # Create a dictionary to map bus IDs to their coordinates for easy access

    bus_coords = gdf_buses[["x", "y"]].to_dict("index")

    norm = plt.Normalize(vmin=0, vmax=1)

    # Generate list of line segments for LineCollection based on bus coordinates

    lines = []

    line_colors = []

    line_widths = []

    min_line_width = 2

    max_line_width = 5
    line_midpoints = []

    for _, row in branches.iterrows():

        if row["bus0"] in bus_coords and row["bus1"] in bus_coords:

            point0 = (bus_coords[row["bus0"]]["x"], bus_coords[row["bus0"]]["y"])

            point1 = (bus_coords[row["bus1"]]["x"], bus_coords[row["bus1"]]["y"])

            lines.append([point0, point1])

            # Get congestion rate for the branch

            branch_id = str(row.name)

            color = plt.cm.bwr(congestion_rate.get(branch_id, 0))

            line_colors.append(color)

            if plot_line_numbers:
                # Calculate the midpoint for adding line numbers later
                midpoint = (
                    (point0[0] + point1[0]) / 2,
                    (point0[1] + point1[1]) / 2,
                )

                # Calculate the perpendicular vector for the line
                perpendicular_vector = np.array(
                    [point1[1] - point0[1], point0[0] - point1[0]]
                )
                perpendicular_vector = perpendicular_vector / np.linalg.norm(
                    perpendicular_vector
                )

                # Adjust perpendicular vector to point consistently
                if perpendicular_vector[0] > 0:
                    perpendicular_vector = -perpendicular_vector
                if perpendicular_vector[1] > 0:
                    perpendicular_vector = -perpendicular_vector

                # Offset the midpoint by the perpendicular vector for text placement
                offset_midpoint = (
                    midpoint[0] + text_offset * perpendicular_vector[0] / divider,
                    midpoint[1] + text_offset * perpendicular_vector[1] / divider,
                )
                line_midpoints.append(
                    (offset_midpoint, _)
                )  # Store offset midpoint and line number

            # Normalize p_max between min_line_width and max_line_width
            divide_by = (branches["p_max"].max() - branches["p_max"].min()) * (
                max_line_width - min_line_width
            )
            divide_by = 1 if divide_by == 0 else divide_by
            linewidth = (
                min_line_width + (row["p_max"] - branches["p_max"].min()) / divide_by
            )

            line_widths.append(linewidth)

        else:

            print(f"Excluded line with missing bus coordinates: {row}")

    fig, ax = plot_background(buses)

    # Create and plot the LineCollection for transmission lines

    lc = LineCollection(lines, colors=line_colors, linewidths=line_widths, zorder=15)

    ax.add_collection(lc)

    # Plot buses

    gdf_buses.plot(
        ax=ax, color="black", marker="o", markersize=50, zorder=20
    )  # No legend entry for buses

    # Add a colorbar for the congestion rate heatmap

    sm = plt.cm.ScalarMappable(cmap="bwr", norm=norm)

    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, orientation="vertical")

    # cbar.set_label("Congestion Rate", fontsize=12)

    if plot_line_numbers:
        # Add text labels for line numbers at the offset positions
        for offset_midpoint, line_number in line_midpoints:
            plt.text(
                offset_midpoint[0],
                offset_midpoint[1],
                str(line_number),
                fontsize=10,
                ha="center",
                va="center",
                color="blue",
                zorder=30,
            )

    # Customize the plot

    plt.xlabel("Longitude")

    plt.ylabel("Latitude")

    ax.set_axis_off()

    plt.grid(True)
    print(f"Plotting congestion network for {type}")

    plt.show()

    if savefolder:
        savepath = os.path.join(savefolder, f"grid_congestion_network_{type}.png")
        fig.savefig(savepath, bbox_inches="tight")


### Tables ###


def get_branches_overview_table(
    branches: pd.DataFrame, max_loss: float = 0.02, savefolder: str = None
) -> pd.DataFrame:
    branches["city1"] = branches["bus0"].map(node_to_city)
    branches["city2"] = branches["bus1"].map(node_to_city)
    pretty_branches = branches.loc[
        :, ["city1", "city2", "p_max", "length", "capital_cost"]
    ]
    pretty_branches.columns = [
        "City 1",
        "City 2",
        "Capacity [MW]",
        "Length [km]",
        "Cost [€]",
    ]
    pretty_branches["Loss Factor [%]"] = (
        max_loss * branches["length"] / branches["length"].max() * 1e2
    )
    pretty_branches.round(2)
    if savefolder:
        savepath = os.path.join(savefolder, "table_branches_overview.csv")
        pretty_branches.to_csv(savepath)


def get_generators_overview_table(
    generators: pd.DataFrame, savefolder: str = None
) -> pd.DataFrame:
    p_nom = generators.groupby("carrier")["p_nom"].sum()
    num_generators = generators.groupby("carrier").size()
    generators_overview = pd.DataFrame(
        {"Installed Capacity [MW]": p_nom, "Number of Generators": num_generators}
    )
    if savefolder:
        savepath = os.path.join(savefolder, "table_generators_overview.csv")
        generators_overview.to_csv(savepath)
    return generators_overview


def get_generators_cost_and_emissions_table(
    generators: pd.DataFrame, savefolder: str = None
) -> pd.DataFrame:
    generators_marginal_cost = generators.groupby("carrier")["marginal_cost"].mean()
    generators_co2_emissions = generators.groupby("carrier")["co2_emissions"].mean()
    generators_buildout_cost = generators.groupby("carrier")["capital_cost"].mean()
    generators_methodology_df = pd.concat(
        [generators_marginal_cost, generators_co2_emissions, generators_buildout_cost],
        axis=1,
    )
    generators_methodology_df.columns = [
        "Marginal Cost (€/MWh)",
        "CO2 Emissions (ton/MWh)",
        "Annualized Investment Cost (€/MW)",
    ]
    CO2_price = 85  # €/ton
    generators_methodology_df["CO2 Cost (€/MWh)"] = (
        generators_methodology_df["CO2 Emissions (ton/MWh)"] * CO2_price
    )
    generators_methodology_df["Total Marginal Cost (€/MWh)"] = (
        generators_methodology_df["Marginal Cost (€/MWh)"]
        + generators_methodology_df["CO2 Cost (€/MWh)"]
    )
    if savefolder:
        savepath = os.path.join(
            savefolder,
            "table_genertable_generators_costs_and_emissionsators_methodology.csv",
        )
        generators_methodology_df.to_csv(savepath)
    return generators_methodology_df
