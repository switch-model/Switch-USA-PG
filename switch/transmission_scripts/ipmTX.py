import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pytz import timezone
import pytz

plt.style.use("style.txt")


os.getcwd()
# "/Volumes/GoogleDrive/My Drive/SWITCH_INPUTS_rr_east/outputs_wecc"
# in_folder.mkdir(exist_ok=True)
out_folder = "figures/tables"
###########################################################################
############# remove new built nuclear #################################
#### gen_info.csv, gen_build_costs.csv, and variable_capacity_factors.csv
in_folder = "/Volumes/GoogleDrive/My Drive/SWITCH_INPUTS_rr_east/inputs_conus_transmission/base_thin_origin"
gen_info = pd.read_csv(os.path.join(in_folder, "gen_info.csv"))
build_cost = pd.read_csv(os.path.join(in_folder, "gen_build_costs.csv"))

# Find newgen of solar, wind and battery
newgen_cost = build_cost[
    build_cost["GENERATION_PROJECT"].str.contains("nuclear_nuclear", case=False)
]

# Filter out newgen of solar, wind and battery from csv files
gen_info_updated = gen_info[
    ~gen_info["GENERATION_PROJECT"].isin(newgen_cost.GENERATION_PROJECT.tolist())
]
build_cost_updated = build_cost[
    ~build_cost["GENERATION_PROJECT"].isin(newgen_cost.GENERATION_PROJECT.tolist())
]

gen_info_updated.to_csv(
    "/Volumes/GoogleDrive/My Drive/SWITCH_INPUTS_rr_east/inputs_conus_transmission/inputs_nonuclear/gen_info.csv",
    index=False,
)
build_cost_updated.to_csv(
    "/Volumes/GoogleDrive/My Drive/SWITCH_INPUTS_rr_east/inputs_conus_transmission/inputs_nonuclear/gen_build_costs.csv",
    index=False,
)


##################  remove newgen of solar, wind and battery in
#### gen_info.csv, gen_build_costs.csv, and variable_capacity_factors.csv
in_folder = "/Volumes/GoogleDrive/My Drive/SWITCH_INPUTS_rr_east/inputs_conus_transmission/base_thin_origin"
gen_info = pd.read_csv(os.path.join(in_folder, "gen_info.csv"))
build_cost = pd.read_csv(os.path.join(in_folder, "gen_build_costs.csv"))
capacity_factor = pd.read_csv(os.path.join(in_folder, "variable_capacity_factors.csv"))

# Find newgen of solar, wind and battery
newgen_cost = build_cost[build_cost["gen_fixed_om"] > 0]
newgen_cost = gen_info.merge(newgen_cost, on="GENERATION_PROJECT")
newgen_wind = newgen_cost[newgen_cost["gen_energy_source"] == "Wind"]
newgen_solar = newgen_cost[newgen_cost["gen_energy_source"] == "Solar"]
newgen_battery = newgen_cost[newgen_cost["gen_energy_source"] == "Electricity"]
newgen_VarBattery = pd.concat([newgen_wind, newgen_solar, newgen_battery])

# Filter out newgen of solar, wind and battery from csv files
gen_info_updated = gen_info[
    ~gen_info["GENERATION_PROJECT"].isin(newgen_VarBattery.GENERATION_PROJECT.tolist())
]
build_cost_updated = build_cost[
    ~build_cost["GENERATION_PROJECT"].isin(
        newgen_VarBattery.GENERATION_PROJECT.tolist()
    )
]
capacity_factor_updated = capacity_factor[
    ~capacity_factor["GENERATION_PROJECT"].isin(
        newgen_VarBattery.GENERATION_PROJECT.tolist()
    )
]

gen_info_updated.to_csv(
    "/Volumes/GoogleDrive/My Drive/SWITCH_INPUTS_rr_east/inputs_conus_transmission/inputs_norenewables/gen_info.csv",
    index=False,
)
build_cost_updated.to_csv(
    "/Volumes/GoogleDrive/My Drive/SWITCH_INPUTS_rr_east/inputs_conus_transmission/inputs_norenewables/gen_build_costs.csv",
    index=False,
)
capacity_factor_updated.to_csv(
    "/Volumes/GoogleDrive/My Drive/SWITCH_INPUTS_rr_east/inputs_conus_transmission/inputs_norenewables/variable_capacity_factors.csv",
    index=False,
)

#################################################################################
#########$############# deal with costs_itemized.csv
in_folder = "/Users/rangrang/Library/CloudStorage/GoogleDrive-zhengr@hawaii.edu/My Drive/SWITCH_INPUTS_rr_east/outputs_conus"
in_folder_new = "/Users/rangrang/Library/CloudStorage/GoogleDrive-zhengr@hawaii.edu/My Drive/SWITCH_INPUTS_rr_east/outputs_conus"

#######
nodecarb_no = pd.read_csv(
    os.path.join(
        in_folder_new, "outputs_main_new_update/nodecarb_hydrogen/no/costs_itemized.csv"
    )
)
nodecarb_withall = pd.read_csv(
    os.path.join(
        in_folder_new, "outputs_main_new_update/nodecarb_hydrogen/withall/costs_itemized.csv"
    )
)
nodecarb_withintra = pd.read_csv(
    os.path.join(
        in_folder_new, "outputs_main_new_update/nodecarb_hydrogen/withintra/costs_itemized.csv"
    )
)
#######
decarb_no = pd.read_csv(
    os.path.join(
        in_folder_new, "outputs_main_new_update/decarb_hydrogen/no/costs_itemized.csv"
    )
)
decarb_withall = pd.read_csv(
    os.path.join(
        in_folder_new, "outputs_main_new_update/decarb_hydrogen/withall/costs_itemized.csv"
    )
)
decarb_withintra = pd.read_csv(
    os.path.join(
        in_folder_new, "outputs_main_new_update/decarb_hydrogen/withintra/costs_itemized.csv"
    )
)


#########
nodecarb_co2p_no = pd.read_csv(
    os.path.join(
        in_folder_new, "outputs_main_new_update/nodecarb_hydrogen_co2p/no/costs_itemized.csv"
    )
)
nodecarb_co2p_withall = pd.read_csv(
    os.path.join(
        in_folder_new,
        "outputs_main_new_update/nodecarb_hydrogen_co2p/withall/costs_itemized.csv",
    )
)
nodecarb_co2p_withintra = pd.read_csv(
    os.path.join(
        in_folder_new,
        "outputs_main_new_update/nodecarb_hydrogen_co2p/withintra/costs_itemized.csv",
    )
)

##########################################
# costs = df.loc[df["PERIOD"] == 2050]
decarb_no["type"] = "decarb_no"
decarb_withall["type"] = "decarb_withall"
decarb_withintra["type"] = "decarb_withintra"

nodecarb_no["type"] = "nodecarb_no"
nodecarb_withall["type"] = "nodecarb_withall"
nodecarb_withintra["type"] = "nodecarb_withintra"

nodecarb_co2p_no["type"] = "co2p_no"
nodecarb_co2p_withall["type"] = "co2p_withall"
nodecarb_co2p_withintra["type"] = "co2p_withintra"

costs = pd.concat(
    [
        # decarb_no_renewable_withall,
        # decarb_no_renewable_withintra,
        # decarb_no_renewable_no,
        # nodecarb_no_renewable_withintra,
        # nodecarb_no_renewable_withall,
        # nodecarb_no_renewable_no,
        # nodecarb_no_renewable_co2p_no,
        # nodecarb_no_renewable_co2p_withall,
        # nodecarb_no_renewable_co2p_withintra,
        decarb_withall,
        decarb_no,
        decarb_withintra,
        nodecarb_withall,
        nodecarb_withintra,
        nodecarb_no,
        ##
        nodecarb_co2p_no,
        nodecarb_co2p_withall,
        nodecarb_co2p_withintra,
    ]
).copy()


costs = costs[["PERIOD", "Component", "AnnualCost_NPV", "type"]]
s = costs.loc[:, costs.columns != "PERIOD"].select_dtypes(include=[np.number]) * 1e-9
costs[s.columns] = s
costs

costs_NPV = pd.pivot(costs, index="type", columns="Component", values="AnnualCost_NPV")
col_list = [
    "FuelCostsPerTP",
    "GenVariableOMCostsInTP",
    "HydrogenFixedCostAnnual",
    "HydrogenVariableCost",
    "StorageEnergyFixedCost",
    "TotalGenFixedCosts",
    "TxFixedCosts",
]

costs_NPV["SubTotal"] = costs_NPV[col_list].sum(axis=1)

# costs_NPV['SocialCost'] = 0
# add emmission cost to nodecarb/least cost scenarios
nodecarb_no_dispatch = pd.read_csv(
    os.path.join(in_folder_new, "outputs_main_new_update/nodecarb_hydrogen/no/dispatch.csv")
)
nodecarb_withall_dispatch = pd.read_csv(
    os.path.join(
        in_folder_new, "outputs_main_new_update/nodecarb_hydrogen/withall/dispatch.csv"
    )
)
nodecarb_withintra_dispatch = pd.read_csv(
    os.path.join(
        in_folder_new, "outputs_main_new_update/nodecarb_hydrogen/withintra/dispatch.csv"
    )
)

costs_NPV.loc["nodecarb_no", "EmissionsCosts"] = (
    190 * nodecarb_no_dispatch["DispatchEmissions_tCO2_per_typical_yr"].sum() * 1e-9
)
costs_NPV.loc["nodecarb_withall", "EmissionsCosts"] = (
    190
    * nodecarb_withall_dispatch["DispatchEmissions_tCO2_per_typical_yr"].sum()
    * 1e-9
)
costs_NPV.loc["nodecarb_withintra", "EmissionsCosts"] = (
    190
    * nodecarb_withintra_dispatch["DispatchEmissions_tCO2_per_typical_yr"].sum()
    * 1e-9
)


costs_NPV["Total"] = costs_NPV["SubTotal"] + costs_NPV["EmissionsCosts"]
costs_NPV
cost_types = [
    "GenVariableOMCostsInTP",
    "FuelCostsPerTP",
    "StorageEnergyFixedCost",
    "TotalGenFixedCosts",
    "TxFixedCosts",
    "HydrogenFixedCostAnnual",
    "HydrogenVariableCost",
    "SubTotal",
    "EmissionsCosts",
    "Total",
]
costs_NPV = costs_NPV[cost_types]
# costs_NPV["Total"] = costs_NPV.sum(axis=1)
costs_NPV = costs_NPV.T
# costs_NPV["decarb_no_renewable_no"] = "."


costs_NPV = costs_NPV.round(decimals=2)
costs_NPV = costs_NPV.reindex(
    columns=[
        "nodecarb_no",
        "nodecarb_withintra",
        "nodecarb_withall",
        "decarb_no",
        "decarb_withintra",
        "decarb_withall",
        "co2p_no",
        "co2p_withintra",
        "co2p_withall",
    ]
)
costs_NPV.index = [
    "OM",
    "Fuel",
    "Storage",
    "Gen",
    "Tx",
    "HydrogenFX",
    "HydrogenVar",
    "SubTotal",
    "Emission",
    "Total",
]
costs_NPV.to_csv(
    "/Users/rangrang/Desktop/transmission_project/data/costs_transmission_test1.csv",
    index=True,
)



#################################################################################
################################### make  resource_capacity.csv
main_list = ["main", "base_thin_origin", "outputs_main_new_update"]
norenewables_list = ["norenewables", "inputs_norenewables", "outputs_norenewables"]
nonuclear_list = ["nonuclear", "inputs_nonuclear", "outputs_nonuclear"]
halfprice_list = [
    "halfprice_renewables",
    "inputs_halfprice_renewables",
    "outputs_halfprice_renewables",
]

in_folder = "/Users/rangrang/Library/CloudStorage/GoogleDrive-zhengr@hawaii.edu/My Drive/SWITCH_INPUTS_rr_east"
out_folder = "/Users/rangrang/Desktop/transmission_project/data"

for d in [main_list]:
    # add the retirement back to the resource list
    prebuild = pd.read_csv(
        os.path.join(
            in_folder, "inputs_conus_transmission", d[1], "gen_build_predetermined.csv"
        )
    )

    gen_info = pd.read_csv(
        os.path.join(in_folder, "inputs_conus_transmission", d[1], "gen_info.csv")
    )
    if d == nonuclear_list:
        build_list_path = [
            # "decarb_hydrogen/no", infeasible for no nuclear
            "decarb_hydrogen/withintra",
            "decarb_hydrogen/withall",
            "nodecarb_hydrogen/no",
            "nodecarb_hydrogen/withintra",
            "nodecarb_hydrogen/withall",
            "nodecarb_hydrogen_co2p/no",
            "nodecarb_hydrogen_co2p/withintra",
            "nodecarb_hydrogen_co2p/withall",
        ]
    else:
        build_list_path = [
            "decarb_hydrogen/no",
            "decarb_hydrogen/withintra",
            "decarb_hydrogen/withall",
            "nodecarb_hydrogen/no",
            "nodecarb_hydrogen/withintra",
            "nodecarb_hydrogen/withall",
            "nodecarb_hydrogen_co2p/no",
            "nodecarb_hydrogen_co2p/withintra",
            "nodecarb_hydrogen_co2p/withall",
        ]

    for i in build_list_path:
        build = pd.read_csv(
            os.path.join(in_folder, "outputs_conus", d[2], i, "BuildGen.csv")
        )
        merge = prebuild.merge(
            build,
            left_on=["GENERATION_PROJECT", "build_year"],
            right_on=["GEN_BLD_YRS_1", "GEN_BLD_YRS_2"],
            how="outer",
            indicator=True,
        )
        retire = pd.DataFrame(merge[merge["_merge"] == "left_only"])

        df = retire
        df["GEN_BLD_YRS_1"] = df["GENERATION_PROJECT"]
        df["GEN_BLD_YRS_2"] = df["build_year"]
        df["BuildGen"] = df["build_gen_predetermined"]
        df = df[["GEN_BLD_YRS_1", "GEN_BLD_YRS_2", "BuildGen"]]
        build_gen = pd.concat([build, df])

        df = build_gen
        df["start_value"] = np.where(df["GEN_BLD_YRS_2"] <= 2040, df["BuildGen"], 0)

        df["end_value"] = df["BuildGen"]
        merge = df.merge(
            gen_info,
            left_on=["GEN_BLD_YRS_1"],
            right_on=["GENERATION_PROJECT"],
            how="outer",
            indicator=True,
        )
        df = pd.DataFrame(merge[merge["_merge"] == "both"])

        resource_capacity = pd.DataFrame()

        resource_capacity["resource_name"] = df["GEN_BLD_YRS_1"]
        resource_capacity["zone"] = df["gen_load_zone"]
        resource_capacity["tech_type"] = "Other"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("batter", case=False),
            "tech_type",
        ] = "Battery"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("storage", case=False),
            "tech_type",
        ] = "Battery"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("coal", case=False),
            "tech_type",
        ] = "Coal"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("solar|pv", case=False),
            "tech_type",
        ] = "Solar"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("wind", case=False),
            "tech_type",
        ] = "Wind"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("hydro|water", case=False),
            "tech_type",
        ] = "Hydro"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("distribute", case=False),
            "tech_type",
        ] = "Distributed Solar"
        # resource_capacity.loc[
        #     resource_capacity["resource_name"].str.contains("geothermal", case=False),
        #     "tech_type",
        # ] = "Geothermal"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("nuclear", case=False),
            "tech_type",
        ] = "Nuclear"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("natural", case=False),
            "tech_type",
        ] = "Natural Gas"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("ccs", case=False),
            "tech_type",
        ] = "CCS"
        if i in build_list_path:
            resource_capacity["model"] = i

            # find out how much hydrogen are built
            hydrogen = pd.read_csv(
                os.path.join(
                    in_folder,
                    "outputs_conus",
                    d[2],
                    i,
                    "BuildFuelCellMW.csv",
                )
            )
            # hydrogen
            hydrogen = hydrogen.rename(
                {
                    "BuildFuelCellMW_index_1": "resource_name",
                    "BuildFuelCellMW_index_2": "planning_year",
                    "BuildFuelCellMW": "end_value",
                },
                axis=1,
            )
            hydrogen["zone"] = hydrogen["resource_name"]
            hydrogen["tech_type"] = "Hydrogen"
            hydrogen["model"] = i
            hydrogen["case"] = "unconstrained_thin"
            hydrogen["unit"] = "MW"
            hydrogen["start_value"] = 0
            hydrogen["end_value"] = hydrogen["end_value"]
        else:
            resource_capacity["model"] = i.replace("_no_renewable", "")

        resource_capacity["case"] = "unconstrained_thin"
        resource_capacity["unit"] = "MW"
        resource_capacity["planning_year"] = 2050

        resource_capacity["start_value"] = df["start_value"]
        resource_capacity["end_value"] = df["end_value"]

        if i in build_list_path:
            resource_capacity_agg = pd.concat([resource_capacity, hydrogen])
        else:
            resource_capacity_agg = resource_capacity

        resource_capacity_agg = resource_capacity_agg.loc[
            resource_capacity_agg["end_value"] > 0
        ]
        resource_capacity_agg.to_csv(
            os.path.join(out_folder, d[0], i, "resource_capacity.csv"), index=False
        )


################################### make  transmission.csv
in_folder = "/Users/rangrang/Library/CloudStorage/GoogleDrive-zhengr@hawaii.edu/My Drive/SWITCH_INPUTS_rr_east"

exist_trans = pd.read_csv(
    "/Users/rangrang/Library/CloudStorage/GoogleDrive-zhengr@hawaii.edu/My Drive/SWITCH_INPUTS_rr_east/inputs_conus_transmission/base_thin_origin/transmission_lines_withall.csv"
)
out_folder = "/Users/rangrang/Desktop/transmission_project/data/main"


tx_list_path = [
    "decarb_hydrogen/no",
    "decarb_hydrogen/withintra",
    "decarb_hydrogen/withall",
    "nodecarb_hydrogen/no",
    "nodecarb_hydrogen/withintra",
    "nodecarb_hydrogen/withall",
    "nodecarb_hydrogen_co2p/no",
    "nodecarb_hydrogen_co2p/withintra",
    "nodecarb_hydrogen_co2p/withall",
]


for i in tx_list_path:
    tx = pd.read_csv(
        os.path.join(
            in_folder,
            "outputs_conus/outputs_main_new_update",
            i,
            "transmission.csv",
        )
    )
    df = tx.merge(exist_trans, how="left", on=["trans_lz1", "trans_lz2"])

    df["model"] = i

    df["line_name"] = df["trans_lz1"] + "_to_" + df["trans_lz2"]
    df["planning_year"] = 2050
    df["case"] = "unconstrained_thin"
    df["unit"] = "MW"
    df["start_value"] = df["existing_trans_cap"]
    df["end_value"] = df["TxCapacityNameplate"]
    df["value"] = df["end_value"] - df["start_value"]
    df1 = df[
        [
            "model",
            "line_name",
            "planning_year",
            "case",
            "unit",
            "start_value",
            "end_value",
        ]
    ]
    df2 = df[["model", "line_name", "planning_year", "case", "unit", "value"]]
    df1.to_csv(os.path.join(out_folder, i, "transmission.csv"), index=False)
    df2.to_csv(os.path.join(out_folder, i, "transmission_expansion.csv"), index=False)


################################### make  generation.csv
in_folder = "/Users/rangrang/Library/CloudStorage/GoogleDrive-zhengr@hawaii.edu/My Drive/SWITCH_INPUTS_rr_east"
out_folder = "/Users/rangrang/Desktop/transmission_project/data/main"

dispatch_list_path = [
    "decarb_hydrogen/no",
    "decarb_hydrogen/withintra",
    "decarb_hydrogen/withall",
    "nodecarb_hydrogen/no",
    "nodecarb_hydrogen/withintra",
    "nodecarb_hydrogen/withall",
    "nodecarb_hydrogen_co2p/no",
    "nodecarb_hydrogen_co2p/withintra",
    "nodecarb_hydrogen_co2p/withall",
]

# to check
# dispatch2030["week"] = [x[1] for x in dispatch2030["timestamp"].str.split("_")]
# check = dispatch2030[["week", "tp_weight_in_year_hrs"]]
# check.drop_duplicates()

# for x in [dispatch_list_path, dispatch_list_path]:
for i in dispatch_list_path:
    dispatch = pd.read_csv(
        os.path.join(in_folder, "outputs_conus/outputs_main_new_update", i, "dispatch.csv")
    )
    df = dispatch.copy()
    # if i in dispatch_list_path:
    df["model"] = i.replace("_hydrogen", "")

    # find out how much hydrogen are built
    hydrogen = pd.read_csv(
        os.path.join(
            in_folder,
            "outputs_conus/outputs_main_new_update",
            i,
            "DispatchFuelCellMW.csv",
        )
    )
    # hydrogen
    hydrogen = hydrogen.rename(
        {
            "DispatchFuelCellMW_index_1": "resource_name",
            "DispatchFuelCellMW_index_2": "timestep",
            "DispatchFuelCellMW": "value",
        },
        axis=1,
    )
    hydrogen["zone"] = hydrogen["resource_name"]
    hydrogen["tech_type"] = "Hydrogen"
    hydrogen["model"] = i.replace("_hydrogen", "")
    hydrogen["case"] = "unconstrained_thin"
    hydrogen["unit"] = "MWh"
    hydrogen["planning_year"] = 2050
    # merge timepoint and timeseries to get tiomepoints weight
    tp = pd.read_csv(
        os.path.join(
            in_folder, "inputs_conus_transmission/base_thin_origin", "timepoints.csv"
        )
    )
    ts = pd.read_csv(
        os.path.join(
            in_folder, "inputs_conus_transmission/base_thin_origin", "timeseries.csv"
        )
    )
    tp_weight = tp.merge(ts, on=["timeseries"], how="left")
    tp_weight["weight"] = tp_weight["ts_scale_to_period"] / 10
    tp_weight = tp_weight[["timepoint_id", "weight"]]
    hydrogen = hydrogen.merge(
        tp_weight, left_on="timestep", right_on="timepoint_id", how="left"
    )
    # change the values of some columns to align with generation.csv
    hydrogen["timestep"] = "all"
    hydrogen["value"] = hydrogen["value"] * hydrogen["weight"]
    hydrogen = hydrogen[
        [
            "value",
            "model",
            "zone",
            "resource_name",
            "tech_type",
            "planning_year",
            "case",
            "timestep",
            "unit",
        ]
    ]
    # else:
    #     df["model"] = i.replace("no_renewable", "")

    df["zone"] = df["gen_load_zone"]
    df["resource_name"] = df["generation_project"]
    df["tech_type"] = "Other"
    df.loc[
        df["resource_name"].str.contains("batter", case=False), "tech_type"
    ] = "Battery"
    df.loc[
        df["resource_name"].str.contains("storage", case=False), "tech_type"
    ] = "Battery"
    df.loc[df["resource_name"].str.contains("coal", case=False), "tech_type"] = "Coal"
    df.loc[
        df["resource_name"].str.contains("solar|pv", case=False), "tech_type"
    ] = "Solar"
    df.loc[df["resource_name"].str.contains("wind", case=False), "tech_type"] = "Wind"
    df.loc[
        df["resource_name"].str.contains("hydro|water", case=False), "tech_type"
    ] = "Hydro"
    df.loc[
        df["resource_name"].str.contains("distributed", case=False), "tech_type"
    ] = "Distributed Solar"
    # df.loc[
    #     df["resource_name"].str.contains("geothermal", case=False), "tech_type"
    # ] = "Geothermal"
    df.loc[
        df["resource_name"].str.contains("nuclear", case=False), "tech_type"
    ] = "Nuclear"
    df.loc[
        df["resource_name"].str.contains("natural", case=False), "tech_type"
    ] = "Natural Gas"
    df.loc[df["resource_name"].str.contains("ccs", case=False), "tech_type"] = "CCS"
    # df.loc[
    #     df["resource_name"].str.contains("bio", case=False), "tech_type"
    # ] = "Bio Solid"

    df["planning_year"] = 2050

    df["case"] = "unconstrained_thin"

    df["timestep"] = "all"
    df["unit"] = "MWh"
    df["value"] = df["DispatchGen_MW"] * df["tp_weight_in_year_hrs"]

    #  combine df df with hydrigen
    df = pd.concat([df, hydrogen])
    generation = df.groupby("resource_name", as_index=False).agg(
        {
            "value": "sum",
            "model": "first",
            "zone": "first",
            "resource_name": "first",
            "tech_type": "first",
            "planning_year": "first",
            "case": "first",
            "timestep": "first",
            "unit": "first",
        }
    )

    generation.to_csv(os.path.join(out_folder, i, "generation.csv"), index=False)

    ##### for a specific hour
    ### DOUBLE CHECK BELOW
    m_mc_all = pd.read_csv(
        "/Users/rangrang/Desktop/transmission_project/data_main/Mmc_list_all.csv"
    )
    m_mc_tp = m_mc_all.loc[m_mc_all["model"] == i, "tp"]
    df_max = df.loc[df["timestamp"].isin(m_mc_tp)]
    ### DOUBLE CHECK ABOVE
    generation_max = df_max.groupby("resource_name", as_index=False).agg(
        {
            "value": "sum",
            "model": "first",
            "zone": "first",
            "resource_name": "first",
            "tech_type": "first",
            "planning_year": "first",
            "case": "first",
            "timestamp": "first",
            "unit": "first",
        }
    )

    generation_max.to_csv(
        os.path.join(out_folder, i, "max_hour_generation.csv"), index=False
    )

    df_median = df.loc[df["timestamp"].isin(median_mc_tp)]
    generation_median = df_median.groupby("resource_name", as_index=False).agg(
        {
            "value": "sum",
            "model": "first",
            "zone": "first",
            "resource_name": "first",
            "tech_type": "first",
            "planning_year": "first",
            "case": "first",
            "timestamp": "first",
            "unit": "first",
        }
    )

    generation_median.to_csv(
        os.path.join(out_folder, i, "median_hour_generation.csv"), index=False
    )
