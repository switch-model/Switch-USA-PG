import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pytz import timezone
import pytz

##### self_defined functions
# from TX_util import (tp_to_date)

# ############################################################################
# ############################################################################
# Find the special timepoints/dates ; highest marginal cost; highest demand;
path_list = [
    # "nodecarb_hydrogen/no",
    # "nodecarb_hydrogen/withintra",
    # "nodecarb_hydrogen/withall",
    # "decarb_hydrogen/no",
    # "decarb_hydrogen/withintra",
    # "decarb_hydrogen/withall",
    # "nodecarb_hydrogen_co2p/no",
    # "nodecarb_hydrogen_co2p/withintra",
    # "nodecarb_hydrogen_co2p/withall",
    "outputs_existing_2023",
]

#############################################################################
#############################################################################
#############################################################################

in_folder = "/Users/rangrang/Library/CloudStorage/GoogleDrive-zhengr@hawaii.edu/My Drive/SWITCH_INPUTS_rr_east/outputs_conus_transmission"
out_folder = "/Users/rangrang/Desktop/transmission_project/data_main"
m_mc_data = pd.DataFrame()
out_data = pd.DataFrame()
median_mc_data = pd.DataFrame()
Mmc_list_all = pd.DataFrame()
Medianmc_list_all = pd.DataFrame()


for i in path_list:
    result = [
        filename
        for filename in os.listdir(os.path.join(in_folder, i))
        if filename.startswith("energy_sources")
    ]
    es = pd.read_csv(os.path.join(in_folder, i, result[0]))
    es = es.loc[es["load_zone"] != "loadzone"]
    # es["timeseries"] = [x[0] + "_" + x[1] for x in es["timepoint_label"].str.split("_")]
    es["timeseries"] = "2023_2023-full"
    #  ts = pd.read_csv("inputs_conus_transmission/base_thin_origin/timeseries.csv")
    ts = pd.read_csv("inputs_conus_transmission/existing_2023/timeseries.csv")
    es_ts = pd.merge(es, ts, how="left", on="timeseries")
    es = es_ts[
        [
            "load_zone",
            "timepoint_label",
            "period",
            "zone_demand_mw",
            "marginal_cost",
            "ts_scale_to_period",
            "peak_day",
            "timeseries",
        ]
    ]
    es["model"] = i
    es.to_csv(os.path.join(out_folder, i, "unweighted_yealy_MC.csv"))

    ##### find the date with largest MC for TRE
    sources = es.copy()
    from transmission_scripts.TX_util import tp_to_date

    sources = tp_to_date(sources, "timepoint_label")
    sources["days"] = pd.to_numeric(sources["days"])
    from datetime import date

    def converttodate(dfrow):
        return date.fromisocalendar(dfrow["period"], dfrow["week"], dfrow["days"])

    sources["date"] = sources.apply(lambda dfrow: converttodate(dfrow), axis=1)
    sources = sources.sort_values(by=["date"])

    def weighted_average(dataframe, value, weight):
        val = dataframe[value]
        wt = dataframe[weight]
        return (val * wt).sum() / wt.sum()

    # Find the most expensive for ecort
    sources = sources.loc[sources["load_zone"] == "TRE"]
    mc_weighted = (
        (
            sources.groupby("date").apply(
                weighted_average, "marginal_cost", "zone_demand_mw"
            )
        )
        .to_frame(name="daily_mc")
        .reset_index()
    )
    Mmc_date = mc_weighted.loc[mc_weighted["daily_mc"] == mc_weighted["daily_mc"].max()]
    Mmc_date["model"] = i
    Mmc_date["zone"] = "TRE"
    m_mc_data = pd.concat([m_mc_data, Mmc_date])

    # #########################################
    # THE most expensive date for TRE at scenario of decarb/no is 2050-11-29
    sources_Mmc_date = sources.loc[sources["date"].isin(Mmc_date["date"])]
    # sources_Mmc_date = sources.loc[sources["date"] == date(2050, 12, 13)]
    sources_Mmc_date["model"] = i
    timepoints = pd.read_csv(
        os.path.join("inputs_conus_transmission/base_thin_origin", "timepoints.csv")
    )
    sources_Mmc_date = pd.merge(
        sources_Mmc_date,
        timepoints,
        left_on="timepoint_label",
        right_on="timestamp",
        how="left",
    )
    sources_Mmc_date.to_csv(
        os.path.join(out_folder, i, "Max_date_zoneMC.csv"), index=False
    )
    m_mc_tp = sources_Mmc_date["timepoint_label"]
    m_mc_date = sources_Mmc_date["date"]
    Mmc_list = pd.DataFrame()
    Mmc_list["tp"] = m_mc_tp
    Mmc_list["date"] = m_mc_date
    Mmc_list["model"] = i
    Mmc_list_all = pd.concat([Mmc_list_all, Mmc_list])
    Mmc_list_all.to_csv(os.path.join(out_folder, "Mmc_list_all.csv"), index=False)
    Mmc_list_all.to_clipboard()
    ##########

    median_mc = sources.loc[sources["timepoint_label"] == "2050_p4_0"]
    sources.loc[sources["marginal_cost"] == 37.31046186167824]
    median_mc["module"] = i
    # results = pd.DataFrame(results)
    median_mc_data = pd.concat([median_mc_data, median_mc])
    median_mc_tp = median_mc["timepoint_label"]
    median_mc_date = median_mc["date"]

    sources_Medianmc_date = sources.loc[sources["date"].isin(median_mc_date)]
    sources_Medianmc_date["model"] = i
    sources_Medianmc_date = pd.merge(
        sources_Medianmc_date,
        timepoints,
        left_on="timepoint_label",
        right_on="timestamp",
        how="left",
    )

    sources_Medianmc_date.to_csv(
        os.path.join(out_folder, i, "Median_date_zoneMC.csv"), index=False
    )
    Medianmc_list = pd.DataFrame()
    Medianmc_list["tp"] = median_mc_tp
    Medianmc_list["date"] = median_mc_date
    Medianmc_list["model"] = i
    Medianmc_list_all = pd.concat([Medianmc_list_all, Medianmc_list])
    Medianmc_list_all.to_csv(
        os.path.join(out_folder, "Medianmc_list_all.csv"), index=False
    )

    #############################################################################
    # m_mc_date = ["2050-07-05"]

    df_list = [m_mc_date, median_mc_date]
    tp_list = [m_mc_tp, median_mc_tp]
    for index, df in enumerate(df_list):
        capacity_tx = pd.read_csv(os.path.join(in_folder, i, "transmission.csv"))
        timepoints = pd.read_csv(
            os.path.join("inputs_conus_transmission/base_thin_origin", "timepoints.csv")
        )
        timepoints = tp_to_date(timepoints, "timestamp")
        timepoints["days"] = pd.to_numeric(timepoints["days"])
        timepoints["date"] = timepoints.apply(
            lambda dfrow: converttodate(dfrow), axis=1
        )
        specific_tp = timepoints.loc[
            timepoints["timestamp"] == tp_list[index].values[0], "timepoint_id"
        ].iloc[0]
        match_stamp = timepoints.loc[timepoints["date"].isin(df)]
        match = timepoints.loc[timepoints["date"].isin(df), "timepoint_id"]
        ###########
        load = pd.read_csv(
            os.path.join("inputs_conus_transmission/base_thin_origin/loads.csv")
        )
        load_zone = pd.DataFrame(load.LOAD_ZONE.unique(), columns=["zone"])
        zone26 = load_zone.loc[load_zone["zone"] != "loadzone"]
        load_match = load.loc[load["TIMEPOINT"].isin(match)]
        load_match["date"] = match_stamp["date"].iloc[0]
        load_match["model"] = i
        ###########
        dispatch_tx = pd.read_csv(os.path.join(in_folder, i, "DispatchTx.csv"))
        dispatch_tx_match = dispatch_tx.loc[
            dispatch_tx["TRANS_TIMEPOINTS_3"].isin(match)
        ]
        out_tx = (
            dispatch_tx_match.groupby(["TRANS_TIMEPOINTS_1", "TRANS_TIMEPOINTS_3"])
            .agg({"DispatchTx": "sum"})
            .reset_index()
        )
        in_tx = (
            dispatch_tx_match.groupby(["TRANS_TIMEPOINTS_2", "TRANS_TIMEPOINTS_3"])
            .agg({"DispatchTx": "sum"})
            .reset_index()
        )
        in_tx = pd.DataFrame(in_tx)
        flow = pd.merge(
            out_tx,
            in_tx,
            left_on=["TRANS_TIMEPOINTS_1", "TRANS_TIMEPOINTS_3"],
            right_on=["TRANS_TIMEPOINTS_2", "TRANS_TIMEPOINTS_3"],
            how="outer",
        )
        zone_flow = flow.rename(
            {
                "TRANS_TIMEPOINTS_1": "zone",
                "TRANS_TIMEPOINTS_3": "timepoint",
                "DispatchTx_x": "outflow",
                "DispatchTx_y": "inflow",
            },
            axis=1,
        )
        zone_flow["date"] = match_stamp["date"].iloc[0]
        zone_flow["model"] = i
        zone_flow["unit"] = "MWh"
        ###########
        dp = pd.read_csv(os.path.join(in_folder, i, "dispatch.csv"))
        dp = pd.merge(dp, timepoints, how="left", on="timestamp")
        dp_match = dp.loc[dp["timestamp"].isin(match_stamp.timestamp)]
        dp_match["date"] = match_stamp["date"].iloc[0]
        dp_match["model"] = i
        df = dp_match.copy()
        df["zone"] = df["gen_load_zone"]
        df["resource_name"] = df["generation_project"]
        from transmission_scripts.TX_util import tech_type_group

        df = tech_type_group(df)
        df["unit"] = "MWh"
        df["value"] = df["DispatchGen_MW"]
        df = df[["zone", "tech_type", "value", "unit", "timepoint_id", "date", "model"]]
        # find out how much hydrogen are built
        hydrogen_original = pd.read_csv(
            os.path.join(
                "outputs_conus/outputs_main_new_update",
                i,
                "DispatchFuelCellMW.csv",
            )
        )
        hydrogen = hydrogen_original.loc[
            hydrogen_original["DispatchFuelCellMW_index_2"].isin(match)
        ]
        hydrogen["date"] = match_stamp["date"].iloc[0]
        hydrogen = hydrogen.rename(
            {
                "DispatchFuelCellMW_index_1": "resource_name",
                "DispatchFuelCellMW_index_2": "timepoint_id",
                "DispatchFuelCellMW": "value",
            },
            axis=1,
        )
        hydrogen["zone"] = hydrogen["resource_name"]
        hydrogen["tech_type"] = "Hydrogen"
        hydrogen["model"] = i
        hydrogen["unit"] = "MWh"
        hydrogen = hydrogen[
            ["zone", "tech_type", "value", "unit", "timepoint_id", "date", "model"]
        ]

        #  combine df df with hydrigen
        df = pd.concat([df, hydrogen])

        # denote the max/median timepoint
        load_match["specific_tp"] = specific_tp
        load_match.to_csv(
            os.path.join(out_folder, i, str(index + 1) + "load_date.csv"),
            index=False,
        )

        zone_flow["specific_tp"] = specific_tp
        zone_flow.to_csv(
            os.path.join(out_folder, i, str(index + 1) + "TXflow_date.csv"),
            index=False,
        )

        df["specific_tp"] = specific_tp
        df.to_csv(
            os.path.join(out_folder, i, str(index + 1) + "dp_date.csv"),
            index=False,
        )
        ###########
        dispatch = pd.read_csv(os.path.join(in_folder, i, "DispatchTx.csv"))
        dispatch_tx = dispatch.loc[dispatch["TRANS_TIMEPOINTS_3"].isin(match)]

        cap_tx = capacity_tx[
            [
                "TRANSMISSION_LINE",
                "trans_lz1",
                "trans_lz2",
                "TxCapacityNameplateAvailable",
            ]
        ]
        cap_tx["line"] = (
            cap_tx["trans_lz1"].astype(str) + " " + cap_tx["trans_lz2"].astype(str)
        )
        cap_tx["sorted_line"] = [
            " ".join(sorted(x.split())) for x in cap_tx["line"].tolist()
        ]
        cap_tx = cap_tx.loc[cap_tx["TxCapacityNameplateAvailable"] > 0]

        disp_tx = dispatch_tx.groupby(
            ["TRANS_TIMEPOINTS_1", "TRANS_TIMEPOINTS_2", "TRANS_TIMEPOINTS_3"],
            as_index=False,
        ).agg({"DispatchTx": "mean"})

        disp_tx["line"] = (
            disp_tx["TRANS_TIMEPOINTS_1"].astype(str)
            + " "
            + disp_tx["TRANS_TIMEPOINTS_2"].astype(str)
        )
        disp_tx["sorted_line"] = [
            " ".join(sorted(x.split())) for x in disp_tx["line"].tolist()
        ]
        disp_tx = disp_tx.sort_values(
            ["sorted_line", "TRANS_TIMEPOINTS_3", "DispatchTx"]
        )
        disp_tx["diff"] = disp_tx.groupby(["sorted_line", "TRANS_TIMEPOINTS_3"])[
            "DispatchTx"
        ].diff()
        # disp_tx["max"] = disp_tx.groupby(["sorted_line"])["DispatchTx"].transform(max)
        df = disp_tx.copy()

        results = pd.merge(cap_tx, df, how="left", on="sorted_line", indicator=True)
        #  edit the unmatched ones, the ones with no use of tx
        results["diff"] = results["diff"].fillna(0)
        results.loc[results["_merge"] == "left_only", "diff"] = 0
        results.loc[results["_merge"] == "left_only", "TRANS_TIMEPOINTS_1"] = results[
            "trans_lz1"
        ]
        results.loc[results["_merge"] == "left_only", "TRANS_TIMEPOINTS_2"] = results[
            "trans_lz2"
        ]

        results = results[
            [
                "TRANS_TIMEPOINTS_1",
                "TRANS_TIMEPOINTS_2",
                "TRANS_TIMEPOINTS_3",
                "diff",
                "TxCapacityNameplateAvailable",
            ]
        ]
        results["utility_rate"] = (
            results["diff"] / results["TxCapacityNameplateAvailable"]
        )
        results["utility_rate"] = results["utility_rate"].fillna(0)
        results["model"] = i
        results["hour_tp"] = index + 1
        results = pd.DataFrame(results)
        m_mc_data = pd.concat([out_data, results])
        m_mc_data.to_csv(
            os.path.join(out_folder, str(index + 1) + "Max_date_alldata.csv"),
            index=False,
        )

        results.to_csv(
            os.path.join(out_folder, i, str(index + 1) + "disp_date_tx.csv"),
            index=False,
        )

################################### Rents for eaach resource ###################################
################################### Rents for eaach resource ###################################
################################### Rents for eaach resource ###################################
# calculate rents: real costs for each resource: ("GenCapitalRecovery", "GenFixedOMCosts") in gen_cap.csv
# and variable OM cost in dispatch.csv

in_folder = "/Users/rangrang/Library/CloudStorage/GoogleDrive-zhengr@hawaii.edu/My Drive/SWITCH_INPUTS_rr_east"
out_folder = "/Users/rangrang/Desktop/transmission_project/data_main"

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
disp_all = pd.DataFrame()
gen_cap_all = pd.DataFrame()
fc_all = pd.DataFrame()
# for x in [dispatch_list_path, dispatch_list_path]:
for i in dispatch_list_path:
    dispatch = pd.read_csv(
        os.path.join(
            in_folder, "outputs_conus/outputs_main_new_update", i, "dispatch.csv"
        )
    )
    df = dispatch.copy()
    # if i in dispatch_list_path:
    df["model"] = i

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
            "DispatchFuelCellMW_index_1": "zone",
            "DispatchFuelCellMW_index_2": "timestep",
            "DispatchFuelCellMW": "value",
        },
        axis=1,
    )
    hydrogen["tech_type"] = "Hydrogen"
    hydrogen["model"] = i
    hydrogen["unit"] = "MWh"
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
    tp_weight = tp_weight[["timepoint_id", "timestamp", "weight"]]
    hydrogen = hydrogen.merge(
        tp_weight, left_on="timestep", right_on="timepoint_id", how="left"
    )

    # change the values of some columns to align with generation.csv
    hydrogen["value"] = hydrogen["value"] * hydrogen["weight"]
    hydrogen["VariableCost_per_yr"] = "."
    hydrogen = hydrogen[
        [
            "value",
            "model",
            "zone",
            "tech_type",
            "timestamp",
            "unit",
            "VariableCost_per_yr",
        ]
    ]
    # else:
    #     df["model"] = i.replace("no_renewable", "")

    df["zone"] = df["gen_load_zone"]
    df["resource_name"] = df["generation_project"]

    from transmission_scripts.TX_util import tech_type_group

    df = tech_type_group(df)
    df["unit"] = "MWh"
    df["value"] = df["DispatchGen_MW"] * df["tp_weight_in_year_hrs"]
    df = df[
        [
            "resource_name",
            "value",
            "model",
            "zone",
            "tech_type",
            "timestamp",
            "unit",
            "VariableCost_per_yr",
        ]
    ]
    #  combine df df with hydrigen
    df = pd.concat([df, hydrogen])
    generation = df.groupby("resource_name", as_index=False).agg(
        {
            "value": "sum",
            "model": "first",
            "zone": "first",
            "tech_type": "first",
            "timestamp": "first",
            "unit": "first",
            "VariableCost_per_yr": "sum",
        }
    )
    gen = generation.groupby(["model", "tech_type", "zone"], as_index=False).agg(
        {
            "VariableCost_per_yr": "sum",
        }
    )
    disp_all = pd.concat([disp_all, df])

    mc = pd.read_csv(
        os.path.join(
            "/Users/rangrang/Desktop/transmission_project/data_main",
            i,
            "unweighted_yealy_MC.csv",
        )
    )
    mc = mc[
        ["load_zone", "timepoint_label", "zone_demand_mw", "marginal_cost", "model"]
    ]
    df_mc = df.merge(
        mc,
        left_on=["zone", "timestamp", "model"],
        right_on=["load_zone", "timepoint_label", "model"],
        how="left",
    )
    df_mc["implied_Bil"] = df_mc["value"] * df_mc["marginal_cost"] / 1e9
    disp_mc = df_mc.groupby(["model", "tech_type", "zone"], as_index=False).agg(
        {"implied_Bil": "sum"}
    )
    disp_mc.to_csv(
        os.path.join(
            "/Users/rangrang/Desktop/transmission_project/data/main", i, "disp_mc.csv"
        )
    )

    gen_cap = pd.read_csv(
        os.path.join(
            in_folder, "outputs_conus/outputs_main_new_update", i, "gen_cap.csv"
        )
    )
    gc = gen_cap.groupby(
        ["gen_load_zone", "gen_energy_source"],
        as_index=False,
    ).agg({"GenCapitalRecovery": "sum", "GenFixedOMCosts": "sum"})

    gc["tech_type"] = "other"
    gc.loc[gc["gen_energy_source"] == "Coal", "tech_type"] = "Coal"
    gc.loc[gc["gen_energy_source"] == "Electricity", "tech_type"] = "Battery"
    gc.loc[gc["gen_energy_source"] == "Naturalgas", "tech_type"] = "Natural Gas"
    gc.loc[gc["gen_energy_source"] == "Water", "tech_type"] = "Hydro"
    gc.loc[gc["gen_energy_source"] == "Uranium", "tech_type"] = "Nuclear"
    gc.loc[gc["gen_energy_source"] == "Solar", "tech_type"] = "Solar"
    gc.loc[gc["gen_energy_source"] == "Wind", "tech_type"] = "Wind"

    gc["model"] = i
    gen_cap_all = pd.concat([gen_cap_all, gc])
    ########################## Calculate fuel cost for each resource and zone
    fuel_use_rate = pd.read_csv(
        os.path.join(
            in_folder,
            "outputs_conus/outputs_main_new_update",
            i,
            "GenFuelUseRate.csv",
        )
    )
    fuel_price = pd.read_csv(
        os.path.join(
            in_folder,
            "inputs_conus_transmission/base_thin_origin",
            "fuel_cost.csv",
        )
    )

    gen_info = pd.read_csv(
        os.path.join(
            in_folder, "inputs_conus_transmission/base_thin_origin", "gen_info.csv"
        )
    )
    fuel_use_rate = fuel_use_rate.merge(
        gen_info, left_on="GEN_TP_FUELS_1", right_on="GENERATION_PROJECT", how="left"
    )
    fuel_use = fuel_use_rate.merge(
        tp_weight, left_on="GEN_TP_FUELS_2", right_on="timepoint_id", how="left"
    )
    fuel_use = fuel_use[
        [
            "GEN_TP_FUELS_1",
            "GEN_TP_FUELS_2",
            "GEN_TP_FUELS_3",
            "GenFuelUseRate",
            "gen_load_zone",
            "weight",
        ]
    ]
    fuel_use = fuel_use.groupby(
        ["GEN_TP_FUELS_2", "GEN_TP_FUELS_3", "gen_load_zone"], as_index=False
    ).agg({"GenFuelUseRate": "sum", "weight": "first"})
    fuel_cost = fuel_use.merge(
        fuel_price,
        left_on=["gen_load_zone", "GEN_TP_FUELS_3"],
        right_on=["load_zone", "fuel"],
        how="left",
    )
    fuel_cost["fuel_cost_yearly"] = (
        fuel_cost["fuel_cost"] * fuel_cost["GenFuelUseRate"] * fuel_cost["weight"]
    )
    fuel_cost["fuel_cost_Bil"] = fuel_cost["fuel_cost_yearly"]
    fuel_cost_all = fuel_cost.groupby(
        ["gen_load_zone", "GEN_TP_FUELS_3"], as_index=False
    ).agg({"fuel_cost_yearly": "sum"})
    fuel_cost_all["model"] = i
    fuel_cost_all.loc[
        fuel_cost_all["GEN_TP_FUELS_3"] == "Uranium", "GEN_TP_FUELS_3"
    ] = "Nuclear"
    fuel_cost_all.loc[
        fuel_cost_all["GEN_TP_FUELS_3"] == "Naturalgas", "GEN_TP_FUELS_3"
    ] = "Natural Gas"
    fc_all = pd.concat([fc_all, fuel_cost_all])
    ########### merge all datasets
    cost_disp = pd.merge(
        disp_all,
        gen_cap_all,
        left_on=["model", "tech_type", "zone"],
        right_on=["model", "tech_type", "gen_load_zone"],
        how="inner",
        # indicator=True,
    )
    cost_disp = cost_disp.drop_duplicates(subset=["model", "tech_type", "zone"])
    cost_real = pd.merge(
        cost_disp,
        fc_all,
        left_on=["model", "tech_type", "zone"],
        right_on=["model", "GEN_TP_FUELS_3", "gen_load_zone"],
        how="left",
        indicator=True,
    )
    cost_real["fuel_cost_yearly"] = cost_real["fuel_cost_yearly"].fillna(0)
    cost_real.to_csv(
        os.path.join("/Users/rangrang/Desktop/cost_real_zonal.csv"), index=False
    )


################################### Rents for each transmission lines ###################################
################################### Rents for each transmission lines ###################################
################################### Rents for each transmission lines ###################################

in_folder = "/Users/rangrang/Library/CloudStorage/GoogleDrive-zhengr@hawaii.edu/My Drive/SWITCH_INPUTS_rr_east"
out_folder = "/Users/rangrang/Desktop/transmission_project/data_main"

for i in path_list:
    result = [
        filename
        for filename in os.listdir(
            os.path.join(in_folder, "outputs_conus/outputs_main_new_update", i)
        )
        if filename.startswith("energy_sources")
    ]
    es = pd.read_csv(
        os.path.join(in_folder, "outputs_conus/outputs_main_new_update", i, result[0])
    )
    es = es.loc[es["load_zone"] != "loadzone"]
    es["timeseries"] = [x[0] + "_" + x[1] for x in es["timepoint_label"].str.split("_")]

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
    tp_weight = tp_weight[["timepoint_id", "timestamp", "weight"]]

    es_ts = pd.merge(
        es, tp_weight, how="left", left_on="timepoint_label", right_on="timestamp"
    )
    es = es_ts[
        [
            "load_zone",
            "timepoint_id",
            "timepoint_label",
            "marginal_cost",
            "weight",
            "timeseries",
        ]
    ]
    disp_tx = pd.read_csv(
        os.path.join(
            in_folder, "outputs_conus/outputs_main_new_update", i, "DispatchTx.csv"
        )
    )
    disp_tx["line"] = (
        disp_tx["TRANS_TIMEPOINTS_1"].astype(str)
        + " "
        + disp_tx["TRANS_TIMEPOINTS_2"].astype(str)
    )
    disp_tx["sorted_line"] = [
        " ".join(sorted(x.split())) for x in disp_tx["line"].tolist()
    ]
    disp_tx = disp_tx.sort_values(["sorted_line", "TRANS_TIMEPOINTS_3", "DispatchTx"])
    disp_tx["diff"] = disp_tx.groupby(["sorted_line", "TRANS_TIMEPOINTS_3"])[
        "DispatchTx"
    ].diff()
    disp_tx["diff"] = disp_tx["diff"].fillna(0)
    es_send = pd.merge(
        es,
        disp_tx,
        how="left",
        left_on=["load_zone", "timepoint_id"],
        right_on=["TRANS_TIMEPOINTS_1", "TRANS_TIMEPOINTS_3"],
    )
    es_send["send"] = es_send["marginal_cost"] * es_send["diff"] * es_send["weight"]
    es_send = es_send.groupby(
        ["TRANS_TIMEPOINTS_1", "TRANS_TIMEPOINTS_2", "TRANS_TIMEPOINTS_3"],
        as_index=False,
    ).agg(
        {
            "send": "sum",
            "timepoint_label": "first",
            "line": "first",
            "sorted_line": "first",
        }
    )
    es_send = es_send[
        [
            "TRANS_TIMEPOINTS_1",
            "TRANS_TIMEPOINTS_2",
            "TRANS_TIMEPOINTS_3",
            "timepoint_label",
            "line",
            "sorted_line",
            "send",
        ]
    ]

    es_receive = pd.merge(
        es,
        disp_tx,
        how="left",
        left_on=["load_zone", "timepoint_id"],
        right_on=["TRANS_TIMEPOINTS_2", "TRANS_TIMEPOINTS_3"],
    )
    in_tx = pd.read_csv(
        os.path.join(
            in_folder,
            "inputs_conus_transmission/base_thin_origin",
            "transmission_lines_withall.csv",
        )
    )
    in_tx["line"] = (
        in_tx["trans_lz1"].astype(str) + " " + in_tx["trans_lz2"].astype(str)
    )
    in_tx["sorted_line"] = [" ".join(sorted(x.split())) for x in in_tx["line"].tolist()]
    es_receive = pd.merge(
        es_receive,
        in_tx,
        how="left",
        on=["sorted_line"],
    )
    es_receive["line"] = es_receive["line_x"]
    es_receive["receive"] = (
        es_receive["marginal_cost"]
        * es_receive["diff"]
        * es_receive["weight"]
        * es_receive["trans_efficiency"]
    )
    es_receive = es_receive.groupby(
        ["TRANS_TIMEPOINTS_1", "TRANS_TIMEPOINTS_2", "TRANS_TIMEPOINTS_3"],
        as_index=False,
    ).agg(
        {
            "receive": "sum",
            "timepoint_label": "first",
            "line": "first",
            "sorted_line": "first",
        }
    )
    es_receive = es_receive[
        [
            "TRANS_TIMEPOINTS_1",
            "TRANS_TIMEPOINTS_2",
            "TRANS_TIMEPOINTS_3",
            "timepoint_label",
            "line",
            "sorted_line",
            "receive",
        ]
    ]

    tx_value = pd.merge(
        es_send,
        es_receive,
        how="left",
        on=[
            "TRANS_TIMEPOINTS_1",
            "TRANS_TIMEPOINTS_2",
            "TRANS_TIMEPOINTS_3",
            "timepoint_label",
            "line",
            "sorted_line",
        ],
    )
    tx_value["value"] = tx_value["receive"] - tx_value["send"]
    tx_value = tx_value.groupby(["sorted_line"], as_index=False).agg(
        {"value": "sum", "line": "first", "sorted_line": "first"}
    )

    tx = pd.read_csv(
        os.path.join(
            in_folder, "outputs_conus/outputs_main_new_update", i, "transmission.csv"
        )
    )
    tx["line"] = tx["trans_lz1"].astype(str) + " " + tx["trans_lz2"].astype(str)
    tx["sorted_line"] = [" ".join(sorted(x.split())) for x in tx["line"].tolist()]
    tx = tx.sort_values(["sorted_line"])
    tx = tx[["TotalAnnualCost", "line", "sorted_line"]]
    tx_all = pd.merge(tx_value, tx, how="right", on=["sorted_line"])
    tx_all["TotalAnnualCost_npv"] = tx_all["TotalAnnualCost"]
    tx_all["rent"] = tx_all["value"] - tx_all["TotalAnnualCost_npv"]
    tx_all["model"] = i
    tx_all.to_csv(os.path.join(out_folder, i, "tx_rent.csv"), index=False)
