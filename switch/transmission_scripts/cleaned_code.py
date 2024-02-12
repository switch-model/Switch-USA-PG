import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pytz import timezone
import pytz


# ############################################################################
# ############################################################################
# Find the special timepoints/dates ; highest marginal cost; highest demand;
path_list = [
    "nodecarb_hydrogen/no",
    "nodecarb_hydrogen/withintra",
    "nodecarb_hydrogen/withall",
    "decarb_hydrogen/no",
    "decarb_hydrogen/withintra",
    "decarb_hydrogen/withall",
    "nodecarb_hydrogen_co2p/no",
    "nodecarb_hydrogen_co2p/withintra",
    "nodecarb_hydrogen_co2p/withall",
]
in_folder = "outputs_conus_transmission/outputs_main"
data_out = "/Users/rangrang/Desktop/transmission_project/data_main"
figure_out = "/Users/rangrang/Desktop/transmission_project/figure_main"

m_mc_data = pd.DataFrame()
out_data = pd.DataFrame()

for i in path_list:
    sources = pd.read_csv(os.path.join(in_folder, i, "energy_sources.csv"))
    sources = sources.loc[sources["load_zone"] != "loadzone"]
    sources = sources[
        [
            "load_zone",
            "period",
            "timepoint_label",
            "zone_demand_mw",
            "marginal_cost",
            "peak_day",
        ]
    ]

    sources_nation = sources.groupby("timepoint_label", as_index=False).agg(
        {
            "zone_demand_mw": "sum",
            "marginal_cost": "sum",
            "period": "first",
        }
    )

    sources_nation.loc[
        sources_nation["zone_demand_mw"] == sources_nation["zone_demand_mw"].max()
    ]
    sources_nation.loc[
        sources_nation["marginal_cost"] == sources_nation["marginal_cost"].max()
    ]

    sources_nation["week"] = [
        pd.to_numeric(x[1][1:])
        for x in sources_nation["timepoint_label"].str.split("_")
    ]
    sources_nation["hour"] = [
        pd.to_numeric(x[2]) for x in sources_nation["timepoint_label"].str.split("_")
    ]
    #  hours ro days
    def cat(row):
        if row["hour"] <= 23:
            return "1"
        elif row["hour"] > 23 and row["hour"] <= 47:
            return "2"
        elif row["hour"] > 47 and row["hour"] <= 71:
            return "3"
        elif row["hour"] > 71 and row["hour"] <= 95:
            return "4"
        elif row["hour"] > 95 and row["hour"] <= 119:
            return "5"
        elif row["hour"] > 119 and row["hour"] <= 143:
            return "6"
        return "7"

    sources_nation["days"] = sources_nation.apply(lambda row: cat(row), axis=1)
    sources_nation = sources_nation.sort_values(by=["week", "hour"])

    sources_nation["period"] = pd.to_numeric(sources_nation["period"])
    sources_nation["week"] = pd.to_numeric(sources_nation["week"])
    sources_nation["days"] = pd.to_numeric(sources_nation["days"])

    from datetime import date

    def todate(row):
        return date.fromisocalendar(row["period"], row["week"], row["days"])

    sources_nation["date"] = sources_nation.apply(lambda row: todate(row), axis=1)

    sources_days = sources_nation.groupby(["date"], as_index=False).agg(
        {
            "zone_demand_mw": "sum",
            "marginal_cost": "mean",
            "period": "first",
        }
    )
    sources_days = sources_nation.sort_values(by=["date"])
    sources_days["demand_gw"] = sources_days["zone_demand_mw"] / 1000

    #########################################
    # # sources_days['month'] = [x[1] for x in sources_days["date"].str.split("_")]
    # # create figure and axis objects with subplots()
    # fig, ax = plt.subplots()
    # # make a plot
    # ax.plot(sources_days.date, sources_days.zone_demand_mw, color="red", marker="o")
    # # set x-axis label
    # ax.set_xlabel("Date", fontsize=14)
    # # set y-axis label
    # ax.set_ylabel("Demand_MW", color="red", fontsize=14)
    # # twin object for two different y-axis on the sample plot
    # ax2 = ax.twinx()
    # # make a plot with different y-axis using second axis object
    # ax2.plot(sources_days.date, sources_days.marginal_cost, color="blue", marker="o")
    # ax2.set_ylabel("Marginal Cost", color="blue", fontsize=14)
    # # xticks = np.arange(1,12 , 1)
    # # ax2.set_xticks(xticks)
    # plt.xticks(
    #     [
    #         "2050-01",
    #         "2050-02",
    #         "2050-03",
    #         "2050-04",
    #         "2050-05",
    #         "2050-06",
    #         "2050-07",
    #         "2050-08",
    #         "2050-09",
    #         "2050-10",
    #         "2050-11",
    #         "2050-12",
    #     ],
    #     [
    #         "Jan",
    #         "Feb",
    #         "Mar",
    #         "Apr",
    #         "May",
    #         "Jun",
    #         "Jul",
    #         "Aug",
    #         "Sep",
    #         "Oct",
    #         "Nov",
    #         "Dec",
    #     ],
    #     rotation=30,
    # )
    # plt.title(i)
    # plt.show()
    # # save the plot as a file
    # fig.savefig(
    #     os.path.join(figure_out, i, "yearly_demand_MG.png"),
    #     format="png",
    #     dpi=100,
    #     bbox_inches="tight",
    # )

    #############################################################################
    m_demand = sources_days.loc[
        sources_days["zone_demand_mw"] == sources_days["zone_demand_mw"].max()
    ]
    m_mc = sources_days.loc[
        sources_days["marginal_cost"] == sources_days["marginal_cost"].max()
    ]
    m_mc["module"] = i
    # results = pd.DataFrame(results)
    m_mc_data = pd.concat([m_mc_data, m_mc])
    m_mc_data.to_csv(os.path.join(data_out, "Max_hour_data.csv"), index=False)
    m_demand_tp = m_demand["timepoint_label"]
    m_mc_tp = m_mc["timepoint_label"]

    sources_Mmc = sources.loc[sources["timepoint_label"].isin(m_mc_tp)]
    sources_Mmc["model"] = i
    sources_Mmc.to_csv(os.path.join(data_out, i, "Max_hour_zoneMC.csv"), index=False)
    #############################################################################
    df_list = [m_demand_tp, m_mc_tp]
    for index, df in enumerate(df_list):
        capacity_tx = pd.read_csv(os.path.join(in_folder, i, "transmission.csv"))
        timepoints = pd.read_csv(
            os.path.join("inputs_conus_transmission/inputs_main", "timepoints.csv")
        )
        match = timepoints.loc[timepoints["timestamp"].isin(df), "timepoint_id"]
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
            ["TRANS_TIMEPOINTS_1", "TRANS_TIMEPOINTS_2"], as_index=False
        ).agg({"DispatchTx": "mean", "TRANS_TIMEPOINTS_3": "mean"})
        disp_tx["line"] = (
            disp_tx["TRANS_TIMEPOINTS_1"].astype(str)
            + " "
            + disp_tx["TRANS_TIMEPOINTS_2"].astype(str)
        )
        disp_tx["sorted_line"] = [
            " ".join(sorted(x.split())) for x in disp_tx["line"].tolist()
        ]
        disp_tx = disp_tx.sort_values(["sorted_line", "DispatchTx"])
        disp_tx["diff"] = disp_tx.groupby("sorted_line")["DispatchTx"].diff()
        # disp_tx["max"] = disp_tx.groupby(["sorted_line"])["DispatchTx"].transform(max)
        df = disp_tx.loc[disp_tx["diff"] > 0]

        results = pd.merge(cap_tx, df, how="left", on="sorted_line", indicator=True)
        #  edit the unmatched ones, the ones with no use of tx
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
        results["model"] = i

        results = pd.DataFrame(results)
        m_mc_data = pd.concat([out_data, results])
        m_mc_data.to_csv(
            os.path.join(data_out, str(index) + "Max_hour_alldata.csv"), index=False
        )

        results.to_csv(
            os.path.join(data_out, i, str(index) + "disp_tx.csv"), index=False
        )
