import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pytz import timezone
import pytz
from datetime import date

##### self_defined functions
from transmission_scripts.TX_util import tp_to_date


def converttodate(dfrow):
    return date.fromisocalendar(dfrow["period"], dfrow["week"], dfrow["days"])


os.getcwd()
# "/Volumes/GoogleDrive/My Drive/SWITCH_INPUTS_rr_east/outputs_wecc"
# in_folder.mkdir(exist_ok=True)
# out_folder = "figures/tables"
###########################################################################
############# remove new built nuclear #################################
#### gen_info.csv, gen_build_costs.csv, and variable_capacity_factors.csv
in_folder = "outputs_conus_transmission/outputs_main"
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
out_folder = "/Users/rangrang/Desktop/transmission_project/data_main"
cb = pd.DataFrame()
for i in path_list:
    capacity_tx = pd.read_csv(os.path.join(in_folder, i, "transmission.csv"))
    dispatch_tx = pd.read_csv(os.path.join(in_folder, i, "DispatchTx.csv"))

    cap_tx = capacity_tx[
        ["TRANSMISSION_LINE", "trans_lz1", "trans_lz2", "TxCapacityNameplateAvailable"]
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
    ).agg({"DispatchTx": "mean"})
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

    results = pd.merge(cap_tx, df, how="outer", on="sorted_line", indicator=True)
    results = results[
        [
            "TRANS_TIMEPOINTS_1",
            "TRANS_TIMEPOINTS_2",
            "diff",
            "TxCapacityNameplateAvailable",
        ]
    ]
    results["utility_rate"] = results["diff"] / results["TxCapacityNameplateAvailable"]
    results["model"] = i
    results.to_csv(os.path.join(out_folder, i, "disp_tx.csv"), index=False)
    cb = pd.concat([cb, results])

cb.to_csv(os.path.join(out_folder, "ALL_disp_tx.csv"), index=False)


# ############################################################################
# ############################################################################
# Find the special timepoints/dates ; highest marginal cost; highest demand;
in_folder = "outputs_conus_transmission/outputs_main"
out_folder = "/Users/rangrang/Desktop/transmission_project/data_main"
m_mc_data = pd.DataFrame()
out_data = pd.DataFrame()
median_mc_data = pd.DataFrame()
ac = pd.DataFrame()
# check for mc
in_folder = "/Users/rangrang/Library/CloudStorage/GoogleDrive-zhengr@hawaii.edu/My Drive/SWITCH_INPUTS_rr_east/outputs_conus_transmission/outputs_main"
# path_list = ["outputs_test"]
for i in path_list:
    es = pd.read_csv(os.path.join(in_folder, i, "energy_sources.csv"))
    es = es.loc[es["load_zone"] != "loadzone"]

    es = es[
        [
            "load_zone",
            "period",
            "timepoint_label",
            "zone_demand_mw",
            "marginal_cost",
            "peak_day",
        ]
    ]
    es["model"] = i
    es_weighted = es.copy()
    es_weighted["demand_sum"] = (
        es_weighted["zone_demand_mw"]
        .groupby(es_weighted["timepoint_label"])
        .transform("sum")
    )
    es_weighted.to_csv(os.path.join(out_folder, i, "unweighted_yealy_MC.csv"))

    es_weighted["marginal_cost"] = (
        es_weighted["marginal_cost"]
        * es_weighted["zone_demand_mw"]
        / es_weighted["demand_sum"]
    )
    es_weighted.to_csv(os.path.join(out_folder, i, "yealy_MC.csv"))

    # export summary.csv
    average_cost = pd.read_csv(os.path.join(in_folder, i, "summary.csv"))
    average_cost["scenario"] = i
    ac = pd.concat([ac, average_cost])
    ac.to_csv(os.path.join(out_folder, "average_cost.csv"))

    es_zone = es_weighted.groupby("load_zone").agg(
        {"marginal_cost": "mean", "zone_demand_mw": "mean"}
    )
    es_zone["model"] = i
    es_zone.to_csv(os.path.join(out_folder, i, "yealy_averageMC.csv"))

    # weight marginal cost by demand
    sources = es.copy()
    sources["demand_sum"] = (
        sources["zone_demand_mw"].groupby(sources["timepoint_label"]).transform("sum")
    )
    sources["marginal_cost"] = (
        sources["marginal_cost"] * sources["zone_demand_mw"] / sources["demand_sum"]
    )

    ####################### plot the standard deviation

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
    sources_nation = tp_to_date(sources_nation, "timepoint_label")
    sources_nation["days"] = pd.to_numeric(sources_nation["days"])

    sources_nation["date"] = sources_nation.apply(
        lambda dfrow: converttodate(dfrow), axis=1
    )
    sources_days = sources_nation.sort_values(by=["date"])
    sources_days["demand_gw"] = sources_days["zone_demand_mw"] / 1000

    #############################################################################
    m_demand = sources_days.loc[
        sources_days["zone_demand_mw"] == sources_days["zone_demand_mw"].max()
    ]
    m_mc = sources_days.loc[
        sources_days["marginal_cost"] == sources_days["marginal_cost"].max()
    ]
    m_mc["model"] = i
    # results = pd.DataFrame(results)
    m_mc_data = pd.concat([m_mc_data, m_mc])
    m_mc_data.to_csv(os.path.join(out_folder, "Max_hour_data.csv"), index=False)
    m_demand_tp = m_demand["timepoint_label"]
    m_mc_tp = m_mc["timepoint_label"]
    m_mc_date = m_mc["date"]

    ####
    if path_list.index(i) < 3:
        m_mc_date = pd.Series(
            [datetime.strptime("2050-07-05", "%Y-%m-%d").date()], index=[609]
        )
        m_mc_tp = pd.Series(["2050_p27_42"], index=[609])
    sources_Mmc = es.loc[es["timepoint_label"].isin(m_mc_tp)]
    sources_Mmc["model"] = i
    sources_Mmc.to_csv(os.path.join(out_folder, i, "Max_hour_zoneMC.csv"), index=False)
    ##########

    median_mc = sources_days.loc[sources_days["timepoint_label"] == "2050_p4_0"]
    sources_days.loc[sources_days["marginal_cost"] == 37.31046186167824]
    median_mc["module"] = i
    # results = pd.DataFrame(results)
    median_mc_data = pd.concat([median_mc_data, median_mc])
    median_mc_data.to_csv(os.path.join(out_folder, "Median_hour_data.csv"), index=False)
    median_mc_tp = median_mc["timepoint_label"]

    sources_Medianmc = es.loc[es["timepoint_label"].isin(median_mc_tp)]
    sources_Medianmc["model"] = i
    sources_Medianmc.to_csv(
        os.path.join(out_folder, i, "Median_hour_zoneMC.csv"), index=False
    )
    #############################################################################
    df_list = [m_demand_tp, m_mc_tp, median_mc_tp]
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
        disp_tx["diff"] = disp_tx["diff"].fillna(0)
        disp_tx = disp_tx.loc[disp_tx["diff"] > 0]
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
        results["hour_tp"] = index
        results = pd.DataFrame(results)
        m_mc_data = pd.concat([out_data, results])
        m_mc_data.to_csv(
            os.path.join(out_folder, str(index) + "Max_hour_alldata.csv"), index=False
        )

        results.to_csv(
            os.path.join(out_folder, i, str(index) + "disp_tx.csv"), index=False
        )


####################################################################################
####################################################################################
####################################################################################

# in_folder = "outputs_conus_transmission/outputs_main/decarb_hydrogen"
in_folder = "outputs_conus_transmission/outputs_main"
out_folder = "/Users/rangrang/Desktop/transmission_project/data_main"
p_list = [
    "no",
    "withintra",
    "withall",
]
scenario_list = [
    "nodecarb_hydrogen",
    "decarb_hydrogen",
    "nodecarb_hydrogen_co2p",
]

plt.figure(figsize=(10, 6), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.2)
plt.suptitle("Mean and SD of MC for all timepoints", fontsize=18, y=0.95)
for s in range(len(scenario_list)):
    mean_std_scatter = pd.DataFrame()
    fig = plt.subplot(1, 3, s + 1)
    for i in range(len(p_list)):
        print(s, i)
        sources = pd.read_csv(
            os.path.join(in_folder, scenario_list[s], p_list[i], "energy_sources.csv")
        )
        sources = sources.loc[sources["load_zone"] != "loadzone"]
        # sources = sources.loc[sources["marginal_cost"]>500]
        # plt.scatter(x= sources["timepoint_label"], y=sources["load_zone"], s=sources["marginal_cost"])
        # plt.show()
        # weight marginal cost by demand
        sources["demand_sum"] = (
            sources["zone_demand_mw"]
            .groupby(sources["timepoint_label"])
            .transform("sum")
        )
        sources["marginal_cost"] = (
            sources["marginal_cost"] * sources["zone_demand_mw"] / sources["demand_sum"]
        )

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
        sources_zone = sources["marginal_cost"].groupby(sources["load_zone"]).mean()
        sources_zone = pd.DataFrame(sources_zone)
        sources_zone["module"] = scenario_list[s] + "/" + p_list[i]
        sources_zone.to_csv(
            os.path.join(out_folder, scenario_list[s], p_list[i], "yealyMC_zone.csv"),
            index=True,
        )
        ####################### plot the standard deviation
        std = sources.groupby(sources["timepoint_label"])["marginal_cost"].std()
        mean = sources.groupby(sources["timepoint_label"])["marginal_cost"].mean()
        std_Df = pd.DataFrame({"tp": std.index, "SD": std.values})
        mean_Df = pd.DataFrame({"tp": mean.index, "Mean": mean.values})
        std_mean = pd.merge(std_Df, mean_Df, how="right", on=["tp"])
        std_mean["transmission"] = p_list[i]
        mean_std_scatter = pd.concat([mean_std_scatter, std_mean])

        # plot mean and sd as pairs
        # ax = fig.add_subplot(3,1,i+1)
        # df = mean_std_scatter.loc[mean_std_scatter['transmission']==p_list[i]]
        # ax.errorbar(range(1,1681),df['Mean'],  df['SD'], linestyle='None', marker='^')
        # ax.title.set_text(p_list[i])

    # plot mean and sd in different axises
    sns.scatterplot(
        y=mean_std_scatter["SD"] ** (1 / 2),
        x=mean_std_scatter["Mean"] ** (1 / 2),
        hue=mean_std_scatter["transmission"],
        style=mean_std_scatter["transmission"],
        alpha=0.5,
    )
    fig.set_xlim(-1, 15)
    fig.set_ylim(-1, 25)
    # fig.set_xticks(np.arange(0, 15, 3))
    # fig.set_yticks(np.arange(0, 30, 5))
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    fig.set_xlabel("Sqr_Mean")
    fig.set_ylabel("Sqr_SD")
    fig.set_title(scenario_list[s], fontsize=12)
    fig.grid()
    # handles, labels = fig.get_legend_handles_labels()
    # fig._legend.remove()
# plt.set_ylabel(" SD")
# plt.legend(handles, labels, loc='upper center')
plt.tight_layout(pad=2)
plt.show()
plt.savefig(
    os.path.join(out_folder, "mean_MG.png"),
    # format="png",
    # dpi=100,
    # bbox_inches="tight",
)
# graph.axes.set_title("Standard deviation for all timepoints",fontsize=16)
# fig.set_xlabel("Square root of MEAN",fontsize=14)
# fig.set_ylabel("Square root of SD",fontsize=14)

# fig.tight_layout(pad=2.0)
# fig.suptitle("Mean and sd across regions for each timepoint -- co2 $190")
# plt.show()


graph = sns.scatterplot(
    y=mean_std_scatter["SD"] ** (1 / 2),
    x=mean_std_scatter["Mean"] ** (1 / 2),
    hue=mean_std_scatter["transmission"],
    style=mean_std_scatter["transmission"],
    alpha=0.5,
)
graph.tick_params(axis="x", rotation=90)
graph.axes.set_xlabel("Square root of MEAN", fontsize=14)
graph.axes.set_ylabel("Square root of SD", fontsize=14)
# graph.axes.set_title("Standard deviation for all timepoints\n (scenario -- 100 decarb)",fontsize=16)
graph.axes.set_title(
    "Standard deviation for all timepoints\n (scenario -- CO2 $190)", fontsize=16
)
plt.show()


###############################  TEST , Ignore below #####################################################
###################################################################################
########################################## test ##########################################
