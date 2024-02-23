import pandas as pd
import os
import numpy as np
import subprocess


################################### FOR 26 ZONE ###################################
################################### make  resource_capacity.csv
in_folder = ""

# results folder should be one level up from this script
results_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "MIP_results_comparison")
)

year_list = [2030, 2040, 2050]
case_list = {
    "base_short": "26z-short-base-200",
    "base_short_co2_1000": "26z-short-base-1000",
    "base_short_co2_50": "26z-short-base-50",
    "base_short_current_policies": "26z-short-current-policies",
    "base_short_no_ccs": "26z-short-no-ccs",
    "base_short_retire": "26z-short-retire",
    "base_short_tx_100": "26z-short-base-tx-100",
    "base_short_tx_15": "26z-short-base-tx-15",
    "base_short_tx_200": "26z-short-base-tx-200",
    "base_short_tx_50": "26z-short-base-tx-50",
    "base_short_commit": "26z-short-commit",
}

# TODO:
# 26z-short-base-tx-0
# 26z-short-foresight
# 26z-short-reserves


def skip_case(i):
    if not all(
        os.path.exists(
            os.path.join(in_folder, "26-zone/out", str(y), i, "BuildGen.csv")
        )
        for y in year_list
    ):
        print(f"Skipping unsolved case '{i}'.")
        return True

    print(f"Processing case '{i}'")
    return False


print("\ncreating resource_capacity.csv")
for i in case_list:
    if skip_case(i):
        continue
    tocompare_folder = os.path.join(
        results_folder, case_list[i], "SWITCH_results_summary"
    )
    resource_capacity_agg = pd.DataFrame()
    for y in year_list:
        # add the retirement back to the resource list
        ###### ADD capacity
        prebuild2030 = pd.read_csv(
            os.path.join(
                in_folder,
                "26-zone/in",
                str(y),
                i,
                "gen_build_predetermined.csv",
            )
        )
        # prebuild2050 = pd.read_csv(os.path.join(in_folder, "inputs_myopic2050/gen_build_predetermined.csv"))
        build2030 = pd.read_csv(
            os.path.join(in_folder, "26-zone/out", str(y), i, "BuildGen.csv")
        )

        merge2030 = prebuild2030.merge(
            build2030,
            left_on=["GENERATION_PROJECT", "build_year"],
            right_on=["GEN_BLD_YRS_1", "GEN_BLD_YRS_2"],
            how="outer",
            indicator=True,
        )
        retired_2030 = pd.DataFrame(merge2030[merge2030["_merge"] == "left_only"])

        df = retired_2030
        df["GEN_BLD_YRS_1"] = df["GENERATION_PROJECT"]
        df["GEN_BLD_YRS_2"] = df["build_year"]
        df["BuildGen"] = df["build_gen_predetermined"]
        df = df[["GEN_BLD_YRS_1", "GEN_BLD_YRS_2", "BuildGen"]]
        build_gen = pd.concat([build2030, df])
        build_gen["end_value"] = build_gen["BuildGen"]
        ###### ADD energy capacity
        energy_build2030 = pd.read_csv(
            os.path.join(
                in_folder,
                "26-zone/out",
                str(y),
                i,
                "BuildStorageEnergy.csv",
            )
        )
        df_energy = energy_build2030
        df_energy = df_energy.rename(
            {
                "STORAGE_GEN_BLD_YRS_1": "GEN_BLD_YRS_1",
                "STORAGE_GEN_BLD_YRS_2": "GEN_BLD_YRS_2",
                "BuildStorageEnergy": "MWh",
            },
            axis=1,
        )
        resource = build_gen.merge(
            df_energy,
            on=["GEN_BLD_YRS_1", "GEN_BLD_YRS_2"],
            how="left",
            # indicator=True,
        )
        ###############################
        df = resource

        if y == "foresight":
            df2030 = df.loc[df["GEN_BLD_YRS_2"] <= 2030]
            df2030["start_value"] = np.where(
                df2030["GEN_BLD_YRS_2"] < 2030, df2030["BuildGen"], 0
            )
            df2030["start_MWh"] = np.where(
                df2030["GEN_BLD_YRS_2"] < 2030, df2030["MWh"], 0
            )
            df2030["planning_year"] = 2030

            df2040 = df.loc[df["GEN_BLD_YRS_2"] <= 2040]
            df2040["start_value"] = np.where(
                df2040["GEN_BLD_YRS_2"] < 2040, df2040["BuildGen"], 0
            )
            df2040["start_MWh"] = np.where(
                df2040["GEN_BLD_YRS_2"] < 2040, df2040["MWh"], 0
            )
            df2040["planning_year"] = 2040

            df2050 = df.loc[df["GEN_BLD_YRS_2"] <= 2050]
            df2050["start_value"] = np.where(
                df2050["GEN_BLD_YRS_2"] < 2050, df2050["BuildGen"], 0
            )
            df2050["start_MWh"] = np.where(
                df2050["GEN_BLD_YRS_2"] < 2050, df2050["MWh"], 0
            )
            df2050["planning_year"] = 2050

            df = pd.concat([df2030, df2040, df2050])
        else:
            df["start_value"] = np.where(df["GEN_BLD_YRS_2"] < y, df["BuildGen"], 0)
            df["start_MWh"] = np.where(df["GEN_BLD_YRS_2"] < y, df["MWh"], 0)
            df["planning_year"] = y

        df.loc[df["MWh"] == "NaN", "start_MWh"] = "NaN"
        df["end_MWh"] = df["MWh"]
        resource_capacity = pd.DataFrame()

        resource_capacity["resource_name"] = df["GEN_BLD_YRS_1"]
        resource_capacity["zone"] = [x[0] for x in df["GEN_BLD_YRS_1"].str.split("_")]
        resource_capacity["tech_type"] = "Other"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("hydrogen", case=False),
            "tech_type",
        ] = "Hydrogen"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("batter", case=False),
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
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("geothermal", case=False),
            "tech_type",
        ] = "Geothermal"
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
        resource_capacity["model"] = "SWITCH"

        resource_capacity["planning_year"] = df["planning_year"]
        resource_capacity["case"] = i
        resource_capacity["unit"] = "MW"
        resource_capacity["start_value"] = df["start_value"]
        resource_capacity["end_value"] = df["end_value"]
        resource_capacity["start_MWh"] = df["start_MWh"]
        resource_capacity["end_MWh"] = df["end_MWh"]

        resource_capacity_agg = pd.concat([resource_capacity_agg, resource_capacity])

    resource_capacity_agg = resource_capacity_agg.loc[
        resource_capacity_agg["end_value"] > 0
    ]
    resource_capacity_agg.to_csv(
        os.path.join(in_folder, tocompare_folder, "resource_capacity.csv"),
        index=False,
    )


####################################### make  transmission.csv

print("\ncreating transmission.csv")
for i in case_list:
    if skip_case(i):
        continue
    tocompare_folder = os.path.join(
        results_folder, case_list[i], "SWITCH_results_summary"
    )
    tx_agg = pd.DataFrame()
    for y in year_list:
        transmission2030_new = pd.read_csv(
            os.path.join(in_folder, "26-zone/out", str(y), i, "transmission.csv")
        )

        # find the existing transmission capacity
        transmission2030_ex = pd.read_csv(
            os.path.join(in_folder, "26-zone/in", str(y), i, "transmission_lines.csv")
        )

        transmission2030 = transmission2030_new.merge(transmission2030_ex, how="left")

        df = transmission2030.copy()
        df["model"] = "SWITCH"
        df["line_name"] = df["trans_lz1"] + "_to_" + df["trans_lz2"]
        df["planning_year"] = df["PERIOD"]
        df["case"] = i
        df["unit"] = "MW"
        # for foreight scenario:
        # define a function to find the values of "TxCapacityNameplate" from last period
        # and fill it as the "start_value" of next period for same line.

        if y == "foresight":
            df.loc[df["PERIOD"] == 2030, "start_value"] = df["existing_trans_cap"]

            fill_dict2030 = (
                df.loc[df["PERIOD"] == 2030]
                .groupby("line_name")["TxCapacityNameplate"]
                .last()
                .to_dict()
            )
            df.loc[df["PERIOD"] == 2040, "start_value"] = df["line_name"].map(
                fill_dict2030
            )

            fill_dict2040 = (
                df.loc[df["PERIOD"] == 2040]
                .groupby("line_name")["TxCapacityNameplate"]
                .last()
                .to_dict()
            )
            df.loc[df["PERIOD"] == 2050, "start_value"] = df["line_name"].map(
                fill_dict2040
            )
        else:
            df["start_value"] = df["existing_trans_cap"]

        df["end_value"] = df["TxCapacityNameplate"]
        df = df[
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
        tx_agg = pd.concat([tx_agg, df])

    tx_agg.to_csv(
        os.path.join(in_folder, tocompare_folder, "transmission.csv"), index=False
    )
    df = tx_agg.copy()
    df["value"] = df["end_value"] - df["start_value"]
    df2 = df[["model", "line_name", "planning_year", "case", "unit", "value"]]
    df2.to_csv(
        os.path.join(in_folder, tocompare_folder, "transmission_expansion.csv"),
        index=False,
    )

################################### make  generation.csv

print("\ncreating generation.csv")
for i in case_list:
    if skip_case(i):
        continue
    tocompare_folder = os.path.join(
        results_folder, case_list[i], "SWITCH_results_summary"
    )
    generation_agg = pd.DataFrame()
    dispatch_agg = pd.DataFrame()
    for y in year_list:
        ts = pd.read_csv(
            os.path.join(in_folder, "26-zone/in", str(y), i, "timeseries.csv")
        )
        dispatch2030 = pd.read_csv(
            os.path.join(in_folder, "26-zone/out", str(y), i, "dispatch.csv")
        )
        df = dispatch2030.copy()

        df["model"] = "SWITCH"
        df["zone"] = df["gen_load_zone"]
        df["resource_name"] = df["generation_project"]
        df["tech_type"] = "Other"
        df.loc[
            df["resource_name"].str.contains("hydrogen", case=False), "tech_type"
        ] = "Hydrogen"
        df.loc[df["resource_name"].str.contains("batter", case=False), "tech_type"] = (
            "Battery"
        )
        df.loc[df["resource_name"].str.contains("storage", case=False), "tech_type"] = (
            "Battery"
        )
        df.loc[df["resource_name"].str.contains("coal", case=False), "tech_type"] = (
            "Coal"
        )
        df.loc[
            df["resource_name"].str.contains("solar|pv", case=False), "tech_type"
        ] = "Solar"
        df.loc[df["resource_name"].str.contains("wind", case=False), "tech_type"] = (
            "Wind"
        )
        df.loc[
            df["resource_name"].str.contains("hydro|water", case=False), "tech_type"
        ] = "Hydro"
        df.loc[
            df["resource_name"].str.contains("distributed", case=False), "tech_type"
        ] = "Distributed Solar"
        df.loc[
            df["resource_name"].str.contains("geothermal", case=False), "tech_type"
        ] = "Geothermal"
        df.loc[df["resource_name"].str.contains("nuclear", case=False), "tech_type"] = (
            "Nuclear"
        )
        df.loc[df["resource_name"].str.contains("natural", case=False), "tech_type"] = (
            "Natural Gas"
        )
        df.loc[df["resource_name"].str.contains("ccs", case=False), "tech_type"] = "CCS"

        # df["planning_year"] = y
        df["case"] = i
        df["timestep"] = "all"
        df["unit"] = "MWh"
        dp = df.copy()
        dp["value"] = dp["DispatchGen_MW"]
        from transmission_scripts.TX_util import tp_to_date

        dp = tp_to_date(dp, "timestamp")
        dp["days"] = pd.to_numeric(dp["days"])
        from datetime import date

        def converttodate(dfrow):
            return date.fromisocalendar(dfrow["period"], dfrow["week"], dfrow["days"])

        dp["date"] = dp.apply(lambda dfrow: converttodate(dfrow), axis=1)
        dp["planning_year"] = dp["period"]
        dp = dp.sort_values(by=["week"])
        week_to_report = ts.iloc[0]["timeseries"][6:]
        dp = dp.loc[dp["week"] == int(week_to_report)]
        dp = dp[
            [
                "hour",
                "resource_name",
                "value",
                "zone",
                "planning_year",
                "model",
                "case",
                "week",
                "date",
            ]
        ]
        dp = dp.loc[dp["value"] > 0]
        df["value"] = df["DispatchGen_MW"] * df["tp_weight_in_year_hrs"]
        generation = df.groupby(["resource_name", "period"], as_index=False).agg(
            {
                "value": "sum",
                "model": "first",
                "zone": "first",
                "resource_name": "first",
                "tech_type": "first",
                "case": "first",
                "timestep": "first",
                "unit": "first",
            }
        )
        generation["planning_year"] = generation["period"]

        dispatch_agg = pd.concat([dispatch_agg, dp])
        generation_agg = pd.concat([generation_agg, generation])

    dispatch_agg.to_csv(
        os.path.join(in_folder, tocompare_folder, "dispatch.csv"), index=False
    )
    generation_agg.to_csv(
        os.path.join(in_folder, tocompare_folder, "generation.csv"), index=False
    )

    # generation2030 = generation2030.rename({"DispatchGen_MW": "value"}, axis=1)


################################### make  emission.csv

print("\ncreating emission.csv")
for i in case_list:
    if skip_case(i):
        continue
    tocompare_folder = os.path.join(
        results_folder, case_list[i], "SWITCH_results_summary"
    )
    emission_agg = pd.DataFrame()
    for y in year_list:
        emission2030 = pd.read_csv(
            os.path.join(in_folder, "26-zone/out", str(y), i, "dispatch.csv")
        )
        df = emission2030.copy()

        df = df.groupby(["gen_load_zone", "period"], as_index=False).agg(
            {
                "DispatchEmissions_tCO2_per_typical_yr": "sum",
            }
        )
        df["planning_year"] = df["period"]
        df = df.rename(
            {"gen_load_zone": "zone", "DispatchEmissions_tCO2_per_typical_yr": "value"},
            axis=1,
        )
        df["model"] = "SWITCH"
        df["case"] = i
        df["unit"] = "kg"
        df["value"] = df["value"] * 1000  # from ton to kg
        df = df[["model", "zone", "planning_year", "case", "unit", "value"]]
        emission_agg = pd.concat([emission_agg, df])

    emission_agg.to_csv(
        os.path.join(in_folder, tocompare_folder, "emissions.csv"), index=False
    )


print(
    f"\nResults were added to {results_folder}.\n"
    "They are ready to be pushed to the remote repository."
)
