import os, json, sys
from datetime import date
import pandas as pd
import numpy as np

from transmission_scripts.TX_util import tp_to_date

# note: this will work relative to the directory where the script is run, which
# should usually be Switch-USA-PG. But for special cases, you can copy the
# inputs and outputs directories to a different location and run it there, as
# long as they are in the expected structure (26-zone/in/year/case and 26-zone/out/year/case)

################################### FOR 26 ZONE ###################################
root_folder = ""

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


def skip_case(case):
    test_in_files = [output_file(case, y, "BuildGen.csv") for y in year_list]
    test_out_files = [
        comparison_file(case, f)
        for f in [
            "resource_capacity.csv",
            "transmission.csv",
            "transmission_expansion.csv",
            "dispatch.csv",
            "generation.csv",
            "emissions.csv",
        ]
    ]

    if not all(os.path.exists(f) for f in test_in_files):
        print(f"Skipping unsolved case '{i}'.")
        return True

    if "--skip-saved" in sys.argv and all(os.path.exists(f) for f in test_out_files):
        latest_change = max(os.path.getmtime(f) for f in test_in_files)
        earliest_record = min(os.path.getmtime(f) for f in test_out_files)

        if latest_change < earliest_record:
            print(f"Skipping previously saved case '{i}'.")
            return True

    print(f"Processing case '{i}'")
    return False


def output_file(case, year, file):
    """
    Give path to output file generated for the specified case.
    """
    # This is easy because every model has its own outputs dir
    return os.path.join(root_folder, "26-zone/out", str(year), case, file)


def input_file(case, year, file):
    """
    Give path to output file generated for the specified case.
    """
    # This is tricky because the inputs dir may not have the same name as the
    # outputs dir (several models may use the same inputs with different
    # settings for each case)
    with open(output_file(case, year, "model_config.json"), "r") as f:
        case_settings = json.load(f)

    in_dir = case_settings["options"]["inputs_dir"]
    return os.path.join(root_folder, in_dir, file)


def comparison_file(case, file):
    return os.path.join(results_folder, case_list[case], "SWITCH_results_summary", file)


################################### make  resource_capacity.csv
print("\ncreating resource_capacity.csv")
for i in case_list:
    if skip_case(i):
        continue
    resource_capacity_agg = pd.DataFrame()
    for y in year_list:
        # add the retirement back to the resource list
        ###### ADD capacity
        prebuild2030 = pd.read_csv(
            input_file(
                i,
                y,
                "gen_build_predetermined.csv",
            )
        )
        # prebuild2050 = pd.read_csv(os.path.join(in_folder, "inputs_myopic2050/gen_build_predetermined.csv"))
        build2030 = pd.read_csv(output_file(i, y, "BuildGen.csv"))

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
            output_file(
                i,
                y,
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
        comparison_file(i, "resource_capacity.csv"),
        index=False,
    )


####################################### make  transmission.csv

print("\ncreating transmission.csv")
for i in case_list:
    if skip_case(i):
        continue
    tx_agg = pd.DataFrame()
    for y in year_list:
        transmission2030_new = pd.read_csv(output_file(i, y, "transmission.csv"))

        # find the existing transmission capacity
        transmission2030_ex = pd.read_csv(input_file(i, y, "transmission_lines.csv"))

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

    tx_agg.to_csv(comparison_file(i, "transmission.csv"), index=False)
    df = tx_agg.copy()
    df["value"] = df["end_value"] - df["start_value"]
    df2 = df[["model", "line_name", "planning_year", "case", "unit", "value"]]
    df2.to_csv(
        comparison_file(i, "transmission_expansion.csv"),
        index=False,
    )

################################### make  generation.csv

print("\ncreating generation.csv")
for i in case_list:
    if skip_case(i):
        continue
    generation_agg = pd.DataFrame()
    dispatch_agg = pd.DataFrame()
    for y in year_list:
        ts = pd.read_csv(input_file(i, y, "timeseries.csv"))
        df = pd.read_csv(output_file(i, y, "dispatch.csv"))

        df["model"] = "SWITCH"
        df["zone"] = df["gen_load_zone"]
        df["resource_name"] = df["generation_project"]
        df["tech_type"] = "Other"

        # create a mapping of resource_name -> tech_type and apply it back
        # to the dataframe (faster than working directly on the large df)
        techs = df[["resource_name", "tech_type"]].drop_duplicates()
        techs.loc[
            techs["resource_name"].str.contains("hydrogen", case=False), "tech_type"
        ] = "Hydrogen"
        techs.loc[
            techs["resource_name"].str.contains("batter", case=False), "tech_type"
        ] = "Battery"
        techs.loc[
            techs["resource_name"].str.contains("storage", case=False), "tech_type"
        ] = "Battery"
        techs.loc[
            techs["resource_name"].str.contains("coal", case=False), "tech_type"
        ] = "Coal"
        techs.loc[
            techs["resource_name"].str.contains("solar|pv", case=False), "tech_type"
        ] = "Solar"
        techs.loc[
            techs["resource_name"].str.contains("wind", case=False), "tech_type"
        ] = "Wind"
        techs.loc[
            techs["resource_name"].str.contains("hydro|water", case=False), "tech_type"
        ] = "Hydro"
        techs.loc[
            techs["resource_name"].str.contains("distributed", case=False), "tech_type"
        ] = "Distributed Solar"
        techs.loc[
            techs["resource_name"].str.contains("geothermal", case=False), "tech_type"
        ] = "Geothermal"
        techs.loc[
            techs["resource_name"].str.contains("nuclear", case=False), "tech_type"
        ] = "Nuclear"
        techs.loc[
            techs["resource_name"].str.contains("natural", case=False), "tech_type"
        ] = "Natural Gas"
        techs.loc[
            techs["resource_name"].str.contains("ccs", case=False), "tech_type"
        ] = "CCS"
        techs_dict = techs.set_index("resource_name")["tech_type"].to_dict()
        df["tech_type"] = df["resource_name"].map(techs_dict)

        # df["planning_year"] = y
        df["case"] = i
        df["timestep"] = "all"
        df["unit"] = "MWh"

        dp = df.copy()
        dp["value"] = dp["DispatchGen_MW"]

        # do the time conversions once and map back (for speed)
        dp_stamps = dp[["timestamp"]].drop_duplicates()
        dp_stamps = tp_to_date(dp_stamps, "timestamp")
        dp_stamps["days"] = pd.to_numeric(dp_stamps["days"])
        dp_stamps["date"] = dp_stamps.apply(
            lambda dfrow: date.fromisocalendar(
                dfrow["period"], dfrow["week"], dfrow["days"]
            ),
            axis=1,
        )
        dp_stamps = dp_stamps.rename(columns={"period": "planning_year"})
        dp = dp.merge(dp_stamps, on="timestamp")

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

    dispatch_agg.to_csv(comparison_file(i, "dispatch.csv"), index=False)
    generation_agg.to_csv(comparison_file(i, "generation.csv"), index=False)

    # generation2030 = generation2030.rename({"DispatchGen_MW": "value"}, axis=1)


################################### make  emission.csv

print("\ncreating emission.csv")
for i in case_list:
    if skip_case(i):
        continue
    emission_agg = pd.DataFrame()
    for y in year_list:
        emission2030 = pd.read_csv(output_file(i, y, "dispatch.csv"))
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

    emission_agg.to_csv(comparison_file(i, "emissions.csv"), index=False)


print(
    f"\nResults were added to {results_folder}.\n"
    "They are ready to be pushed to the remote repository."
)
