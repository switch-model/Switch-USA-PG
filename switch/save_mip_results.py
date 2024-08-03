import os, json, sys
from datetime import date
import pandas as pd
import numpy as np

from transmission_scripts.TX_util import tp_to_date

# - Switch model inputs and outputs should be in a "26-zone" directory
#   inside the same directory as this script.
# - MIP_results_comparison directory should be in the directory one level above
#   this script

################################### FOR 26 ZONE ###################################
root_folder = os.path.dirname(__file__)

# results folder should be one level up from this script
model_folder = os.path.join(root_folder, "26-zone")
results_folder = os.path.abspath(
    os.path.join(root_folder, "..", "MIP_results_comparison")
)

year_list = [2027, 2030, 2035, 2040, 2045, 2050]
case_list = {
    # local_dir: MIP_results_comparison_dir
    # "base_short": "26z-short-base-200-new",
    # "base_short_co2_1000": "26z-short-base-1000",
    # "base_short_co2_50": "26z-short-base-50",
    # "base_short_current_policies": "26z-short-current-policies",
    # "base_short_no_ccs": "26z-short-no-ccs",
    # "base_short_retire": "26z-short-retire",
    # "base_short_tx_0": "26z-short-base-tx-0",
    # "base_short_tx_15": "26z-short-base-tx-15",
    # "base_short_tx_50": "26z-short-base-tx-50",
    # "base_short_tx_100": "26z-short-base-tx-100",
    # "base_short_tx_200": "26z-short-base-tx-200",
    # "base_short_commit": "26z-short-commit",
    # "base_short_retire": "short-base-200-retire",
    # "base_short": "short-base-200",
    # "base_short_current_policies": "short-current-policies",
    # "base_short_current_policies_retire": "short-current-policies-retire",
    "base_52_week": "full-base-200",
    "base_52_week_retire": "full-base-200-retire",
    "base_52_week_no_ccs": "full-base-200-no-ccs",
    "base_52_week_tx_0": "full-base-200-tx-0",
    "base_52_week_tx_15": "full-base-200-tx-15",
    "base_52_week_tx_50": "full-base-200-tx-50",
    # "base_short_commit":
    "base_52_week_commit": "full-base-200-commit",
    # "current_policies_52_week_high_ccs": "full-current-policies",
    # "current_policies_52_week_retire_high_ccs": "full-current-policies_retire"
}


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
        print(f"Skipping unsolved case '{c}'.")
        breakpoint()
        return True

    if "--skip-saved" in sys.argv and all(os.path.exists(f) for f in test_out_files):
        latest_change = max(os.path.getmtime(f) for f in test_in_files)
        earliest_record = min(os.path.getmtime(f) for f in test_out_files)

        if latest_change < earliest_record:
            print(f"Skipping previously saved case '{c}'.")
            return True

    print(f"Processing case '{c}'")
    return False


def output_file(case, year, file):
    """
    Give path to output file generated for the specified case.
    """
    # This is easy because every model has its own outputs dir
    return os.path.join(model_folder, "out", str(year), case, file)


def input_file(case, year, file):
    """
    Give path to output file generated for the specified case.
    """
    # This is tricky because the inputs dir may not have the same name as the
    # outputs dir (several models may use the same inputs with different
    # settings for each case) and the inputs file could be an alternative
    # version for this scenario and/or chained from the previous period
    # (for myopic models)
    with open(output_file(case, year, "model_config.json"), "r") as f:
        case_settings = json.load(f)

    # lookup correct inputs_dir
    in_dir = case_settings["options"]["inputs_dir"]
    # apply input_alias if specified
    alias_list = case_settings["options"].get("input_aliases", [])
    aliases = dict(x.split("=") for x in alias_list)
    file = aliases.get(file, file)

    return os.path.join(root_folder, in_dir, file)


def comparison_file(case, file):
    return os.path.join(results_folder, case_list[case], "SWITCH_results_summary", file)


def tech_type(gen_proj):
    """
    Accept a vector of generation projects / resources, and return
    a vector of tech types for them.
    """
    df = pd.DataFrame({"resource_name": gen_proj.unique()})
    df["tech_type"] = "Other"
    patterns = [
        ("hydrogen", "Hydrogen"),
        ("batter", "Battery"),
        ("storage", "Battery"),
        ("coal", "Coal"),
        ("solar|pv", "Solar"),
        ("wind", "Wind"),
        ("hydro|water", "Hydro"),
        ("distribute", "Distributed Solar"),
        ("geothermal", "Geothermal"),
        ("nuclear", "Nuclear"),
        ("natural", "Natural Gas"),
        ("ccs", "CCS"),
    ]
    for pat, tech_type in patterns:
        df.loc[df["resource_name"].str.contains(pat, case=False), "tech_type"] = (
            tech_type
        )

    # Now use a mapping to assign all of them at once. For very long vectors,
    # this is much faster than running the code above on the original vector.
    all_tech_types = gen_proj.map(df.set_index("resource_name")["tech_type"])
    return all_tech_types


################################### make  resource_capacity.csv
# resource_name,zone,tech_type,model,planning_year,case,unit,start_value,end_value,start_MWh,end_MWh


print("\ncreating resource_capacity.csv")
for c in case_list:
    if skip_case(c):
        continue

    cap_dfs = [
        pd.DataFrame(
            columns=[
                "resource_name",
                "zone",
                "tech_type",
                "model",
                "planning_year",
                "case",
                "unit",
                "start_value",
                "end_value",
                "start_MWh",
                "end_MWh",
            ]
        )
    ]
    for y in year_list:
        # find the capacity online at the start and end of each period

        # get capacity online and retirements in effect from gen_cap.csv;
        # get amount built during the period from gen_build.csv.
        cap = pd.read_csv(output_file(c, y, "gen_cap.csv"), na_values=".").merge(
            pd.read_csv(output_file(c, y, "gen_build.csv"), na_values=".")
        )
        cap = cap.rename(
            columns={
                "GENERATION_PROJECT": "resource_name",
                "PERIOD": "planning_year",
                "gen_load_zone": "zone",
                "GenCapacity": "end_value",
                "GenStorageCapacity": "end_MWh",
            }
        )

        # Fill missing capacity values; these are generally due to old plants
        # that got carried forward to later models. They don't participate in
        # the objective or constraints, so the solver doesn't assign them values.
        # This also converts the MWh values from NaN to 0 for the non-storage
        # generators, which seems to be the preferred treatment.
        cap = cap.fillna(0.0)

        # Infer starting values from ending values. These may not match the
        # previous period's ending value if existing capacity aged out right
        # after the previous period. But these values aren't used for anything
        # and this probably matches GenX's values.
        cap["start_value"] = cap.eval("end_value + SuspendGen_total - BuildGen")
        # note: MWh can't be retired
        cap["start_MWh"] = cap.eval("end_MWh - BuildStorageEnergy")

        cap = cap[
            [
                "resource_name",
                "zone",
                "planning_year",
                "start_value",
                "start_MWh",
                "end_value",
                "end_MWh",
            ]
        ]

        # drop resources that have zero capacity at the start and end
        cap = cap.query(
            "start_value > 0 or end_value > 0 or start_MWh > 0 or end_MWh > 0"
        )
        # add other columns needed for the report
        cap["tech_type"] = tech_type(cap["resource_name"])
        cap["model"] = "SWITCH"
        cap["planning_year"] = y
        cap["case"] = c
        cap["unit"] = "MW"

        cap_dfs.append(cap)

    # combine and round results (there are some 1e-14's in there)
    cap_agg = pd.concat(cap_dfs).round(6)

    # drop empty rows
    cap_agg = cap_agg.loc[(cap_agg["end_value"] > 0) | (cap_agg["end_MWh"] > 0)]
    cap_agg.to_csv(
        comparison_file(c, "resource_capacity.csv"),
        index=False,
    )

####################################### make  transmission.csv

print("\ncreating transmission.csv")
for c in case_list:
    if skip_case(c):
        continue
    tx_agg = pd.DataFrame()
    for y in year_list:
        transmission2030_new = pd.read_csv(output_file(c, y, "transmission.csv"))

        # find the existing transmission capacity
        transmission2030_ex = pd.read_csv(input_file(c, y, "transmission_lines.csv"))

        transmission2030 = transmission2030_new.merge(transmission2030_ex, how="left")

        df = transmission2030.copy()
        df["model"] = "SWITCH"
        df["line_name"] = df["trans_lz1"] + "_to_" + df["trans_lz2"]
        df["planning_year"] = df["PERIOD"]
        df["case"] = c
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

    tx_agg.to_csv(comparison_file(c, "transmission.csv"), index=False)
    df = tx_agg.copy()
    df["value"] = df["end_value"] - df["start_value"]
    df2 = df[["model", "line_name", "planning_year", "case", "unit", "value"]]
    df2.to_csv(
        comparison_file(c, "transmission_expansion.csv"),
        index=False,
    )

################################### make  generation.csv

print("\ncreating generation.csv")
for c in case_list:
    if skip_case(c):
        continue
    generation_agg = pd.DataFrame()
    dispatch_agg = pd.DataFrame()
    for y in year_list:
        ts = pd.read_csv(input_file(c, y, "timeseries.csv"))
        df = pd.read_csv(output_file(c, y, "dispatch.csv"))

        df["model"] = "SWITCH"
        df["zone"] = df["gen_load_zone"]
        df["resource_name"] = df["generation_project"]
        df["tech_type"] = tech_type(df["resource_name"])
        # df["planning_year"] = y
        df["case"] = c
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

    dispatch_agg.to_csv(comparison_file(c, "dispatch.csv"), index=False)
    generation_agg.to_csv(comparison_file(c, "generation.csv"), index=False)

    # generation2030 = generation2030.rename({"DispatchGen_MW": "value"}, axis=1)


################################### make  emission.csv

print("\ncreating emission.csv")
for c in case_list:
    if skip_case(c):
        continue
    emission_agg = pd.DataFrame()
    for y in year_list:
        emission2030 = pd.read_csv(output_file(c, y, "dispatch.csv"))
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
        df["case"] = c
        df["unit"] = "kg"
        df["value"] = df["value"] * 1000  # from ton to kg
        df = df[["model", "zone", "planning_year", "case", "unit", "value"]]
        emission_agg = pd.concat([emission_agg, df])

    emission_agg.to_csv(comparison_file(c, "emissions.csv"), index=False)


print(
    f"\nResults were added to {results_folder}.\n"
    "They are ready to be pushed to the remote repository."
)
