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

myopic_year_list = [2027, 2030, 2035, 2040, 2045, 2050]

myopic_case_list = {
    "base_short": "short-base-200",
    "base_short_commit": "short-base-200-commit",
    "base_short_no_ccs": "short-base-200-no-ccs",
    "base_short_retire": "short-base-200-retire",
    "base_short_tx_100": "short-base-200-tx-100",
    "base_short_tx_15": "short-base-200-tx-15",
    "base_short_tx_200": "short-base-200-tx-200",
    "base_short_tx_50": "short-base-200-tx-50",
    "current_policies_short_high_ccs": "short-current-policies",
    "current_policies_short_low_ccs": "short-current-policies-low-ccs",
    "current_policies_short_retire_high_ccs": "short-current-policies-retire",
    "current_policies_short_retire_low_ccs": "short-current-policies-retire-low-ccs",
    "base_20_week": "20-week-myopic",
    # old names
    "base_52_1000": "full-base-1000",
    "base_52_50": "full-base-50",
    # next two are the new names for these cases
    "base_52_week_co2_1000": "full-base-1000",
    "base_52_week_co2_50": "full-base-50",
    "base_52_week": "full-base-200",
    "base_52_week_commit": "full-base-200-commit",
    "base_52_week_no_ccs": "full-base-200-no-ccs",
    "base_52_week_retire": "full-base-200-retire",
    "base_52_week_tx_0": "full-base-200-tx-0",
    "base_52_week_tx_15": "full-base-200-tx-15",
    "base_52_week_tx_50": "full-base-200-tx-50",
    "current_policies_20_week": "20-week-myopic-current-policy",
    # old names
    "current_policies_52_week_commit_high_ccs": "full-current-policies-commit",
    "current_policies_52_week_high_ccs": "full-current-policies",
    "current_policies_52_week_retire_high_ccs": "full-current-policies-retire",
    # new names
    "current_policies_52_week_commit": "full-current-policies-commit",
    "current_policies_52_week": "full-current-policies",
    "current_policies_52_week": "full-current-policies-retire",
}
foresight_case_list = {
    "base_20_week": "20-week-foresight",
    # old name for this case
    "base_short_current_policies_high_ITC": "20-week-foresight-current-policy",
    # new name for this case
    "current_policies_20_week": "20-week-foresight-current-policy",
}


# case_list = {
#     "base_short_retire": "short-base-200-retire-derate",
# }


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

for case_list, year_list in [
    (myopic_case_list, myopic_year_list),
    (foresight_case_list, ["foresight"]),
]:
    for c in case_list:
        if skip_case(c):
            continue

        # make sure the necessary dir(s) exist
        os.makedirs(comparison_file(c, ""), exist_ok=True)

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

            # derate capacity according to forced outage rate to match GenX for
            # reporting and operational simulation
            gen_info = pd.read_csv(input_file(c, y, "gen_info.csv"), na_values=".")
            availability = 1 - cap["resource_name"].map(
                gen_info.set_index("GENERATION_PROJECT")["gen_forced_outage_rate"]
            )
            cap["start_value"] *= availability
            cap["end_value"] *= availability

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

        # sort by year then gen to match previous versions
        cap_agg = cap_agg.sort_values(["planning_year", "resource_name"])

        cap_agg.to_csv(
            comparison_file(c, "resource_capacity.csv"),
            index=False,
        )

    ####################################### make  transmission.csv

    print("\ncreating transmission.csv and transmission_expansion.csv")
    for c in case_list:
        if skip_case(c):
            continue
        tx_agg = pd.DataFrame()
        for y in year_list:
            transmission = pd.read_csv(output_file(c, y, "transmission.csv"))

            transmission_addition = pd.read_csv(
                output_file(c, y, "BuildTx.csv")
            ).rename(
                columns={
                    "TRANS_BLD_YRS_1": "TRANSMISSION_LINE",
                    "TRANS_BLD_YRS_2": "PERIOD",
                }
            )

            transmission = transmission.merge(transmission_addition, how="left")
            transmission["start_value"] = transmission.eval(
                "TxCapacityNameplate - BuildTx"
            ).round(5)
            transmission["end_value"] = transmission.eval("TxCapacityNameplate").round(
                5
            )

            df = transmission.copy()
            df["model"] = "SWITCH"
            df["line_name"] = df["trans_lz1"] + "_to_" + df["trans_lz2"]
            df["planning_year"] = df["PERIOD"]
            df["case"] = c
            df["unit"] = "MW"

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
        generation_dfs = []
        dispatch_dfs = []
        for y in year_list:
            df = pd.read_csv(output_file(c, y, "dispatch.csv"), engine="pyarrow")

            # use first timestamp from dispatch.csv to identify first week for reporting
            first_timestamp = df["timestamp"].iloc[0]

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
            week_to_report = dp_stamps.loc[
                dp_stamps["timestamp"] == first_timestamp, "week"
            ].iloc[0]

            dp = dp.loc[dp["week"] == week_to_report]
            dp = dp[
                [
                    "hour",
                    "resource_name",
                    "value",
                    "zone",
                    "planning_year",
                    "model",
                    "case",
                ]
            ]
            # filter out unused resources (seem to be very few)
            non_zeros = (
                dp.groupby("resource_name")["value"]
                .sum()
                .reset_index()
                .query("value > 1e-5")["resource_name"]
            )
            dp = dp.loc[dp["resource_name"].isin(non_zeros), :]

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

            dispatch_dfs.append(dp)
            generation_dfs.append(generation)
            # save memory
            del df, dp, generation

        dispatch_agg = pd.concat(dispatch_dfs)
        del dispatch_dfs
        generation_agg = pd.concat(generation_dfs)
        del generation_dfs

        dispatch_agg.to_csv(comparison_file(c, "dispatch.csv.gz"), index=False)
        generation_agg.to_csv(comparison_file(c, "generation.csv"), index=False)

        # save memory
        del dispatch_agg, generation_agg

    ################################### make  emission.csv

    print("\ncreating emission.csv")
    for c in case_list:
        if skip_case(c):
            continue
        emission_agg = pd.DataFrame()
        for y in year_list:
            df = pd.read_csv(output_file(c, y, "dispatch_zonal_annual_summary.csv"))

            df = df.groupby(["gen_load_zone", "period"], as_index=False).agg(
                {
                    "DispatchEmissions_tCO2_per_typical_yr": "sum",
                }
            )
            df["planning_year"] = df["period"]
            df = df.rename(
                {
                    "gen_load_zone": "zone",
                    "DispatchEmissions_tCO2_per_typical_yr": "value",
                },
                axis=1,
            )
            df["model"] = "SWITCH"
            df["case"] = c
            df["unit"] = "kg"
            df["value"] = df["value"] * 1000  # from tonne to kg
            df = df[["model", "zone", "planning_year", "case", "unit", "value"]]
            emission_agg = pd.concat([emission_agg, df])

        emission_agg.to_csv(comparison_file(c, "emissions.csv"), index=False)

        # save memory
        del df, emission_agg

    print(
        f"\nResults were added to {results_folder}.\n"
        "They are ready to be pushed to the remote repository."
    )
