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
    "base_short_tx_0": "26z-short-base-tx-0",
    "base_short_tx_15": "26z-short-base-tx-15",
    "base_short_tx_50": "26z-short-base-tx-50",
    "base_short_tx_100": "26z-short-base-tx-100",
    "base_short_tx_200": "26z-short-base-tx-200",
    "base_short_commit": "26z-short-commit",
}

# TODO:
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
for i in case_list:
    if skip_case(i):
        continue

    build_dfs = [
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
        # find the capacity built in this model (including existing projects)
        # that is still online for the period, i.e., has
        # build year + max_age_years > model start year

        # lookup start and end years for period(s) in this model
        periods = pd.read_csv(input_file(i, y, "periods.csv")).rename(
            columns={"INVESTMENT_PERIOD": "planning_year"}
        )

        # get the construction plan (all years up through this model, but some
        # capacity may have retired before the model started)
        build_mw = pd.read_csv(output_file(i, y, "BuildGen.csv")).rename(
            columns={
                "GEN_BLD_YRS_1": "resource_name",
                "GEN_BLD_YRS_2": "build_year",
                "BuildGen": "build_mw",
            }
        )
        build_mwh = pd.read_csv(output_file(i, y, "BuildStorageEnergy.csv")).rename(
            columns={
                "STORAGE_GEN_BLD_YRS_1": "resource_name",
                "STORAGE_GEN_BLD_YRS_2": "build_year",
                "BuildStorageEnergy": "build_mwh",
            }
        )
        build = build_mw.merge(build_mwh, how="outer")

        # cross-reference with the period information for this model
        build = (
            build.assign(__x=1)
            .merge(periods.assign(__x=1), on="__x")
            .drop(columns=["__x"])
        )

        # add suspension/retirement info if available
        # TODO: avoid the retirement calculations by adding --save-expression
        # GenCapacity when solving, then read GenCapacity.csv, or alternatively,
        # pull info from gen_cap.csv instead of working from GenBuild.csv (which
        # includes the obsolete generators).
        susp_file = output_file(i, y, "SuspendGen.csv")
        if os.path.exists(susp_file):
            # get endogenous retirements (suspensions)
            suspend_mw = pd.read_csv(susp_file).rename(
                columns={
                    "GEN_BLD_SUSPEND_YRS_1": "resource_name",
                    "GEN_BLD_SUSPEND_YRS_2": "build_year",
                    "GEN_BLD_SUSPEND_YRS_3": "planning_year",
                    "SuspendGen": "suspend_mw",
                }
            )
            build = build.merge(
                suspend_mw, on=["resource_name", "build_year", "planning_year"]
            )
            missing = build[build["suspend_mw"].isna()]
            if not missing.empty:
                print(
                    f"WARNING: unexpected missing SuspendGen values found in case {i} in year {y}:"
                )
                print(missing)
                build["suspend_mw"] = build["suspend_mw"].fillna(0)
        else:
            # no suspension file
            build["suspend_mw"] = 0

        # get retirement age from gen_info.csv
        gen_info = pd.read_csv(input_file(i, y, "gen_info.csv")).set_index(
            "GENERATION_PROJECT"
        )
        build["retire_year"] = build["build_year"] + build["resource_name"].map(
            gen_info["gen_max_age"]
        )

        # find amount of capacity online before and after each period
        # (note: with retirement, "before" becomes ill-defined in myopic models,
        # because we don't know how much was suspended in the previous period,
        # so we fix that up later)
        # Assume "start" means capacity available immediately prior to running
        # this model, possibly including capacity that retired just as this
        # model started.
        cap_start = (
            build.query(
                "(build_year < period_start) & (retire_year > period_start - 1)"
            )
            .groupby(["resource_name", "planning_year"], as_index=False)[
                ["build_mw", "build_mwh"]
            ]
            .sum()
        ).rename(columns={"build_mw": "start_value", "build_mwh": "start_MWh"})
        # assume anything that made it _past_ the start of this period is still
        # there at the end, since that is what Switch does and it captures the
        # notion of "what's running in this period" (if capacity is online one
        # period and retired by the next period, it is treated as retired in the
        # second period)
        cap_end = (
            build.query("(build_year <= period_end) & (retire_year > period_start)")
            # subtract suspensions/retirements from the capacity that would
            # otherwise be online through this study period (shows as not online
            # at end)
            .assign(build_mw=lambda df: df["build_mw"] - df["suspend_mw"])
            .groupby(["resource_name", "planning_year"], as_index=False)[
                ["build_mw", "build_mwh"]
            ]
            .sum()
        ).rename(columns={"build_mw": "end_value", "build_mwh": "end_MWh"})

        build_sum = cap_start.merge(cap_end)
        # add other columns needed for the report
        build_sum["zone"] = build_sum["resource_name"].map(gen_info["gen_load_zone"])
        build_sum["tech_type"] = tech_type(build_sum["resource_name"])
        build_sum["model"] = "SWITCH"
        build_sum["planning_year"] = y
        build_sum["case"] = i
        build_sum["unit"] = "MW"

        build_dfs.append(build_sum)

    # combine and round results (there are some 1e-14's in there)
    resource_capacity_agg = pd.concat(build_dfs).round(6)

    # TODO: use previous period's end_value as start_value for next period,
    # if available; this is a better estimate than the ones above, because it
    # accounts for retirements/suspensions in the previous period (they are
    # treated as not there at the start, but there at the end if unsuspended)
    # For now, we ignore this because start_value is not used anywhere.

    # fill missing capacity values; these are generally due to old plants that
    # got carried forward to later models. They don't participate in the
    # objective or constraints, so the solver doesn't assign them values
    # resource_capacity_agg = resource_capacity_agg.fillna(
    #     {"start_value": 0, "end_value": 0, "start_MWh": 0, "end_MWh": 0}
    # )

    # drop empty rows
    resource_capacity_agg = resource_capacity_agg.loc[
        (resource_capacity_agg["end_value"] > 0)
        | (resource_capacity_agg["end_MWh"] > 0)
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
        df["tech_type"] = tech_type(df["resource_name"])
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
