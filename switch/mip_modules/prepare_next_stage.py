"""
Prepare inputs for the next model stage when running a series of single-year
(myopic) models. This requires the following:

* --inputs-dir and --outputs-dir for this model are in the form
  <something>/<year>/<case_name>
* <year> is 2030 or 2040 and matches the period of this model. These will be
  passed forward to 2040 or 2050 respectively.
* an inputs dir for the next period already exists and contains gen_build_costs.csv

Then, for chained models, alternative versions of gen_build_predetermined.csv,
gen_build_costs.csv and transmission_lines.csv will be stored in the next inputs
directory, with the filename changed to <filename>.chained.<case_name>.csv.
"""

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd


def post_solve(m, outdir):
    # note: this function uses previously written model outputs instead of
    # built-in model info, so the code can be reused without running the
    # model if needed.

    # how to choose the next year in the chain
    model_years = [2027, 2030, 2035, 2040, 2045, 2050]
    next_year_dict = dict(zip(model_years[:-1], model_years[1:]))

    # we can tell which period we're dealing with by looking at m.PERIODS
    # or outdir, which should be be <root>/year/case
    in_path = Path(m.options.inputs_dir)
    out_path = Path(m.options.outputs_dir)

    year_name, case_name = out_path.parts[-2:]

    # sanity checks on directory names
    if year_name not in {str(y) for y in next_year_dict.keys()}:
        try:
            year = int(year_name)
            if 2020 <= year <= 2060:
                raise NotImplementedError(
                    f"{__name__} needs to be updated to handle model year {year}."
                )
        except:
            pass
        raise ValueError(
            f"{__name__} requires --outputs-dir in the form <root>/<year>/<case>; "
            f"'{out_path}' is not recognized."
        )
    if year_name != in_path.parts[-2]:
        raise ValueError(
            f"Year '{in_path.parts[-2]}' in --inputs-dir doesn't match '{year_name}' in --outputs-dir."
        )

    year = int(year_name)
    next_year = next_year_dict[year]

    # input dir for the first model in the chain (used as starting point)
    next_in_path = Path(
        *in_path.parts[:-2], str(next_year_dict[year]), in_path.parts[-1]
    )

    # finished preparing and validating year, year_name, case_name, in_path (this
    # model's inputs dir), out_path (this model's outputs dir) and next_in_path (
    # inputs dir for next model in the chain)

    # note: in_path may be shared between multiple cases and/or this script may
    # be run multiple times, so we generate new input files with
    # ".chained.{case_name}" appended. We also read those in as the starting
    # point for the next step in the chain when available. This means we start
    # with a 2030 predetermined build plan (and costs), then add 2030
    # construction to that, then add 2040 construction to that, not to the 2040
    # predetermined build. So some old plants may be in
    # 2040/data_case/gen_build_predetermined.chained.{scenario}.csv that aren't
    # in 2040/data_case/gen_build_predetermined.csv. This is consistent with the
    # multi-period (foresight) models, the extra plants will be ignored
    # after their retirement date, and this is easier than starting with a
    # 2050 predetermined-build file, then going back and adding 2030 and 2040
    # construction to it.

    def chained(*parts):
        """
        Return file path, joined together if needed, with fname.csv
        converted to fname.chained.{case_name}.csv.
        """
        path = Path(*parts)
        return Path(path.parent, f"{path.stem}.chained.{case_name}{path.suffix}")

    def possibly_chained(*parts):
        """
        Return file path, joined together if needed, with fname.csv
        converted to fname.chained.{case_name}.csv if that file exists.
        """
        path = Path(*parts)
        new_path = chained(path)
        return new_path if new_path.exists() else path

    def read_csv(*parts):
        return pd.read_csv(Path(*parts), na_values=["."])

    def to_csv(df, *parts):
        path = Path(*parts)
        # path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, na_rep=".")

    """
    df = predet
    filename = "gen_build_predetermined.csv"
    """

    def merge_build_data(df, filename):
        """
        Use construction data from the current dataframe with later construction
        data from the specified table in the next stage.

        df and <filename> should have generation project and build year as first
        two columns.

        This uses all data from df plus any data from <filename> that occurs
        after the start of the next period. However, if use_later_records()
        returns True, it will always drop the row from df and always use data
        from <filename>.

        Splitting in time ensures that future stages are consistent with the
        past, including any retirements (which may get dropped completely from
        the model.)
        """
        next_period_start = read_csv(next_in_path, "periods.csv")["period_start"].min()
        next_df = read_csv(next_in_path, filename)
        # make sure column names are consistent
        next_df = next_df.rename(columns=dict(zip(next_df.columns[:2], df.columns[:2])))
        # drop any use_next_record() records from incoming df
        df = df.loc[~use_next_stage_records(df), :]
        # drop data from next stage prior to the start of that stage
        next_df = next_df.loc[
            (next_df.iloc[:, 1] >= next_period_start)
            | (use_next_stage_records(next_df)),
            :,
        ]
        df = pd.concat([df, next_df], ignore_index=True)
        # The first two columns should be ["GENERATION_PROJECT", "BUILD_YEAR"]
        # but spelling/capitalization may differ.
        dup_cols = df.columns[:2]
        # save any duplicate gen/build_year with different capacities (hopefully none)
        dups = df.loc[df.duplicated(subset=dup_cols, keep=False), :]
        if not dups.empty:
            # Keep the data from the selected period, but save a copy of the
            # duplicates in case the user wants to inspect them (these are
            # fairly common because unit sizes vary between periods due to
            # derating with a capacity factor that may vary between periods, and
            # fixed O&M rises over time for some technologies in PG; retirements
            # also create differences between predetermined capacity in the
            # dataframe and in the next model).
            df = df.drop_duplicates(subset=dup_cols, keep="first")
            to_csv(dups, chained(next_in_path, "dup." + filename))

        return df

    def use_next_stage_records(df):
        """
        Accept a dataframe with generation project names in the first column and
        return a vector of True or False indicating whether we should _always_
        use data for this generator from the next stage (True) or prefer data
        from the current stage (False).

        This is only used to identify distributed generation (usually rooftop
        solar). PowerGenome doesn't have construction years for this capacity,
        so pg_to_switch.py calculates a suitable schedule. But this schedule may
        differ from one stage of a myopic model to the next, so we drop anything
        from the earlier stage and just use the schedule calculated for the
        later stage (generally constructed in the reference year of the period.)
        Note: this makes it impossible to carry distributed gen retirements
        through to later stages, but that is not allowed anyway.
        """
        return df.iloc[:, 0].str.contains("distributed_generation")

    # use actual construction from current model (includes both predetermined and
    # optimized resources) as the predetermined construction for the next stage.
    build_mw = read_csv(out_path, "BuildGen.csv").rename(
        columns={
            "GEN_BLD_YRS_1": "GENERATION_PROJECT",
            "GEN_BLD_YRS_2": "build_year",
            "BuildGen": "build_gen_predetermined",
        }
    )
    build_mwh = read_csv(out_path, "BuildStorageEnergy.csv").rename(
        columns={
            "STORAGE_GEN_BLD_YRS_1": "GENERATION_PROJECT",
            "STORAGE_GEN_BLD_YRS_2": "build_year",
            "BuildStorageEnergy": "build_gen_energy_predetermined",
        }
    )
    predet = build_mw.merge(build_mwh, how="left")

    # Treat any retired capacity as if it was never built.
    # Note: this will not carry forward capital costs of retired plants, so it
    # should only be used with plants with no capital costs. That is the
    # approach in MIP, where it only applies to existing plants, which are shown
    # with $0 capital cost.
    retire = read_csv(out_path, "SuspendGen.csv").rename(
        columns={
            "GEN_BLD_SUSPEND_YRS_1": "GENERATION_PROJECT",
            "GEN_BLD_SUSPEND_YRS_2": "build_year",
            "GEN_BLD_SUSPEND_YRS_3": "retire_year",
            "SuspendGen": "retire_mw",
        }
    )
    gen_info = read_csv(possibly_chained(in_path, "gen_info.csv"))
    # note: retire will have rows if gen_can_retire_early or gen_can_suspend
    # are set, but it is only binding for the future if gen_can_suspend is _not_
    # set but there is still a value (implying gen_can_retire_early was set).
    if not "gen_can_suspend" in gen_info.columns:
        gen_info["gen_can_suspend"] = 0
    must_retire_gens = gen_info.query("gen_can_suspend == 0").iloc[:, 0]
    retire = retire.loc[
        retire["GENERATION_PROJECT"].isin(must_retire_gens)
        & (retire["retire_mw"] > 1e-6),
        :,
    ]
    # handle chained models with multi-period stages (possible in the future)
    retire = (
        retire.groupby(["GENERATION_PROJECT", "build_year"])[["retire_mw"]]
        .sum()
        .reset_index()
    )
    predet_cols = predet.columns
    predet = predet.merge(retire, how="left")
    predet["build_gen_predetermined"] -= predet["retire_mw"].fillna(0)
    predet = predet[predet_cols]

    # drop any that don't appear in the next stage (this should never occur
    # in principle, but in practice in the MIP project there are some
    # inconsistencies and this is the only way to resolve them)
    next_gen_info = read_csv(next_in_path, "gen_info.csv")
    predet = predet.merge(next_gen_info["GENERATION_PROJECT"])

    # Merge with the predetermined construction data from the next stage.
    predet = merge_build_data(predet, "gen_build_predetermined.csv")
    # drop any that are zero or very small
    predet = predet.query(
        "build_gen_predetermined > 0.001 or build_gen_energy_predetermined > 0.001"
    )
    to_csv(predet, chained(next_in_path, "gen_build_predetermined.csv"))

    # use this model's costs for everything that was built and next model's
    # costs for anything in the next period (or later, if we eventually chain
    # multi-period models). This maintains consistency with the new
    # gen_build_predetermined.
    costs = read_csv(possibly_chained(in_path, "gen_build_costs.csv"))
    # drop any that don't match up with capacity being carried forward
    # (e.g., predetermined capacity == 0 or BuildGen == 0)
    # (match using first two cols, however they're capitalized)
    costs = costs.merge(
        predet,
        left_on=costs.columns[:2].to_list(),
        right_on=predet.columns[:2].to_list(),
        how="inner",
    )[costs.columns]
    # drop any that don't appear in the next stage
    costs = costs.merge(next_gen_info["GENERATION_PROJECT"])
    # merge cost data from this model with cost data from the next model
    # (this will use data from this model for projects/build_years from this
    # model and data from the next model for additional projects/build_years)
    costs = merge_build_data(costs, "gen_build_costs.csv")
    to_csv(costs, chained(next_in_path, "gen_build_costs.csv"))

    # combine starting transmission for this case with transmission expansion
    trans = read_csv(possibly_chained(in_path, "transmission_lines.csv"))
    trans_built = (
        read_csv(out_path, "BuildTx.csv")
        .rename(columns={"TRANS_BLD_YRS_1": "TRANSMISSION_LINE"})
        .groupby("TRANSMISSION_LINE")[["BuildTx"]]
        .sum()
        .reset_index()
    )
    trans = trans.merge(trans_built)
    trans["existing_trans_cap"] += trans["BuildTx"].fillna(0)
    # round very small capacities to zero (generally occur due to solver
    # rounding and may cause numerical warnings in next stage)
    trans.loc[trans["existing_trans_cap"] < 0.001, "existing_trans_cap"] = 0
    trans = trans.drop(columns=["BuildTx"])
    to_csv(trans, chained(next_in_path, "transmission_lines.csv"))


class Test:
    """
    generic object that can be assigned any attributes needed
    """

    pass


if __name__ == "__main__":
    # called from command line or run interactively in VS Code
    m = Test()
    m.options = Test()
    if len(sys.argv) == 3:
        # called from command line with inputs-dir and outputs-dir
        # run a test case using the specified inputs and outputs directories
        m.options.inputs_dir = sys.argv[1]
        m.options.outputs_dir = sys.argv[2]
        post_solve(m, m.options.outputs_dir)
    elif (
        len(sys.argv) == 2 and os.path.basename(sys.argv[0]) == "ipykernel_launcher.py"
    ):
        # running interactively from VS Code
        io_dir = "2024-08-04"
        year = "2030"
        case = "base_short_retire"
        root = os.path.join(os.path.dirname(__file__), "..", io_dir)
        m.options.inputs_dir = os.path.join(root, "in", year, case)
        m.options.outputs_dir = os.path.join(root, "out", year, case)
        outdir = m.options.outputs_dir
        print("Run the contents of post_solve() interactively to see results.")
    else:
        raise RuntimeError(
            "usage: python prepare_next_stage.py <inputs-dir> <outputs-dir>"
        )
