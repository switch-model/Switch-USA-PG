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
    next_year_dict = {2030: 2040, 2040: 2050}

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
            "Year '{in_path.parts[-2]}' in --inputs-dir doesn't match '{year_name}' in --outputs-dir."
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
    # drop unbuilt options
    predet = predet.query(
        "build_gen_predetermined > 0 | build_gen_energy_predetermined > 0"
    )
    to_csv(predet, chained(next_in_path, "gen_build_predetermined.csv"))

    # use this model's costs for everything that was built and next model's
    # costs for anything in the next period (or later, if we eventually chain
    # multi-period models). This maintains consistency with the new
    # gen_build_predetermined.
    costs_predet = read_csv(possibly_chained(in_path, "gen_build_costs.csv"))
    # filter to only cover predetermined builds
    costs_predet = costs_predet.merge(predet[["GENERATION_PROJECT", "build_year"]])
    # get costs for future construction
    costs_new = read_csv(next_in_path, "gen_build_costs.csv")
    # filter to only cover future builds
    costs_new = costs_new.loc[costs_new["build_year"] > year, :]
    costs = pd.concat([costs_predet, costs_new])
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
    trans = trans.drop(columns=["BuildTx"])
    to_csv(trans, chained(next_in_path, "transmission_lines.csv"))


class Test:
    pass


if __name__ == "__main__" and len(sys.argv) == 3:
    # run a test case using the specified inputs and outputs directories
    m = Test()
    m.options = Test()
    m.options.inputs_dir = sys.argv[1]
    m.options.outputs_dir = sys.argv[2]
    post_solve(m, m.options.outputs_dir)
