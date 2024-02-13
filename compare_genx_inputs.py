"""
This script copies two Switch input directories to a temporary folder,
standardizing them on the way, then uses the opendiff command to compare them.
This can be useful for seeing whether two versions of the data pipeline produce
the same results, ignoring insignificant differences.

Standardization consists of rounding all numerical values to 10 significant
figures, converting all-int columns from float to int, and sorting rows in each
file in ascending order across all columns.

usage: python compare_inputs.py old_dir new_dir
"""

import sys, tempfile, os, shutil, subprocess
import pandas as pd, numpy as np


def round(x, n_figs):
    """
    Round series or dataframe to specified number of significant figures;
    this uses vectorized calculations so it is fairly fast.
    Note: numbers smaller than 1e-200 will be rounded to zero.
    """
    power = 10 ** (np.floor(np.log10(np.abs(x).clip(1e-200))))
    rounded = np.round(x / power, n_figs - 1) * power
    return rounded


source_dir = dict([("old", sys.argv[1]), ("new", sys.argv[2])])
common_cols = dict()

with tempfile.TemporaryDirectory() as td:
    for out, orig in source_dir.items():
        outdir = os.path.join(td, out)
        os.mkdir(outdir)

        for root, dirs, files in os.walk(orig):
            for name in dirs:
                os.mkdir(os.path.join(outdir, name))
            for name in files:
                old_file = os.path.join(root, name)
                new_file = os.path.join(outdir, name)
                print(f"copying {old_file}")
                if name.endswith(".csv"):
                    df = pd.read_csv(old_file)
                    df = df.sort_values(axis=0, by=df.columns.to_list())

                    # Sort common columns between new and old files in the same
                    # order (based on old file) and move any uncommon ones to
                    # the end
                    if out == "old":
                        # replace the 'old' path with the 'new' path
                        new_orig_file = (
                            source_dir["new"] + old_file[len(source_dir["old"]) :]
                        )
                        if os.path.exists(new_orig_file):
                            # get the column names
                            new_df = pd.read_csv(new_orig_file)
                            cc = [c for c in df.columns if c in new_df.columns]
                            if len(cc) < len(df.columns) or len(cc) < len(
                                new_df.columns
                            ):
                                common_cols[name] = cc

                    cc = common_cols.get(name, [])
                    if cc:
                        df = df.loc[:, cc + [c for c in df.columns if c not in cc]]

                    # simplify comparisons (filter out known differences)
                    if name == "Generators_data.csv":
                        # drop CCS (shouldn't be in the GenX one)
                        df = df.query(
                            'technology != "NaturalGas_CCCCSAvgCF_Conservative"'
                        )
                        # drop resource ID since they don't align
                        df = df.drop("R_ID", axis=1)
                        df = df.drop(
                            ["Inv_Cost_per_MWyr", "Inv_Cost_per_MWhyr", "wacc_real"],
                            axis=1,
                        )

                    # convert float columns to int if possible
                    floats = df.columns[df.dtypes == "float64"]
                    for c in floats:
                        # if (df[c] % 1 == 0).all():
                        #     df[c] = df[c].astype(int)
                        try:
                            df[c] = df[c].astype("Int64")  # can keep NaNs
                        except TypeError:
                            # non-convertible, e.g., has non-int or inf values
                            pass
                    # standardize on 8 digit rounding (seems to be that way in Greg's files)
                    floats = df.columns[df.dtypes == "float64"]
                    df[floats] = round(df[floats], 6)

                    if cc:
                        # make versions with and without extra columns
                        df.loc[:, [c for c in cc if c in df.columns]].to_csv(
                            new_file[:-4] + "_common_cols.csv", index=False
                        )
                        df.to_csv(new_file[:-4] + "_different_cols.csv", index=False)
                    else:
                        # no weird columns
                        df.to_csv(new_file, index=False)

                else:  # not a .csv
                    shutil.copy(old_file, new_file)

    print("These files are in the 'old' and 'new' directories in")
    print(td)

    subprocess.run(["opendiff", os.path.join(td, "old"), os.path.join(td, "new")])

    input("Press Enter to finish and delete the temporary files...")
