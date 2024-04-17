import os
import pandas as pd

dfs = []
for y in [2030, 2040, 2050]:
    dfs.append(
        pd.read_csv(
            f"https://raw.githubusercontent.com/lbeatty1/MIP_AirPollution/main/marginal_gen_exposure_coefs_{y}.csv",
            index_col=0,
        )
    )
exposure = (
    pd.concat(dfs)
    .groupby(["Cluster", "year", "Race"], as_index=False)["Exposure"]
    .sum()
).query("Exposure != 0")
exposure.columns = [
    "GENERATION_PROJECT",
    "PERIOD",
    "EXPOSURE_GROUP",
    "group_exposure_coefficient",
]

# note: in the code below we use forward slashes instead of os.sep or os.path.join()
# so that files can be constructed on a Windows machine and run on a Linux/Unix
# machine or vice versa.
base_scens = []
cap_scens = []
for case in ["base", "current_policies"]:
    case_in_dir = f"in/foresight/{case}"
    print(f"creating inputs directory {case_in_dir}")
    case_out_dir = f"out/{case}"
    exposure.to_csv(
        f"{case_in_dir}/group_exposure_coefficients.csv",
        na_rep=".",
        index=False,
    )
    base_scens.append(
        f"--scenario-name {case} --inputs-dir {case_in_dir} --outputs-dir {case_out_dir} "
        f"--include-module mip_modules.cap_pollution_exposure "
        f"--save-expression GroupExposure "
        f"--no-minimize-pollution-exposure "
    )
    for adder in [50, 100, 200]:
        case_adder = f"{case}_{adder}"
        adder_out_dir = f"out/{case_adder}"
        print(f"creating inputs for case {case_adder}")
        df = pd.DataFrame(
            {
                "system_cost_baseline_dir": case_out_dir,
                "system_cost_multiplier": 1,
                "system_cost_adder": adder * 1e9,
            },
            index=[0],
        )
        df.to_csv(
            f"in/foresight/{case}/system_cost_limit.{case_adder}.csv",
            na_rep=".",
            index=False,
        )
        cap_scens.append(
            f"--scenario-name {case_adder} --inputs-dir {case_in_dir} "
            f"--outputs-dir {adder_out_dir} "
            f"--input-alias system_cost_limit.csv=system_cost_limit.{case_adder}.csv "
            f"--include-module mip_modules.cap_pollution_exposure "
            f"--save-expression GroupExposure "
            f"--solver gurobi --solver-options-string 'method=2 crossover=0 BarHomogeneous=1' "
            # may also want to add NumericFocus=1 above, but BarHomogeneous=1 seems to be enough to avoid numerical problems in these cases
        )

print("creating scenarios_1.txt")
with open("scenarios_1.txt", "w") as f:
    for case in base_scens:
        f.write(case + "\n")

print("creating scenarios_2.txt")
with open("scenarios_2.txt", "w") as f:
    for case in cap_scens:
        f.write(case + "\n")

print("You can solve these cases by running")
print("switch solve-scenarios --scenario-list scenarios_1.txt")
print("and then")
print("switch solve-scenarios --scenario-list scenarios_2.txt")
