# Introduction

The sections below show how to use this reposotiry.

# Install miniconda
Download from https://docs.conda.io/projects/miniconda/en/latest/
Install
or whatever powergenome recommends (mainly to get git)
if you already have this or miniconda or anaconda, you can skip ahead

# Install VS Code

*copy instructions from Switch tutorial*

# Setup modeling software

Open VS Code

Set interpreter to base of your miniconda

Open a terminal pane.

Run these commands.

```
# create a minimal switch-pg environemnt with enough to bootstrap the rest
conda create -y -c conda-forge -n switch-pg mamba git
conda activate switch-pg

# clone this repository and the dependency submodules (PowerGenome and MIP_results_comparison)
git clone XXXXXXXXXXX --recurse-submodules
cd Switch-USA-PG

# Setup powergenome environment
# Create and activate the environment
mamba env update -n switch-pg -f environment.yml
mamba env update -n switch-pg -f PowerGenome/environment.yml

conda activate switch-pg
# install PowerGenome from local repository
pip install -e PowerGenome

```

# Download PowerGenome input data

```
conda activate switch-pg
python download_pg_data.py
```

# Configure PowerGenome

Open PowerGenome/powergenome.env, and change everything before "Switch-USA-PG" to match the location on your computer.

Open `MIP_results_comparison/case_settings/26-zone/settings/resources.yml` and change everything before "Switch-USA-PG" to match the location on your computer.

## Notes about PowerGenome scenario configuration

`MIP_results_comparison/case_settings/26-zone/settings` holds the settings currently used for all scenarios in this study in a collection of `*.yml` files. In addition to these, tabular data is stored in `*.csv` files. The location of the .csv files and the specific files to use for the study are identified in `extra_inputs.yml`. The location should be a subdirectory (currently `CONUS_extra_inputs`) at the same level as the `settings` folder that holds the .yml files. One special .csv file, identified by the `scenario_definitions_fn` setting (currently `MIP_results_comparison/case_settings/26-zone/CONUS_extra_inputs/conus_scenario_inputs.csv`), defines all the cases available and identifies named groups of settings to use for various aspects of the model for each one. The .yml files in turn provide specific values to use in the model, depending which group of settings is selected.

# Generate Switch inputs

Run these scripts to generate data.

```
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in/2030 --case-id base_short --year 2030
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in/2040 --case-id base_short --year 2040
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in/2050 --case-id base_short --year 2050

python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in/2030 --case-id base_short_no_ccs --year 2030
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in/2040 --case-id base_short_no_ccs --year 2040
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in/2050 --case-id base_short_no_ccs --year 2050

python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in/2030 --case-id base_short_current_policies --year 2030
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in/2040 --case-id base_short_current_policies --year 2040
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in/2050 --case-id base_short_current_policies --year 2050

# how do we prepare the foresight models?
# is there a way to prepare multiple cases and/or multiple years at the same time?
```

(Note: for comparison, you can generate GenX inputs by going to `MIP_results_comparison/case_settings/26-zone` and running `run_powergenome_multiple -sf settings -rf genx_inputs -c base_short`. They will be stored in `MIP_results_comparison/case_settings/26-zone/genx_inputs`.)

# Run Switch

*how do we handle settings that are not embedded in a directory structure, *
*e.g., --input-alias or different module lists?*

*TODO: define a switch.yml that for each scenarios/configuration*
*identifies the data scenario to use and other settings or adjustments, e.g.,*
*add an alternative carbon price file and use that. Then build scenarios_yyyy.txt files*
*to run each set of scenarios in sequence, i.e., scenarios_2030.txt, scenarios_2040.txt,*
*scenarios_foresight.txt*.

```
cd switch
switch solve --inputs-dir 26-zone/in/2030 --outputs-dir 26-zone/in/2030
```

# Prepare myopic models

Run lines 426-591 of `switch/to_myopic.py`

TODO: move this code into a new module so it runs automatically at all but the final stage of myopic models (using appropriate flags in scenarios.txt)

# Prepare result summaries for comparison

Run lines 1-425 of `switch/to_myopic.py`

Then run these commands:

```
cd MIP_results_comparison
git add .
git commit -m 'new Switch results'
git pull
git push
```

TODO: move all of this into a new module so it runs automatically when each model finishes

# Notes

To update this repository and all the submodules (PowerGenome and
MIP_results_comparison), use

```
git pull --recurse-submodules
```

To update individual submodules, just cd into the relevant directory and run
`git pull`.

After either of these, run `git add <submodule_dir>` in the main Switch-USA-PG
directory and then commit to save the updated submodules in the Switch-USA-PG
repository. This will save pointers in Switch-USA-PG showing which commit we are
using in each submodule.
