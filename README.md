# Introduction

The sections below show how to use this repository. Note that this will take
about 20 GB of storage on your computer before you start generating model
inputs. This is mainly used by the `MIP_results_comparison` sub-repository,
which holds one copy of all the comparison data (6.2 GB) in the active file
system and one copy in the underlying git database, and also by the PowerGenome
input files that will be stored in the `pg_data` directory (9.2 GB).

# Install miniconda
Download from https://docs.conda.io/projects/miniconda/en/latest/
Install
or whatever powergenome recommends (mainly to get git)
if you already have this or miniconda or anaconda, you can skip ahead

# Install VS Code and Python Extensions

We assume you are using the Visual Studio Code (VS Code) text editor to view and
edit code and data files and run Switch. You can use a different text editor
(and terminal app) if you like, but it should be capable of doing
programming-oriented tasks, like quickly adjusting the indentation of many lines
in a text file. If you prefer, you can also open the .csv data files directly in
your spreadsheet software instead of using VS Code.

Download and install the VS Code text editor from https://code.visualstudio.com/.

If you need more information on installing VS Code, see
https://code.visualstudio.com/docs/setup/setup-overview. (On a Mac you may need
to double-click on the downloaded zip file to uncompress it, then use the Finder
to move the “Visual Studio Code” app from your download folder to your
Applications folder.)

If you'd like a quick introduction to VS Code, see
https://code.visualstudio.com/docs.

Launch Visual Studio Code from the Start menu (Windows) or Applications folder
(Mac). You can choose a color theme and/or work your way through the “Get
Started” steps (it’s a scrollable list), or you can skip them if you don’t want
to do that now.

Follow these steps to install the Python extension for VS Code:

- Click on the Extensions icon on the Activity Bar on the left side of the
  Visual Studio Code window (or choose View > Extensions from the menu). The
  icon looks like four squares.
- This will open the Extensions pane on the left side of the window. Type
  “Python” in the search box, then click on “Install” next to the Python
  extension that lists Microsoft as the developer:
- After installing the Python extension, you will see a “Get started with Python
  development” tab and a “Get started with Jupyter Notebooks” tab. You can close
  these.

Follow these steps to install two more extensions that will be useful. These are
optional, but they make it easier to read and edit data stored in text files,
such as the .csv files used by Switch:

- Type “rainbow csv” in the search box in the Extensions pane, then click on
  “Install” next to the Rainbow CSV extension (this is optional, but makes it
  easier to read and edit data stored in text files, such as the .csv files used
  by Switch):
- Type “excel viewer” in the search box, then click to install the Excel Viewer
  extension (this is also optional, but gives a nice grid view of .csv files):


# Setup modeling software

Open VS Code.

Press shift-control-P (Windows) or shift-cmd-P (Mac). Choose `Python: Select
Interpreter`, then select the Python interpreter you installed in the previous
step (you may be able to find it by searching for "base").

Open a terminal pane: Terminal > New Terminal

Run these commands in the terminal pane.

```
# create a minimal switch-pg environement with enough to bootstrap the rest
conda create -y -c conda-forge -n switch-pg python=3.10 mamba git ipykernel
conda activate switch-pg

# clone this repository and the dependency submodules (PowerGenome and MIP_results_comparison)
cd <wherever you want the Switch-USA-PG code>
git clone https://github.com/switch-model/Switch-USA-PG --recurse-submodules
cd Switch-USA-PG

# Setup powergenome environment
# Create and activate the environment
mamba env update -n switch-pg -f environment.yml
mamba env update -n switch-pg -f PowerGenome/environment.yml

conda activate switch-pg
# install PowerGenome from local sub-repository
pip install -e PowerGenome

```

Close the current VS Code window. Then choose File > Open, then navigate to the
Switch-USA-PG folder and choose "Open". You can repeat this step anytime you
want to work with this repository in the future.

Set VS Code to use the switch-pg Python environment for future work:
shift-ctrl/cmd-P > `Python: Select Interpreter` > search for `switch-pg` > enter

# Download PowerGenome input data

In VS Code, choose Terminal > New Terminal, then run these commands in the
terminal pane (inside the Switch-USA-PG directory):

```
conda activate switch-pg
python download_pg_data.py
```

# Configure PowerGenome

In VS Code, click the Explorer icon at the top left corner, then open
PowerGenome, right click on the powergenome folder inside and create a new file
called ".env". Then paste the following into PowerGenome/powergenome/.env,
replacing <Switch-USA-PG> with the full path to your `Switch-USA-PG` directory.
(You can get the text for this path by right-clicking in the blank space below
the files in the VS Code Explorer Pane on the left, then choosing "Copy Path".
Then you can paste this into powergenome.env instead of <Switch-USA-PG>.)

```
PUDL_DB="<Switch-USA-PG>/pg_data/PowerGenome Data Files/PUDL Data/pudl.sqlite"
PG_DB="<Switch-USA-PG>/pg_data/PowerGenome Data Files/pg_misc_tables_efs.sqlite"
RESOURCE_GROUP_PROFILES="<Switch-USA-PG>/pg_data/PowerGenome Data Files/PowerGenome Resource Groups/generation_profiles"
EFS_DATA="<Switch-USA-PG>/pg_data/PowerGenome Data Files/efs_files_utc"
DISTRIBUTED_GEN_DATA="<Switch-USA-PG>/pg_data/PowerGenome Data Files"
# defined in resources.yml, so not set here
# RESOURCE_GROUPS=<not set>
```

Also create a `MIP_results_comparison/case_settings/26-zone/settings/env.yml`
file with the following line (replace`<Switch-USA-PG>` with the path to your
`Switch-USA-PG` directory) change the `` setting to

```
RESOURCE_GROUPS: "<Switch-USA-PG>/pg_data/corrected-20z-resource-groups"
```

## Notes about PowerGenome scenario configuration

`MIP_results_comparison/case_settings/26-zone/settings` holds the settings
currently used for all scenarios in this study in a collection of `*.yml` files.
In addition to these, tabular data is stored in `*.csv` files. The location of
the .csv files and the specific files to use for the study are identified in
`extra_inputs.yml`. The location should be a subdirectory (currently
`CONUS_extra_inputs`) at the same level as the `settings` folder that holds the
.yml files. One special .csv file, identified by the `scenario_definitions_fn`
setting (currently
`MIP_results_comparison/case_settings/26-zone/CONUS_extra_inputs/conus_scenario_inputs.csv`),
defines all the cases available and identifies named groups of settings to use
for various aspects of the model for each one. The .yml files in turn provide
specific values to use in the model, depending which group of settings is
selected.

# Generate Switch inputs

Run these scripts to generate data. (You can run individual ones as needed for
testing.)

```
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in/ --case-id base_short --case-id base_short_no_ccs --case-id base_short_current_policies --myopic
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in/ --case-id base_short
```

By default, `pg_to_switch.py` will generate multi-year (foresight) models for each case-id, with each model using all available years of data. If you'd like to make single-period (myopic)) models, you can use the `--myopic` flag. To generate data for a specific year, use `--year NNNN`. To generate data for multiple specific years, use `--year MMMM --year NNNN`, etc.

If you omit the `--case-id` flag, `pg_to_switch.py` will generate inputs for all available cases. This is probably not a good idea, since they will require a lot of space and many of them will not be needed.

(Note: for comparison, you can generate GenX inputs by going to
`MIP_results_comparison/case_settings/26-zone` and running
`run_powergenome_multiple -sf settings -rf genx_inputs -c base_short`. They will
be stored in `MIP_results_comparison/case_settings/26-zone/genx_inputs`.)

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
switch solve --inputs-dir 26-zone/in/2030/base_short --outputs-dir 26-zone/out/2030/base_short
```

# Prepare next stage of myopic models

Run lines 426-591 of `switch/to_myopic.py` to prepare the next stage of a myopic
model, e.g., a 2040 model that follows from a 2030 plan.

TODO: move this code into a new module so it runs automatically at all but the
final stage of myopic models (using appropriate flags in scenarios.txt)

*(Do they need to change settings to indicate which model and which periods are
being chained?)*

# Prepare result summaries for comparison

Run lines 1-425 of `switch/to_myopic.py` to prepare standardized results and
copy them to the MIP_results_comparison sub-repository *(does this work?)*.

Then run these commands:

```
cd MIP_results_comparison
git add .
git commit -m 'new Switch results'
git pull
git push
```

TODO: move all of this into a new module so it runs automatically when each
model finishes

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
