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
# add some tools to your base environment to use for installing the rest
# (if you prefer not to alter your base environment, you could add these to a
# "pre-install" environment and use that for the initial setup)
conda install -y -c conda-forge mamba git

# clone this repository and the dependency submodules (PowerGenome and MIP_results_comparison)
cd <wherever you want the Switch-USA-PG code>
git clone https://github.com/switch-model/Switch-USA-PG --recurse-submodules --depth=1
cd Switch-USA-PG

# Create and activate powergenome environment
mamba create -y -c conda-forge -n switch-pg python=3.10 mamba git ipykernel
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

# Download PowerGenome input data and configure PowerGenome to use it

In VS Code, choose Terminal > New Terminal, then run these commands in the
terminal pane (inside the Switch-USA-PG directory):

```
conda activate switch-pg

python download_pg_data.py
```

## Notes about PowerGenome scenario configuration

`MIP_results_comparison/case_settings/26-zone/settings-atb2023` holds the settings
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

To setup one model case for one year for testing, you can run this command:

```
# setup one example case (specify case-id and year)
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings-atb2023 switch/26-zone/in/ --case-id base_short --year 2050
```

The `pg_to_switch.py` script uses settings from the first directory you specify
(`MIP_results_comparison/case_settings/26-zone/settings-atb2023`) and places
Switch model input files below the second directory you specify
(`switch/26-zone/in/`).

To generate data for a specific model case, use `--case-id <case_name>`. To
generate data for multiple cases, use `--case-id <case_1> --case-id <case_2>`,
etc. If you omit the `--case-id` flag, `pg_to_switch.py` will generate inputs
for all available cases.

Similarly, to generate data for a specific year, use `--year NNNN`, for multiple
years, use `--year MMMM --year NNNN`, etc. If you omit the `--year` flag,
`pg_to_switch.py` will generate inputs for all available years.

By default, `pg_to_switch.py` will generate foresight models for each case-id
when multiple years are requested. In this case, each model will use all
available years of data. If you'd like to make single-period (myopic) models,
you can use the `--myopic` flag.

For the MIP project, most cases were setup as myopic models, where one model was
created for each case for each reference year, then they were solved in
sequence, from the first to the last, with extra code to carry construction
plans and retirements forward to later years.

The following commands will generate all model data for the MIP study.

```
# setup all myopic cases (don't specify case-id)
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings-atb2023 switch/26-zone/in/ --myopic
# setup foresight cases (only for two main cases)
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings-atb2023 switch/26-zone/in/ --case-id base_20_week --case-id current_policies_20_week
```

On an HPC system that uses the slurm scheduling manager, this can be done as
follows:

```
srun setup_cases.slurm
```

(Note: for comparison, you can generate GenX inputs by running `mkdir -p
genx/in`, then `run_powergenome_multiple -sf
MIP_results_comparison/case_settings/26-zone -rf genx/in -c base_short`. They
will stored in `genx/in`.)

# Run Switch

You can solve one case for one year like this:

```
cd switch
switch solve --inputs-dir 26-zone/in/2030/base_52_week_2027 --outputs-dir 26-zone/out/2030/base_52_week_2027
```

This works well for the foresight cases, which only have one model to solve per
case. However, for the myopic cases, it is necessary to solve each year in turn
and chain the results forward to the next stage. The chaining can be done by
adding `--include-module mip_modules.prepare_next_stage` to the command line for
all but the last stage and adding
`--input-aliases gen_build_predetermined.csv=gen_build_predetermined.chained.base_short.csv gen_build_costs.csv=gen_build_costs.chained.base_short.csv transmission_lines.csv=transmission_lines.chained.base_short.csv`
for all but the first stage. (The `prepare_next_stage` module prepares
alternative inputs for the next stage that include the construction plan from
the current stage. Then the `--input-aliases` flag tells Switch to use those
alternative inputs.)

So you _could_ solve the myopic version of the `base_short` model with these commands (but there's a better option, see below):

```
cd switch
switch solve --inputs-dir 26-zone/in/2027/base_short --outputs-dir 26-zone/out/2027/base_short  --include-module mip_modules.prepare_next_stage
switch solve --inputs-dir 26-zone/in/2030/base_short --outputs-dir 26-zone/out/2030/base_short  --include-module mip_modules.prepare_next_stage --input-aliases gen_build_predetermined.csv=gen_build_predetermined.chained.base_short.csv gen_build_costs.csv=gen_build_costs.chained.base_short.csv transmission_lines.csv=transmission_lines.chained.base_short.csv
switch solve --inputs-dir 26-zone/in/2035/base_short --outputs-dir 26-zone/out/2035/base_short  --include-module mip_modules.prepare_next_stage --input-aliases gen_build_predetermined.csv=gen_build_predetermined.chained.base_short.csv gen_build_costs.csv=gen_build_costs.chained.base_short.csv transmission_lines.csv=transmission_lines.chained.base_short.csv
switch solve --inputs-dir 26-zone/in/2040/base_short --outputs-dir 26-zone/out/2040/base_short  --include-module mip_modules.prepare_next_stage --input-aliases gen_build_predetermined.csv=gen_build_predetermined.chained.base_short.csv gen_build_costs.csv=gen_build_costs.chained.base_short.csv transmission_lines.csv=transmission_lines.chained.base_short.csv
switch solve --inputs-dir 26-zone/in/2045/base_short --outputs-dir 26-zone/out/2045/base_short  --include-module mip_modules.prepare_next_stage --input-aliases gen_build_predetermined.csv=gen_build_predetermined.chained.base_short.csv gen_build_costs.csv=gen_build_costs.chained.base_short.csv transmission_lines.csv=transmission_lines.chained.base_short.csv
switch solve --inputs-dir 26-zone/in/2050/base_short --outputs-dir 26-zone/out/2050/base_short  --input-aliases gen_build_predetermined.csv=gen_build_predetermined.chained.base_short.csv gen_build_costs.csv=gen_build_costs.chained.base_short.csv transmission_lines.csv=transmission_lines.chained.base_short.csv
```

To simplify solving myopic models, `pg_to_switch.py` creates scenario definition
files in the `switch/26-zone/in` directory, with names like
`scenarios_<case_name>.txt`. The `switch solve-scenarios` command can use these
to solve all the steps in sequence. (Each one contains the command line flags
needed for each stage of the model, and `swtich solve-scenarios` solves each one
in turn.) So you can solve the reference case (`base_52_week`) with this
command:

```
cd switch
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_foresight.txt
```

The `pg_to_switch.py` command also creates scenario definition files for some
alternative cases that share the same inputs directory as the standard cases,
but use alternative versions of some input files (currently only the carbon
price file). The definitions for these can also be found in `26-zone/in/`, and
they can be solved the same way as the standard cases, e.g.,
`switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_co2_50.txt`.
You can also look inside these to see the extra flags used setup these cases.

To run all the cases for the MIP study, you can use the following commands:

```
cd switch

# myopic cases
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_20_week.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_co2_50.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_co2_1000.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_commit.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_no_ccs.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_retire.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_tx_0.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_tx_15.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_tx_50.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_current_policies_20_week.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_current_policies_52_week.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_current_policies_52_week_commit.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_current_policies_52_week_retire.txt

# foresight cases
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_20_week_foresight.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_current_policies_20_week_foresight.txt
```

Note: If you ever need to manually create next-stage inputs from a previous
stage's outputs, you can run a command like this:

```
cd switch
# prepare 2035 inputs from 2030 model (specify 2030 inputs and outputs directories)
python -m mip_modules.prepare_next_stage 26-zone/in/2030/base_short 26-zone/out/2030/base_short
# or:
python mip_modules/prepare_next_stage.py 26-zone/in/2030/base_short 26-zone/out/2030/base_short
```

# Prepare result summaries for comparison

After solving the models, run these commands to prepare standardized results and
copy them to the `MIP_results_comparison` sub-repository.

```
cd MIP_results_comparison
git pull
cd ../switch
python save_mip_results.py
cd ../MIP_results_comparison
git add .
git commit -m 'new Switch results'
git push
```

TODO: maybe move all of this into a switch module so it runs automatically when
each case finishes

# Notes

To update this repository and all the submodules (PowerGenome and
MIP_results_comparison), use

```
git pull --recurse-submodules
```

To update a submodule, `cd` into the relevant directory and run `git pull`. Then
run `git add <submodule_dir>` and `git commit` in the main Switch-USA-PG
directory to save the updated submodules in the Switch-USA-PG repository. This
will save pointers in Switch-USA-PG showing which commit we are using in each
submodule.
