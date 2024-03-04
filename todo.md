todo:

(- means to do, + means done)

+ make sure switch can run with gen_build_predetermined after start of study (may cause problems in post solve); if not, drop any unusable gens in pg_to_switch

- add retire_early and retire_late options to gen_build_suspend and set that in options.txt; will also need to add the can_suspend flags below, so we can use this
module full-time

- add hydrogen as a fuel

+ add gen_can_suspend and gen_can_retire_early settings (both default to 0) to gen_info.csv for use by gen_build_suspend (if gen_can_retire and not gen_can_suspend, require later suspensions >= previous suspensions for each gen/vintage)

- add --retire {early, mid, late} flag for gen_build, to control whether
  retirement (and construction) are moved to the start, middle or end of the
  period when they occur (which affects how we decide what is online during that
  period)

+ resolve cases where gen_build_predetermined for 2040 has capacity that wasn't in the 2030 file (e.g., with date of construction in 2025) or has less capacity than the 2030 file (e.g., BASN_natural_gas_fired_combined_cycle_1,1994 is 141 Mw in 2030 and 53 MW in 2040). Could the second part be something about clustering plants that are already scheduled to retire? Do we not use a uniform retirement age for each cluster?
+ get fixed O&M to show up for existing plants so there is money to save by retiring them
- update transmission scenario definitions to all branch off base_short instead of using separate dirs for most
- add code to pg_to_switch to auto-generate scenarios.txt (for all possible cases), depending on whether it is a pre-chain year, post-chain year, which transmission and co2 costs are used, whether retirement or ramp limits are enabled, etc.
- make dirs a more readable color in ls on hpc
- setup .slurm scripts to run the scenarios as array jobs with different jobs for different years and dependencies between them (can we have dependencies between tasks, so 2040 case 1 depends on 2030 case 1?)
- setup hpc to share history right away between sessions
- incorporate early retirements into reported capacity for MIP

(later) grapple somehow with the fact that powergenome thinks it's setting up scenario dirs, but really it's data dirs, and then a scenario consists of a data dir plus some other settings
(later) implement transmission-limit cases in pipeline (caps)

generate case data: base_short, base_short_current_policies, base_short_no_ccs


create a module to limit transmission additions on each corridor to a specified amount

create a script to setup the alternative cases (alternative input files for base_short; flags in scenarios.txt):
+ base:
  + need to add a script or module to generate next-stage gen_build_predetermined from prior stage
+ transmission limits
  + update pg_switch.py: use powergenome.GenX.network_max_reinforcement to find expansion limits per line per period
+ carbon slack
  + update pg_switch.py: generate multiple carbon price files at end, maybe with settings from switch_params.py
+ Early plant retirement​
+ Foresight with sample weeks​
- save results in MIP_results_comparison

later:
- Unit commitment​
  - update pg_switch.py (Ramp_Up_Percentage, Ramp_Dn_Percentage and maybe Min_Power returned by add_misc_gen_values)
- Best effort​ (all of the above)

eventually: use switch_params.py to define scenarios.txt somehow

eventually: more cases, e.g., longer strips of weeks and maybe tx/carbon cost cases)

run all the models


old notes:

put everything in Switch-USA-PG dir/repo
- PowerGenome subpackage (points to develop branch for now)
- MIP_results_comparison subpackage

Use git commit --author="Rangrang Zheng <zhengr@hawaii.edu>" to set the author to Rangrang for the iniial commits

Create a switch directory.

Move SWITCH_INPUTS_rr_east/self_defined_modules to switch/study_modules

*** Make sure Rangrang can run with only Switch 2.0.7 (not pre-release) and study_modules. (Or else with the next_version branch from the main switch repo.)

Add download_pg_data.py script to download corrected-20z-resource-groups and PowerGenome Data Files into a pg_data folder.

Add a module to study_modules to chain the myopic models (needs to know where
the next stage model will be, e.g., --next-stage-inputs-dir
in/this_scenario_2050), based on `to_myopic.py`. This will then copy
gen_build_predetermined.csv and gen_build_costs.csv to the other dir, keeping
anything already in those files in the other dir from later than the end of this
study (but not earlier, because we need to transfer building plans from this
study and any inherited from prior studies).

Update pg_to_switch.py instructions to use settings from `MIP_Results_comparison/case_settings/26-zone`.

Update pg_to_switch.py to
- put the switch inputs in switch/in/*
- write scenarios_2030.txt, scenarios_2040.txt, scenarios_2050.txt and scenarios_foresight.txt to run the scenarios. These should put outputs in switch/out/*.

Find script that converts model outputs to standard form for comparisons (line 1
to 380 of to_myopic.py), and move it to switch/. Add module that runs that
script automatically to create standard model outputs in `MIP_compare_results`
when the model finishes.

Update settings in `MIP_compare_results` to generate blank production cost
model(s)

Add script to `MIP_compare_results` to setup operational simulation stage:

  - add alternative gen_build_predetermined.csv and gen_build_costs.csv files
    for each model output summary
  - create a scenarios.txt file to run for each of these





old notes:

hard to see a clean way through this: there's a lot of study-specific code and data
in the pg_to_switch pipeline, which can't be easily split into generic US stuff
plus study-specific choices
  - e.g., maybe we should have a settings file for the Switch version of
    PowerGenome, where we specify the gen_info columns we want (?) and then
    the Switch plugin adjusts the settings as needed to get the right upstream
    generator columns from PowerGenome, then uses that to produce the required
    output columns; maybe our plugin could have a big list of final
    tables.columns, source columns.tables they depend on and conversion
    function to produce them. Then we call powergenome to get the required
    tables and columns, then call the converter functions to translate them.
  - also, what to do with data that doesn't exist from a public source and is
    currently specified in settings.yml or hard-coded? e.g., retirement ages
    for equipment
  - note: it looks like settings.yml should mainly be static for Switch
    models, though it may vary depending on the features requested, so e.g.,
    requesting advanced spinning reserves in Switch may require changing some
    entries in settings.yml, but the user shouldn't know what those are and
    we do. On the other hand, we should let the user specify zones, resolution,
    etc. So we should have a switch_settings.yml (including module selection?)
    that we convert into settings for powergenome.


repos:
  - powergenome (just gets installed)
  - PG_TO_SWTICH_refactor (overlaps MIP, but we want this to evolve into freestanding repo)
  - MIP (PG_TO_SWTICH_refactor)



collect MIP powergenome data files together into one folder (maybe in the
PowerGenome/data folder in the PowerGenome dev branch that we are using,
similar to Rangrang's email on 11/8/23?)

collect public powergenome data into a different location that can be bundled
or configured automatically for public use

update conversion_functions.py to create Switch 2.0.7 outputs instead of 2.0.6

