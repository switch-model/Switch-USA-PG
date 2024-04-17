To run scenarios with caps on pollution exposure:

In the root Switch-USA-PG directory, run a command like this. (You may need to
setup your case settings to define 2030, 2040, 2050 foresight models.)

```
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/exposure/in --case-id base --case-id current_policies
```

Then in the switch/exposure directory, run this command:

```
python setup_exposure_scenarios.py
```

Then run these two commands. Wait for the first to finish before running the second.

```
switch solve-scenarios --scenario-list scenarios_1.txt
# wait for first to finish, then
switch solve-scenarios --scenario-list scenarios_2.txt
```
