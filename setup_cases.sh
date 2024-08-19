#!/bin/bash

# setup all myopic cases (don't specify case-id)
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings-atb2023 switch/26-zone/in/ --myopic
# setup foresight cases (only for two main cases)
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings-atb2023 switch/26-zone/in/ --case-id base_20_week --case-id current_policies_20_week
