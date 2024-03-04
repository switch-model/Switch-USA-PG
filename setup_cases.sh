#!/bin/bash

python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in --case-id base_short --case-id base_short_no_ccs --case-id base_short_current_policies --myopic
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in --case-id base_short
