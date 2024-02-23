#!/bin/bash

python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in --case-id base_short --case-id base_short_no_ccs --case-id base_short_current_policies --case-id base_short_tx_15 --case-id base_short_tx_50 --case-id base_short_tx_100 --case-id base_short_tx_200 --myopic
python pg_to_switch.py MIP_results_comparison/case_settings/26-zone/settings switch/26-zone/in --case-id base_short
