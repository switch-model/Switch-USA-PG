from pg_to_switch import main

# main(settings_file="settings_TD_east.yml", results_folder="test4")
main(
    settings_file="case_settings/26-zone/settings",
    results_folder="test4",
    case_id=["base"],
    year=[2050],
)
