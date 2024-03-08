"""
Retrieve and decompress input files to run PowerGenome and set PowerGenome
parameters to access them.
"""

import os, sys, re, zipfile
import gdown, yaml, powergenome


def first_subdir(filename):
    parts = filename.split("/")
    return parts[0] if len(parts) > 1 else ""


def zip_is_one_object(paths):
    if not paths:
        # empty zip file, treat as empty folder
        return False
    subdir = first_subdir(paths[0])
    all_same_dir = all(first_subdir(f) == subdir for f in paths)
    return all_same_dir


def unzip(filename):
    print(f"unzipping {filename}")
    with zipfile.ZipFile(filename, "r") as zip_ref:
        # identify the files we want to extract (ignore metadata)
        names = [n for n in zip_ref.namelist() if not n.startswith("__MACOSX")]
        if zip_is_one_object(names):
            # expand in place
            dest = os.path.dirname(filename)
        else:
            # use zip file name as outer subdir
            dest = os.path.splitext(filename)[0]
        zip_ref.extractall(dest, members=names)
    os.remove(filename)


def main(filter=[]):
    with open("pg_data.yml", "r") as f:
        settings = yaml.safe_load(f)

    for dest, url in settings["download_folders"].items():
        if not filter or any(f in dest for f in filter):
            print(f"\nretrieving {dest}")
            files = gdown.download_folder(url, output=dest)
            for filename in files:
                if filename.endswith(".zip"):
                    unzip(filename)

    for dest, url in settings["download_files"].items():
        if not filter or any(f in dest for f in filter):
            print(f"\nretrieving {dest}")
            filename = gdown.download(url, fuzzy=True, output=dest)
            if filename.endswith(".zip"):
                # unzip the file and delete the .zip
                unzip(filename)

    # create powergenome/.env
    env_file = os.path.join(powergenome.__path__[0], ".env")
    rel_path = os.path.relpath(env_file, os.getcwd())
    if not rel_path.startswith(".."):
        # change to relative path if powergenome is in a subdir, for neater reporting
        env_file = rel_path
    print(f"\ncreating {env_file}")
    with open(env_file, "w") as f:
        for var, dest in settings["env"].items():
            abs_dest = os.path.abspath(dest)
            f.write(f"{var}='{abs_dest}'\n")

    # create case_settings/**/env.yml
    for model, dest in settings["resource_groups"].items():
        abs_dest = os.path.abspath(dest)
        yml_file = os.path.join(
            "MIP_results_comparison", "case_settings", model, "settings", "env.yml"
        )
        print(f"\ncreating {yml_file}")
        with open(yml_file, "w") as f:
            f.write(f"RESOURCE_GROUPS: '{abs_dest}'\n")

    print(f"\n{sys.argv[0]} finished.")


if __name__ == "__main__":
    main(filter=sys.argv[1:])
