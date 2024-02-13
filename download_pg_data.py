"""
Retrieve and decompress input files to run PowerGenome.
"""

import os, sys, re, zipfile
import gdown

# All the files that end up in `PowerGenome Data Files` originate from
# https://drive.google.com/drive/folders/1K5GWF5lbe-mKSTUSuJxnFdYGCdyDJ7iE
# which is referenced from https://github.com/PowerGenome/PowerGenome#data
# (links to individual files found via right click > share > copy link;
# don't remove view?usp stuff from the end).
# We download them selectively instead of downloading the whole folder
# because the resource groups folder there is very large and we don't need
# all of it.

# The `corrected-20z-resource-groups` file (which gets unzipped to a folder) is
# from custom Google Drive folder for MIP project:
# https://drive.google.com/drive/u/2/folders/1KBdoonCeDfvAgQ10KpwVhmlyfKi1rY5K

# TODO: move these download lists into a pg_data.yml file

needed_folders = [
    (
        "pg_data/PowerGenome Data Files/PUDL Data",
        "https://drive.google.com/drive/folders/1z9BdvbwgpS5QjPTrcgyFZJUb-eN2vebu",
    ),
    (
        "pg_data/PowerGenome Data Files/PowerGenome Resource Groups/generation_profiles",
        "https://drive.google.com/drive/folders/1ZYxnl4U_3HXlYPxm8qlmqyWB8NyC3PpG",
    ),
    # (
    #     "pg_data/PowerGenome Data Files/PowerGenome Resource Groups/resource_groups/us-26-zone",
    #     "https://drive.google.com/drive/folders/1RmHofRL5-xcuCf3zvgIeqcmueBZVxlrn",
    # ),
]
needed_files = [  # note: .zip files will be replaced by their expanded version
    (
        "pg_data/corrected-20z-resource-groups.zip",
        "https://drive.google.com/file/d/1MXkdRW-YQ-hq3KzK_TbzTfD1-0CRZmdR/view?usp=drive_link",
    ),
    (
        "pg_data/PowerGenome Data Files/pg_misc_tables_efs.sqlite.zip",
        "https://drive.google.com/file/d/1XrLOqVGNP1qjvsXeTt1YH2Pyppqad0fc/view?usp=drive_link",
    ),
    (
        "pg_data/PowerGenome Data Files/efs_files_utc.zip",
        "https://drive.google.com/file/d/1dWA35bQpPksnSb6auybMbrIqyaBG6wBM/view?usp=drive_link",
    ),
    (
        "pg_data/PowerGenome Data Files/cambium_dg_data.zip",
        "https://drive.google.com/file/d/1nbhWwOsNeOtcUew9Mn4QGuAtCsZo0VZ2/view?usp=drive_link",
    ),
]


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
    for dest, url in needed_folders:
        if not filter or any(f in dest for f in filter):
            print(f"\nretrieving {dest}")
            files = gdown.download_folder(url, output=dest)
            for filename in files:
                if filename.endswith(".zip"):
                    unzip(filename)

    for dest, url in needed_files:
        if not filter or any(f in dest for f in filter):
            print(f"\nretrieving {dest}")
            filename = gdown.download(url, fuzzy=True, output=dest)
            if filename.endswith(".zip"):
                # unzip the file and delete the .zip
                unzip(filename)


###############
# code below is obsolete
# helpers from https://stackoverflow.com/a/39225272
def download_file_from_google_drive(id):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    # this is good enough for now, but if we ever have filenames
    # with non-ascii characters, we'll need to use this:
    # https://stackoverflow.com/a/73418983
    # from https://stackoverflow.com/a/51570425
    filename = (
        re.findall(
            "filename\*?=([^;]+)",
            response.headers["Content-Disposition"],
            flags=re.IGNORECASE,
        )[0]
        .strip()
        .strip('"')
    )

    # if os.path.exists(filename):
    #     print("deleting existing f{filename}.")
    #     os.remove(filename)

    print(f"retrieving {filename}.")
    save_response_content(response, filename)

    if response.headers["Content-Type"] == "application/zip":
        # unzip the file and delete the .zip
        print(f"unzipping {filename}")
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(filename)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


########### end helpers


if __name__ == "__main__":
    main(filter=sys.argv[1:])
