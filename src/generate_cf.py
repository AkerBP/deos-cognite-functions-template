import os
import re
import shutil
import subprocess

def generate_cf(cf_name: str, add_packages: list = []):
    cwd = os.getcwd().replace("\\", "/")
    cf_path = cwd+"/cf_"+cf_name

    # Make mandatory files
    if not os.path.exists(cf_path):
        os.mkdir(cf_path)
        with open(cf_path+"/__init__.py", "w") as file:
            print(f"Writing __init__.py in ...")
            file.close()
        with open(cf_path+"/handler.py", "w") as file:
            print(f"Writing handler.py in ...")
            file.write(write_handle())
            file.close()
        with open(cf_path+"/transformation.py", "w") as file:
            print(f"Writing transformation.py ...")
            file.write(write_transformation())
            file.close()

    # Copy template dependency requirements to cf subfolder
    get_toml_dependencies(path_to_cf=cf_path, path_to_toml="../", include_version=True)

    os.chdir(cf_path)
    add_packages = add_packages + [str(line.strip()) for line in open("requirements.txt", "r") if len(line.strip()) > 0]
    add_packages = list(set(add_packages))
    print("Packages to add: ", add_packages)
    src_dir = os.path.dirname(os.getcwd())
    # Initialize Poetry environment in cf subfolder
    run_poetry_command("poetry init")
    for package in add_packages:
        run_poetry_command(f"poetry add {package}")
    run_poetry_command("poetry install")

    # Remove versioning for requirements.txt (NB: may need to manually look into requirements.txt afterwards to ensure internal consistency!)
    get_toml_dependencies(path_to_cf=cf_path, include_version=True)

    os.chdir(src_dir)



def get_toml_dependencies(path_to_cf: str, path_to_toml: str = "", include_version: bool = True):
    if len(path_to_toml) > 0 and path_to_toml[-1] != "/":
        path_to_toml = path_to_toml+"/"
    toml = f"{path_to_toml}pyproject.toml"

    with open(toml, "r") as file:
        content = file.read()

        pattern = r"\[tool.poetry.dependencies\](.*?)\["
        match = re.findall(pattern, content, re.DOTALL)[0]
        match = match.replace('"', '')
        if include_version:
            match = re.sub(r"= [0-9\.]+", "", match)
            match = re.sub(r"= \^[0-9\.]+", "", match)
        match = re.sub(r"python(?![^ ])", "", match)

        with open(f"{path_to_cf}/requirements.txt", "w") as out_file:
            print(f"Created requirements.txt in {path_to_cf}")
            out_file.write(match)
            out_file.close()

        file.close()

def move_file_to_subfolders(file_path, subfolder_prefix):
    src_directory = os.getcwd()
    subfolders = [folder for folder in os.listdir(src_directory) if os.path.isdir(folder) and folder.startswith(subfolder_prefix)]

    for subfolder in subfolders:
        destination_path = os.path.join(src_directory, subfolder, os.path.basename(file_path))
        shutil.copy(file_path, destination_path)
        print(f"Moved {file_path} to {destination_path}")
    print(f"Removed {file_path} from {os.getcwd()}")
    os.remove(file_path)

def write_handle():
    return '''
import os
import sys

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from cognite.client._cognite_client import CogniteClient
from handler_utils import PrepareTimeSeries #get_orig_timeseries
from transformation_utils import RunTransformations
from transformation import *

def handle(client: CogniteClient, data: dict) -> str:
    """Main entry point for Cognite Functions fetching input time series,
    transforming the signals, and storing the output in new time series.

    Args:
        client (CogniteClient): client used to authenticate cognite session
        data (dict): data input to the handle

    Returns:
        str: jsonified data from input signals spanning backfilling period
    """
    calculation = data["calculation_function"]
    # STEP 1: Load (and backfill) and organize input time series'
    PrepTS = PrepareTimeSeries(data["ts_input_names"], data["ts_output_names"], client, data)
    data = PrepTS.get_orig_timeseries(eval(calculation))
    ts_df = PrepTS.get_ts_df()
    ts_df = PrepTS.align_time_series(ts_df) # align input time series to cover same time period

    # STEP 2: Run transformations
    transform_timeseries = RunTransformations(data, ts_df)
    ts_out = transform_timeseries(eval(calculation))

    # STEP 3: Structure and insert transformed signal for new time range (done simultaneously for multiple time series outputs)
    df_out = transform_timeseries.store_output_ts(ts_out)
    client.time_series.data.insert_dataframe(df_out)

    # Store original signal (for backfilling)
    return data["ts_input_backfill"]
'''

def write_transformation():
    return '''
def main_calculation(data, *ts_inputs):
    """Main function for transforming a set of input timeseries from
    whatever calculations you would like to perform.

    Args:
        data (dict): calculation-specfic parameters for Cognite Function
        ts_inputs (list | pd.DataFrame): list of inputs time series to transform, each one given as a pd.DataFrame

    Returns:
        (list): transformed time series given as pd.Series
    """
    # The following is just a dummy calculation
    ts_out = []
    for ts in ts_inputs:
        ts_df = ts.rolling(window=int(len(ts)/10)).mean()
        ts_out.append(ts_df.squeeze()) # Necessary to convert from pd.DataFrame to pd.Series!
    return ts_out
'''

def run_poetry_command(command: str):
    try:
        # Use subprocess.run to execute the command
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        # If there is an error, print the error message
        print("Error running Poetry command:")
        print(e.stderr)
