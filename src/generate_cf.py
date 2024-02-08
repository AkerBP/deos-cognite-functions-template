import os
import re
import shutil
import subprocess

def generate_cf(data: dict, add_packages: list = []):
    cwd = os.getcwd().replace("\\", "/")
    cf_name = data["function_name"]
    cf_path = cwd+"/"+cf_name

    # Make mandatory files
    if not os.path.exists(cf_path):
        os.mkdir(cf_path)
        with open(cf_path+"/__init__.py", "w") as file:
            print(f"Writing __init__.py ...")
            file.close()
        with open(cf_path+"/handler.py", "w") as file:
            print(f"Writing handler.py ...")
            file.write(write_handle())
            file.close()
        with open(cf_path+"/transformation.py", "w") as file:
            print(f"Writing transformation.py ...")
            file.write(write_transformation())
            file.close()

    if cf_name == "test":
        get_toml_dependencies(path_to_cf=cf_path, path_to_toml="../../", include_version=True)
        return

    get_toml_dependencies(path_to_cf=cf_path, path_to_toml="../", include_version=True)
    # Copy template dependency requirements to cf subfolder

    os.chdir(cf_path)
    add_packages = add_packages + [str(line.strip()) for line in open("requirements.txt", "r") if len(line.strip()) > 0]
    add_packages = list(set(add_packages))
    print("Packages to add: ", add_packages)
    src_dir = os.path.dirname(os.getcwd())
    # Initialize Poetry environment in cf subfolder
    run_poetry_command("poetry init")

    for package in add_packages:
        if package == "indsl":
            run_poetry_command(f"poetry add {package} --python ^3.11")
        else:
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

def write_handle():
    return '''
import os
import sys
import pandas as pd
from datetime import datetime

# Set file to system path to allow relative import from parent folder
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from cognite.client._cognite_client import CogniteClient
from handler_utils import PrepareTimeSeries
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
    calculation = "main_transformation"
    #calculation = data["calculation_function"]
    # STEP 1: Load (and backfill) and organize input time series'
    PrepTS = PrepareTimeSeries(data["ts_input_names"], data["ts_output"], client, data)
    PrepTS.data = PrepTS.get_orig_timeseries(eval(calculation))

    ts_in = PrepTS.data["ts_input_data"]
    ts_out = PrepTS.data["ts_output"]
    all_inputs_empty = any([ts_in[name].empty if isinstance(ts_in[name], (pd.Series, pd.DataFrame)) else False for name in ts_in])

    if not all_inputs_empty: # can't run calculations if any time series is empty for defined interval
        df_in = PrepTS.get_ts_df()
        df_in = PrepTS.align_time_series(df_in) # align input time series to cover same time period

        # STEP 2: Run transformations
        transform_timeseries = RunTransformations(PrepTS.data, df_in)
        df_out = transform_timeseries(eval(calculation))

        # STEP 3: Ensure output is correctly formatted dataframe as required by template
        assert_df(df_out, ts_out)

        # STEP 4: Insert transformed signal for new time range (done simultaneously for multiple time series outputs)
        client.time_series.data.insert_dataframe(df_out)

    # Store original signal (for backfilling)
    return df_out.to_json()

def assert_df(df_out, ts_out):
    """Check requirements that needs to be satisfied for
    the output dataframe from the calculation.

    Args:
        df_out (pd.DataFrame): output dataframe of calculation
        ts_out (list): names of output time series
    """
    assert isinstance(df_out, pd.DataFrame), f"Output of calculation must be a Dataframe"
    assert type(df_out.index) == pd.DatetimeIndex, f"Dataframe index must be of type DatetimeIndex, not {type(df_out.index)}."
    assert (list(df_out.columns) == list(ts_out.keys())), f"Dataframe columns for calculated time series, {list(df_out.columns)}, not equal to output names, {list(ts_out.keys())}, specified in data dictionary"
'''

def write_transformation():
    return '''
import os
import sys
# Set file to system path to allow relative import from parent folder
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from utilities import AGG_PERIOD
import pandas as pd

def main_aggregate(data, ts_inputs):
    """Sample main function for performing aggregations (here: average drainage).

    Args:
        data (dict): parameters for Cognite Function
        ts_inputs (pd.DataFrame): input time series to transform, one column per time series

    Returns:
        (pd.DataFrame): transformed time series, one column per time series
    """
    import numpy as np

    try:
        aggregate = data["optional"]["aggregate"]
    except:
        raise KeyError(f"Aggregated calculations require 'aggregate' to be specified as an optional parameter to input data dictionary. Only found the optional parameters: {data['optional'].keys()}")

    out_index = ts_inputs.index.floor(AGG_PERIOD[aggregate["period"]]).unique()

    calc_params = data["calc_params"]
    ts_out = filter_ts(ts_inputs, calc_params)

    derivative_value_excl = calc_params['derivative_value_excl']

    input_cols = list([col for col in ts_inputs.columns if col != "time_sec"])
    output_cols = list([col for col in data["ts_output"].keys() if col != "time_sec"])

    df_out = pd.DataFrame(index=out_index, columns=output_cols)

    for in_col, out_col in zip(input_cols, output_cols):

        try:
            ts_out[in_col+"_derivative"] = np.gradient(ts_out[in_col], ts_out["time_sec"])
        except:
            raise IndexError(
                f"No datapoints found for selected date range for time series {in_col}. Cannot compute drainage rate.")

        ts_out[in_col+"_drainage"] = ts_out[in_col+"_derivative"].apply(
            lambda x: 0 if x > derivative_value_excl or pd.isna(x) else x)  # not interested in large INLET fluxes

        df_out[out_col] = ts_out.resample(AGG_PERIOD[aggregate["period"]])[in_col+"_drainage"].agg(aggregate["type"]).values

    return df_out

def filter_ts(ts_inputs, data):
    """Helper function: performs lowess smoothing

    Args:
        ts (pd.DataFrame): input time series
        data (dict): calculation-specific parameters for Cognite Function

    Returns:
        pd.DataFrame: smoothed signal
    """
    from datetime import datetime, timezone
    from statsmodels.nonparametric.smoothers_lowess import lowess

    df_smooth = pd.DataFrame()
    ts_inputs["time_sec"] = (ts_inputs.index - datetime(1970, 1, 1,
                                                        tzinfo=timezone.utc)).total_seconds()

    if "lowess_frac" in data:
        frac = data["lowess_frac"]
    else:
        frac = 0.01
    if "lowess_delta" in data:
        delta = data["lowess_delta"]
    else:
        delta = 0

    ts_input_names = [name for name in ts_inputs.columns if name != "time_sec"]
    for ts in ts_input_names:
        data = ts_inputs[ts]

        smooth = lowess(data, ts_inputs["time_sec"], is_sorted=True,
                        frac=frac, it=0, delta=delta*len(data))

        df_smooth[ts] = smooth[:,1] # only extract values, not timestamp

    df_smooth["time_sec"] = smooth[:,0]
    df_smooth.index = pd.to_datetime(df_smooth["time_sec"], unit="s")

    return df_smooth

def main_test(data, ts_inputs):
    """Sample main function for transforming a set of input timeseries to
    produce a set of associated output time series.

    Args:
        data (dict): parameters for Cognite Function
        ts_inputs (pd.DataFrame): input time series to transform, one column per time series

    Returns:
        (pd.DataFrame): transformed time series, one column per time series
    """
    ts_out = pd.DataFrame()

    for ts in ts_inputs.columns:
        ts_out[ts] = ts_inputs[ts].dropna().rolling(window=10).mean().reindex(ts_inputs[ts].index, method="nearest")

    return ts_out

def main_B(data, ts_inputs):
    """Other sample main function for transforming a set of input timeseries to
    produce a set of associated output time series.

    Args:
        data (dict): parameters for Cognite Function
        ts_inputs (pd.DataFrame): input time series to transform, one column per time series

    Returns:
        (pd.DataFrame): transformed time series, one column per time series
    """
    ts_0 = ts_inputs.iloc[:,0] # first time series in dataframe
    ts_1 = ts_inputs.iloc[:,1] # second time series ...
    ts_2 = ts_inputs.iloc[:,2]

    ts_out = ts_0.max() + 2*ts_1 + ts_0*ts_2/5

    return pd.DataFrame(ts_out) # ensure output given as dataframe
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
