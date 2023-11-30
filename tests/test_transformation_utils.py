import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import ast
from src.handler_utils import PrepareTimeSeries, RunTransformations
from src.initialize import initialize_client

@pytest.fixture
def prep_ts():
    ts_input_names = ["VAL_17-FI-9101-286:VALUE"]
    ts_output_names = ["test_CF"]

    end_date = datetime(2023, 10, 20, 23, 0) #TODO: MAKE SURE THIS DATE HAS SOME VALUES!
    data_dict = {"cdf_env":"dev",
                 "start_time": pd.to_datetime(end_date.date()),
                 "end_time": end_date,
                 "function_name": "cf_calc-test",
                 "calculation_function": "calc_calc_test",
                 "granularity": 60,
                 "backfill_days": 1}

    client = initialize_client(data_dict["cdf_env"], cache_token=False)

    return PrepareTimeSeries(ts_input_names, ts_output_names, client, data_dict)

@pytest.fixture
def run_transform(prep_ts):
    data = prep_ts.data
    ts_input_names = data["ts_input"]
    ts_output_names = data["ts_output"]

    for ts_in, ts_out in zip(ts_input_names.keys(), ts_output_names.keys()):
        df_orig_today = prep_ts.retrieve_orig_ts(ts_in, ts_out)
        data["ts_input_today"][ts_in] = df_orig_today[ts_in]

    ts_df = prep_ts.get_ts_df()
    ts_input_df = prep_ts.align_time_series(ts_df)

    return RunTransformations(data, ts_input_df)

def test_call(prep_ts, run_transform):
    """Test that running transformation produces a correctly aggegated
    time series of correct type.

    Args:
        prep_ts (PrepareTimeSeries): instance of prepared Time Series objects
        run_transform (_type_): _description_
    """
    calculation = prep_ts.data["calculation_function"]
    ts_out = run_transform(eval(calculation))

    assert isinstance(ts_out, list)
    for ts in ts_out:
        assert isinstance(ts, pd.Series)
    # Assuming one transformation per data point
    ts_in = run_transform.ts_df.squeeze()
    assert ts_out.values == ts_in.values - 10 # this is our intended test transformation

def test_store_output_ts(prep_ts, run_transform):
    """Test that output time series are correctly stored in a single, consolidated dataframe

    Args:
        prep_ts (_type_): _description_
        run_transform (_type_): _description_
    """
    calculation = prep_ts.data["calculation_function"]
    ts_out = run_transform(eval(calculation))

    out_df = run_transform.store_output_df(ts_out)
    # Assert correct data type
    assert isinstance(out_df, pd.DataFrame)
    # Assert stored data are same as that from transformation
    for col, out_data_true in zip(out_df.columns, ts_out):
        out_data = out_df[col].values
        out_data_true = out_data_true.values
        assert out_data == out_data_true

def test_transform_multiple_ts(prepare_ts, run_transform):
    import time
    prepare_ts.ts_inputs_names = ["VAL_17-FI-9101-286:VALUE", "VAL_17-PI-95709-258:VALUE", "VAL_11-PT-92363B:X.Value", "VAL_11-XT-95067B:Z.X.Value"]
    prepare_ts.ts_output_names = ["TEST_MultipleOutputs_1", "TEST_MultipleOutputs_2", "TEST_MultipleOutputs_3", "TEST_MultipleOutputs_4"]

    prepare_ts.update_ts("ts_input")
    prepare_ts.update_ts("ts_output")

    prepare_ts.data["calculation_function"] = "calc_multiple_outputs"
    calculation = prepare_ts.data["calculation_function"]

    now = pd.Timestamp.now()
    while now.minute >= 59:
        time.sleep(1)
        now = pd.Timestamp.now()
    prepare_ts.data["backfill_hour"] = now.hour
    prepare_ts.data["backfill_min_start"] = now.minute
    prepare_ts.data["backfill_min_end"] = now.minute + 1

    prepare_ts = prepare_ts.get_orig_timeseries(eval(calculation))

    for ts_name in prepare_ts.ts_output_names:
        ts_out = prepare_ts.client.time_series.data.retrieve(external_id=ts_name)
        assert not ts_out.empty

    ts_out = run_transform(eval(calculation))
    assert isinstance(ts_out, list)
    for ts in ts_out:
        assert isinstance(ts, pd.Series)

    out_df = run_transform.store_output_df(ts_out)
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns == prepare_ts.ts_output_names