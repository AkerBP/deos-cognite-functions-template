import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ast
import os
import sys
from dotenv import load_dotenv

from cognite.client.testing import monkeypatch_cognite_client
from cognite.client.data_classes import TimeSeries
from cognite.client.data_classes.functions import Function, FunctionCallList, FunctionCall
from cognite.client.data_classes.functions import FunctionSchedulesList
from cognite.client.data_classes.functions import FunctionSchedule

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# parent_path = parent_path + "\\src"
# print(parent_path + "\\src")
if parent_path not in sys.path:
    sys.path.append(parent_path)

from transformation_utils import RunTransformations
from handler_utils import PrepareTimeSeries
from initialize import initialize_client
from deploy_cognite_functions import deploy_cognite_functions
from generate_cf import generate_cf

CREATED_TIMESERIES = [
    TimeSeries(
        external_id="test_CF_create_timeseries",
        name="test_CF_create_timeseries",
        data_set_id=1832663593546318,
        asset_id=6489717137640502
    )
]

RETRIEVED_FUNCTION = [
    Function(
        external_id="cf_test"
    )
]

CREATED_SCHEDULE = [
    FunctionSchedule(
        name="cf_test"
    )
]

LIST_SCHEDULES = [
    FunctionSchedulesList(
        [
            FunctionSchedule(
                name="cf_test",
            )
        ]
    )
]

LIST_SCHEDULED_CALLS = [
    FunctionCallList(
        [
            FunctionCall(
                id=123,
                function_id=4422
            ),
            FunctionCall(
                id=864,
                function_id=4422
            )
        ]
    )
]

@pytest.fixture
def cognite_client_mock():
    with monkeypatch_cognite_client() as client:
        client.time_series.create.return_value = CREATED_TIMESERIES
        # client.functions.retrieve.return_value = RETRIEVED_FUNCTION
        # client.functions.schedules.create.return_value = CREATED_SCHEDULE
        # client.functions.schedules.list.return_value = LIST_SCHEDULES
        # client.functions.retrieve.list_calls.return_value = LIST_SCHEDULED_CALLS
        return client



@pytest.fixture
def prepare_ts(cognite_client_mock):
    ts_input_names = ["VAL_17-FI-9101-286:VALUE"]
    ts_output_names = ["test_CF"]

    load_dotenv("../../handler-data.env")

    data_dict = {"cdf_env":"dev",
                 'testing': True,
                 "ts_input_names": ts_input_names,
                 "ts_output_names": ts_output_names,
                 'granularity': 10,
                 'aggregate': {},
                 "backfill_days": 1/(24), # backfill 1 hour back in time
                 "dataset_id": str(os.getenv("DATASET_ID")),
                 "calc_params": {}}

    client = initialize_client(data_dict["cdf_env"], path_to_env="../../authentication-ids.env")
    # data_dict["client"] = client

    prepTS = PrepareTimeSeries(ts_input_names, ts_output_names, client, data_dict)
    return prepTS


@pytest.fixture
def calculation(data, ts):
    return (ts - 10).squeeze()


def test_create_timeseries(cognite_client_mock, prepare_ts):
    """Test that new Time Series object is created upon call.

    Args:
        cognite_client_mock(cognite.client.testing.monkeypatch_cognite_client): mock used for CogniteClient
        prepare_ts (_type_): _description_
    """
    prepare_ts.ts_output_names = ["test_CF_create_timeseries"]
    prepare_ts.update_ts("ts_output")
    prepare_ts.data["ts_output"][prepare_ts.ts_output_names[0]]["exists"] = False

    #true_asset_id = [6489717137640502]#[7210022650246556]
    returned_asset_id = prepare_ts.create_timeseries()
    # assert returned_asset_id == true_asset_id

    # Assert correct cognite object has been called once
    assert cognite_client_mock.time_series.create.call_count == 1
    # Assert cognite object is called with correct arguments
    returned_ts = [ts.dump() for ts in cognite_client_mock.time_series.create.call_args[0]]
    true_ts = [ts.dump() for ts in CREATED_TIMESERIES][0]
    for ts in returned_ts:
        assert ts["externalId"] == true_ts["externalId"]
        assert str(ts["name"]) == str(true_ts["name"])
        assert int(ts["dataSetId"]) == int(true_ts["dataSetId"])
    ### ----- NB: -----
    ### cognite_client_mock doesn't explicitly state asset_id - can't compare this
    ### ---------------

    ### Assert that new time series object is populated with data
    prepare_ts.data["scheduled_calls"] = pd.DataFrame()
    ts_output_list = prepare_ts.client.time_series.list(name=prepare_ts.ts_output_names[0]).to_pandas()
    # assert ts_output_list == LIST_NONEMPTY_TIMESERIES
    # assert ts_output_list["external_id"] == prepare_ts.ts_output_names[0]

    ### Assert correctly assigned asset
    # assert ts_output_list["asset_id"][0] == true_asset_id[0]

    prepare_ts.data["ts_output"][prepare_ts.ts_output_names[0]]["exists"] = True
    prepare_ts.create_timeseries()
    ### Assert Time Series create object has NOT been called twice is time series already exists
    assert cognite_client_mock.time_series.create.call_count == 1
    ### Assert no duplicated time series object if already exists
    # ts_output_list_new = prepare_ts.client.time_series.list(name="random").to_pandas()
    # assert ts_output_list_new == ts_output_list

def test_retrieve_orig_ts(cognite_client_mock, prepare_ts):
    """Test that correct time series signal is retrieved.

    Args:
        prepare_ts (_type_): _description_
    """
    prepare_ts.ts_output_names = ["test_CF"]
    prepare_ts.update_ts("ts_output")
    ts_in = prepare_ts.ts_inputs[0]
    ts_out = prepare_ts.ts_output_names[0]

    # Test that full historic signal is retrieved if output not exists
    prepare_ts.data["ts_output"][prepare_ts.ts_output_names[0]]["exists"] = False
    data = prepare_ts.retrieve_orig_ts(ts_in, ts_out)

    assert cognite_client_mock.time_series.data.retrieve.call_count == 1
    # client_returned = cognite_client_mock.time_series.data.retrieve.call_args[0].dump()
    # assert client_returned["external_id"] == RETRIEVED_TIMESERIES["external_id"]
    # assert len(data) == len(client_returned["datapoints"])

    first_date = data.index[0]
    last_date = data.index[-1]

    data_true = prepare_ts.client.time_series.data.retrieve(external_id=ts_in,
                                                       aggregates="average",
                                                       granularity=f"{prepare_ts.data['granularity']}s",
                                                       limit=-1).to_pandas()
    assert cognite_client_mock.time_series.data.retrieve.call_count == 2

    first_date_true = data_true.index[0]
    last_date_true = data_true.index[-1]
    assert first_date == first_date_true
    assert last_date == last_date_true

    # Test that data only retrieved from current date if output exists
    prepare_ts.data["ts_output"][prepare_ts.ts_output_names[0]]["exists"] = True
    data = prepare_ts.retrieve_orig_ts(ts_in, ts_out)

    assert len(data) == len(cognite_client_mock.time_series.data.retrieve.call_args[2].dump()["datapoints"])

    first_date = data.index[0]
    last_date = data.index[-1]
    data_true = prepare_ts.client.time_series.data.retrieve(external_id=ts_in,
                                                       aggregates="average",
                                                       granularity=f"{prepare_ts.data['granularity']}s",
                                                       start=prepare_ts.data["start_time"],
                                                       end=prepare_ts.data["end_time"]).to_pandas()

    first_date_true = data_true.index[0]
    last_date_true = data_true.index[-1]
    assert first_date.date == first_date_true.date
    assert last_date == last_date_true

    assert data.squeeze().values == data_true.squeeze().values

def test_get_ts_df(prepare_ts):
    ts_data = prepare_ts.get_ts_df()
    # Assert that time series data are either DataFrame or Float objects
    assert all([isinstance(data, float) or isinstance(data, pd.DataFrame) for data in ts_data])

def test_align_time_series(cognite_client_mock, prepare_ts):
    prepare_ts.ts_inputs = ["VAL_17-FI-9101-286:VALUE", "VAL_17-PI-95709-258:VALUE", 56.2] # also testing for scalar
    prepare_ts.ts_output_names = ["test_CF", "test_CF2"]
    prepare_ts.update_ts("ts_inputs")
    prepare_ts.update_ts("ts_output")

    prepare_ts.create_timeseries()

    now = pd.Timestamp.now(tz="CET").floor("1s").tz_convert("UTC")
    # Ensure we dont perform any backfilling at this point in time
    prepare_ts.data["backfill_hour"] = (now - timedelta(hours=1)).hour

    prepare_ts.data = prepare_ts.get_orig_timeseries(eval(lambda x: x)) # dummy calculation (not applied here anyway)
    ts_df = prepare_ts.get_ts_df()
    ts_df_true = prepare_ts.align_time_series(ts_df)

    # Assert all datees truncated to same boundary dates
    assert all(np.equal([df.index[0] for df in ts_df_true]))
    assert all(np.equal([df.index[-1] for df in ts_df_true]))

    # Assert same number of data points (despite some internal data may be NaN)
    assert all(np.equal([len(df) for df in ts_df_true]))

    # Assert latest possible start date
    latest_start_date_true = np.max([df.index[0] for df in ts_df])
    latest_start_date_all = [df.index[0] for df in ts_df_true]
    assert all(date == latest_start_date_true for date in latest_start_date_all)
    # Assert earliest possible end date
    earliest_end_date_true = np.min([df.index[-1] for df in ts_df])
    earliest_end_date_all = [df.index[-1] for df in ts_df_true]
    assert all(date == earliest_end_date_true for date in earliest_end_date_all)

    # Assert all return elements are DataFrames
    assert all([isinstance(df, pd.DataFrame) for df in ts_df_true])

def test_get_aggregated_start_time(prepare_ts):
    prepare_ts.data["start_time"] = datetime(2023,11,20,8,20,30)
    prepare_ts.data["end_time"] = pd.Timestamp.now()
    backfill_time = datetime(2023,10,7,13,12,5)

    prepare_ts.data["aggregate"] = {}

    prepare_ts.data["aggregate"]["period"] = "minute"
    # First: test ordinary aggregate (no backfilling)
    start_agg, _ = prepare_ts.get_aggregated_start_end_time()
    assert start_agg == datetime(2023,11,20,8,20,0)
    # Then: test aggregate for backfilling
    start_agg, end_agg = prepare_ts.get_aggregated_start_end_time(backfill_time)
    assert start_agg == datetime(2023,10,7,13,12,0)
    assert end_agg == datetime(2023,10,7,13,13,0)

    prepare_ts.data["aggregate"]["period"] = "hour"
    start_agg, _ = prepare_ts.get_aggregated_start_end_time()
    assert start_agg == datetime(2023,11,20,8,0,0)
    start_agg, end_agg = prepare_ts.get_aggregated_start_end_time(backfill_time)
    assert start_agg == datetime(2023,10,7,13,0,0)
    assert end_agg == datetime(2023,10,7,14,0,0)

    prepare_ts.data["aggregate"]["period"] = "day"
    daily_aggregate = prepare_ts.get_aggregated_start_end_time()
    assert daily_aggregate == datetime(2023,11,20,0,0,0)
    start_agg, end_agg = prepare_ts.get_aggregated_start_end_time(backfill_time)
    assert start_agg == datetime(2023,10,7,0,0,0)
    assert end_agg == datetime(2023,10,8,0,0,0)

    prepare_ts.data["aggregate"]["period"] = "month"
    assert prepare_ts.get_aggregated_start_end_time() == datetime(2023,11,0,0,0,0)
    start_agg, end_agg = prepare_ts.get_aggregated_start_end_time(backfill_time)
    assert start_agg == datetime(2023,10,0,0,0,0)
    assert end_agg == datetime(2023,11,0,0,0,0)

    prepare_ts.data["aggregate"]["period"] = "year"
    assert prepare_ts.get_aggregated_start_end_time() == datetime(2023,0,0,0,0,0)
    start_agg, end_agg = prepare_ts.get_aggregated_start_end_time(backfill_time)
    assert start_agg == datetime(2023,0,0,0,0,0)
    assert end_agg == datetime(2024,0,0,0,0,0)

    # Finally, assert that the modified and original dates are in same timezone
    assert daily_aggregate.tzinfo == prepare_ts.data["start_time"].tzinfo

def test_join_previous_and_current(prepare_ts):
    """Test that joining data from current schedule with data from aggregating
    period NOT overlapping with current schedule returns correct data and
    non-overlapping dates.

    Args:
        prepare_ts (class): instance of PrepareTimeseries
    """
    datetime_previous = pd.date_range(start=datetime(2023,4,5,0),
                                      end=datetime(2023,4,5,17),
                                      freq="1H")
    data_previous = [4,1,8,5,2,8,6,0,-5,-3,6,5,9,6,9,2,4,10]
    df_previous = pd.DataFrame(data_previous, index=datetime_previous, columns=["data"])
    df_previous = df_previous.iloc[:-1] # to avoid duplicate of datapoint at overlapping timestamp between dataframes

    datetime_current = pd.date_range(start=datetime(2023,4,5,17),
                                     end=datetime(2023,4,6,5),
                                     freq="1H")
    data_current = [10,0,6,-4,-1,7,1,5,1,1,12,-1,2]
    df_current = pd.DataFrame(data_current, index=datetime_current, columns=["data"])

    df = pd.concat([df_previous, df_current]) # join scheduled period with remaining aggregated period

    assert len(df) == 30
    assert df["data"] == data_previous + data_current
    assert len(df.index) == len(set(df.index)) # check that each datetime index is unique


def test_backfilling_unchanged(prepare_ts):
    """Test that no changes have been made to output signal during backfilling
    when no changes in input signal.

    Args:
        prepare_ts (_type_): _description_
    """
    ts_input_name = "VAL_17-FI-9101-286:VALUE_COPY"
    prepare_ts.update_ts("ts_input")
    ts_orig_extid = prepare_ts.client.time_series.list(name=ts_input_name).to_pandas().external_id[0]
    prepare_ts["ts_input"][ts_input_name]["orig_extid"] = ts_orig_extid

    ts_output_name = "test_CF"

    backfill_dates = prepare_ts.check_backfilling(ts_input_name, testing=True)

    # Assert no dates for backfilling
    # assert all(num_dates_new.index.isin(num_dates_old.index))
    assert len(backfill_dates) == 0

    df_out_before = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name)
    prepare_ts.run_backfilling()
    df_out_after = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name)
    # Assert that output time series has not been changed
    assert df_out_before == df_out_after

def test_backfilling_insert(prepare_ts):
    """Test that number of datapoints in output signal has increased after backfilling
    when datapoints have been inserted in input signal .


    Args:
        prepare_ts (_type_): _description_
    """
    ts_input_name = "VAL_17-FI-9101-286:VALUE_COPY"
    prepare_ts.update_ts("ts_input")
    ts_orig_extid = prepare_ts.client.time_series.list(name=ts_input_name).to_pandas().external_id[0]
    prepare_ts["ts_input"][ts_input_name]["orig_extid"] = ts_orig_extid

    ts_output_name = "test_CF"

    # Insert data into copied input
    now = pd.Timestamp.now(tz="CET").floor("1s").tz_convert("UTC")
    new_data = [
        (datetime(now.year, now.month, now.day, now.hour-2, now.minute, now.second), 111),
        (datetime(now.year, now.month, now.day, now.hour-1, now.minute, now.second), 333)
    ]
    prepare_ts.client.time_series.data.insert(new_data, external_id=ts_input_name)

    backfill_dates = prepare_ts.check_backfilling(ts_input_name, testing=True)

    # Assert we get backfilling period(s)
    assert len(backfill_dates) > 0

    df_out_before = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name).to_pandas()
    prepare_ts.run_backfilling()
    df_out_after = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name).to_pandas()

    # Assert that new output version has more data than previous version
    assert len(df_out_after) > len(df_out_before)

    mask_timestamp1_before = df_out_before.index == new_data[0][0]
    mask_timestamp2_before = df_out_before.index == new_data[1][0]
    mask_timestamp1_after = df_out_after.index == new_data[0][0]
    mask_timestamp2_after = df_out_after.index == new_data[1][0]

    # Assert that old values are unchanged
    assert len(df_out_before[mask_timestamp1_before]) == 0
    assert len(df_out_before[mask_timestamp2_before]) == 0

    # Assert that new values have been inserted at given dates
    assert len(df_out_after[mask_timestamp1_after]) > 0
    assert len(df_out_after[mask_timestamp2_after]) > 0

    # Assert that datapoints outside these dates are unchanged
    assert len(df_out_after[~mask_timestamp1_after & ~mask_timestamp2_after]) \
            == len(df_out_before[~mask_timestamp1_before & ~mask_timestamp2_before])


def test_backfilling_delete(prepare_ts):
    """Test that number of datapoints in output signal has decreased after backfilling
    when datapoints have been deleted in input signal.


    Args:
        prepare_ts (_type_): _description_
    """
    ts_input_name = "VAL_17-FI-9101-286:VALUE_COPY"
    prepare_ts.update_ts("ts_input")
    ts_orig_extid = prepare_ts.client.time_series.list(name=ts_input_name).to_pandas().external_id[0]
    prepare_ts["ts_input"][ts_input_name]["orig_extid"] = ts_orig_extid

    ts_output_name = "test_CF"

    # Delete data from copied input
    now = pd.Timestamp.now(tz="CET").floor("1s").tz_convert("UTC")
    start_delete = datetime(now.year, now.month, now.day, now.hour-2, now.minute)
    end_delete = datetime(now.year, now.month, now.day, now.hour, now.minute)

    prepare_ts.client.time_series.data.delete_range(start=start_delete, end=end_delete, external_id=ts_input_name) # inclusive start date but exclusive end date

    df_out_before = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name).to_pandas()
    prepare_ts.run_backfilling()
    df_out_after = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name).to_pandas()

    start_before = df_out_before.index >= start_delete
    end_before = df_out_before.index < end_delete
    start_after = df_out_after.index >= start_delete # inclusive start date
    end_after = df_out_after.index < end_delete # exclusive end date

    # Assert that num datapoints inside deleted range has decreased
    assert len(df_out_after[start_after & end_after]) < len(df_out_before[start_before & end_before])
    # Assert that num datapoints outsie deleted range remains unchanged
    assert len(df_out_after[~(start_after & end_after)]) == len(df_out_before[~(start_before & end_before)])











