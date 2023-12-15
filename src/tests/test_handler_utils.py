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
from deploy_cognite_functions import deploy_cognite_functions, list_scheduled_calls
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
        client.functions.retrieve.return_value = RETRIEVED_FUNCTION
        client.functions.schedules.create.return_value = CREATED_SCHEDULE
        client.functions.schedules.list.return_value = LIST_SCHEDULES
        client.functions.retrieve.list_calls.return_value = LIST_SCHEDULED_CALLS
        return client



@pytest.fixture
def prepare_ts(cognite_client_mock):
    ts_input_names = ["VAL_17-FI-9101-286:VALUE"]
    ts_output_names = ["test_CF"]

    load_dotenv("../../handler-data.env")

    data_dict = {"cdf_env":"dev",
                 "backfill_days": 1/24, # backfill one hours back in time
                 "scheduled_calls": pd.DataFrame(),
                 "dataset_id": str(os.getenv("DATASET_ID"))}

    client = initialize_client(data_dict["cdf_env"], path_to_env="../../authentication-ids.env")
    data_dict["client"] = client

    return PrepareTimeSeries(ts_input_names, ts_output_names, cognite_client_mock, data_dict)

@pytest.fixture
def deploy_schedule(prepare_ts):
    """Fixture for deploying a test CF schedule

    Args:
        prepare_ts (pytest.fixture): PrepareTimeSeries object

    Returns:
        schedule_data (dict): parameters for deployed schedule
    """
    import time
    from datetime import datetime, timedelta

    prepare_ts.client = prepare_ts.data["client"]

    ts_orig_input_name = prepare_ts.ts_input_names[0]

    ts_input_name = "VAL_17-FI-9101-286:VALUE.COPY"
    prepare_ts.ts_output_names = [ts_input_name] # this is NOT output - it is a copy of the input, but ts_output is demanded by create_timeseries() function
    prepare_ts.update_ts("ts_output")
    prepare_ts.create_timeseries()

    ts_output_name = "test_CF"
    prepare_ts.ts_output_names = [ts_output_name]
    prepare_ts.update_ts("ts_output")
    prepare_ts.create_timeseries()

    prepare_ts.data["start_time"] = datetime(2023, 10, 10, 1)
    prepare_ts.data["end_time"] = datetime(2023, 10, 10, 4)

    ts_orig = prepare_ts.client.time_series.list(name=ts_orig_input_name).to_pandas()
    ts_extid = ts_orig.external_id[0]
    # Insert datapoints into copied input time series
    input_data = prepare_ts.client.time_series.data.retrieve(external_id=ts_extid,
                                                             start=pd.to_datetime(prepare_ts.data["start_time"]),
                                                             end=pd.to_datetime(prepare_ts.data["end_time"]),
                                                             limit=-1).to_pandas()

    input_data = input_data.rename(columns={input_data.columns[0]: ts_input_name})
    prepare_ts.client.time_series.data.insert_dataframe(input_data)

    cdf_env = "dev"
    prepare_ts.data["cdf_env"] = cdf_env
    cron_interval = 1
    prepare_ts.data["cron_interval"] = cron_interval
    prepare_ts.data["function_name"] = "cf_test"
    prepare_ts.data["calculation_function"] = "main_test"

    generate_cf("test")
    # First: Make single call to function to perform initial transformation
    deploy_cognite_functions(prepare_ts.data, prepare_ts.client, cron_interval,
                            True, False)

    # Let backfill period be current timestamp - return argument to be compared to
    now = pd.Timestamp.now()
    while now.minute >= 59:
        time.sleep(1)
        now = pd.Timestamp.now()
    prepare_ts.data["backfill_hour"] = now.hour
    prepare_ts.data["backfill_min_start"] = now.minute
    prepare_ts.data["backfill_min_end"] = now.minute + 1
    # Then: Run function on schedule
    deployed_time_start = datetime(now.year, now.month, now.day, now.hour, now.minute)

    deploy_cognite_functions(prepare_ts.data, prepare_ts.client, cron_interval,
                            False, True)

    schedule_id, all_calls = list_scheduled_calls(prepare_ts.data, prepare_ts.client)
    while len(all_calls) < 10: # wait until we have total of 10 calls from schedule
        time.sleep(30)
        schedule_id, all_calls = list_scheduled_calls(prepare_ts.data, prepare_ts.client)

    schedule_data = {"schedule_id":schedule_id, "calls":all_calls, "deploy_start":deployed_time_start}

    get_func = prepare_ts.client.functions.retrieve(external_id=prepare_ts.data["function_name"])

    last_backfill_id = all_calls.id[0]#get_func.retrieve_call(id=schedule_id)
    last_backfill_call = get_func.retrieve_call(id=last_backfill_id)
    output_dict = ast.literal_eval(last_backfill_call.get_response())[
            ts_input_name]

    output_df = pd.DataFrame.from_dict([output_dict]).T
    output_df.index = pd.to_datetime(
        output_df.index.astype(np.int64), unit="ms")
    output_df["Date"] = output_df.index.date  # astype(int)*1e7 for testing
    previous_df = output_df.rename(columns={0: ts_input_name})

    end_date = pd.Timestamp.now()
    start_date = end_date - timedelta(days=prepare_ts.data["backfill_days"])
    schedule_data["start_time"] = start_date
    schedule_data["end_time"] = end_date
    ts_orig_all = prepare_ts.client.time_series.data.retrieve(external_id=ts_input_name,
                                                    aggregates="average",
                                                    granularity=f"{prepare_ts.data['granularity']}s",
                                                    start=start_date,
                                                    end=end_date,
                                                    limit=-1,
                                                    ).to_pandas()

    ts_orig_all = ts_orig_all.rename(
        columns={ts_input_name + "|average": ts_input_name})

    ts_orig_dates = pd.DataFrame(
            {"Date": pd.to_datetime(ts_orig_all.index.date),
            ts_input_name: ts_orig_all[ts_input_name]},
            index=pd.to_datetime(ts_orig_all.index))

    schedule_data["df_ts_previous"] = previous_df # dataframe of previous time series signal
    schedule_data["df_ts_current"] = ts_orig_dates # dataframe of current time series signal

    return schedule_data

@pytest.fixture
def calculation(data, ts):
    return (ts - 10).squeeze()

def test_get_schedules_and_calls(cognite_client_mock, prepare_ts, deploy_schedule):
    # WAIT FOR THIS TEST AFTER SCHEDULE HAS BEEN DEPLOYED WITH deploy_schedule FIXTURE
    schedule_data = deploy_schedule(prepare_ts)
    assert cognite_client_mock.schedules.create.call_count == 1
    created_schedule = [sched.dump() for sched in CREATED_SCHEDULE]
    created_schedule_true = [sched.dump() for sched in cognite_client_mock.schedules.create.call_args[0]]
    assert created_schedule == created_schedule_true
    # 10 calls to schedules expected
    schedule_id, calls = prepare_ts.get_schedules_and_calls()
    assert len(calls) == 10
    retrieved_functions = [func.dump() for func in RETRIEVED_FUNCTION]
    retrieved_functions_true = [func.dump() for func in cognite_client_mock.functions.retrieve.call_args[0]]
    assert retrieved_functions == retrieved_functions_true

    listed_schedules = [sched.dump() for sched in LIST_SCHEDULES]
    listed_schedules_true = [sched.dump() for sched in cognite_client_mock.schedules.list.call_args[0]]
    assert listed_schedules == listed_schedules_true


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

    now = pd.Timestamp.now()
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


def test_deploy_schedule(prepare_ts, deploy_schedule):
    """Test utility function used for testing backfilling

    Args:
        prepare_ts (_type_): _description_
        deploy_schedule (_type_): _description_
    """
    ts_input_name = "VAL_17-FI-9101-286:VALUE"
    df_previous_true = deploy_schedule["df_ts_previous"]
    df_current_true = deploy_schedule["df_ts_current"]
    # current_start_date = deploy_schedule["start_time"]
    # current_end_date = deploy_schedule["end_time"]


    df_current, df_previous, \
        _, _, _ = prepare_ts.check_backfilling(ts_input_name, testing=True)

    assert df_current[ts_input_name] == df_current_true[ts_input_name]
    assert df_previous[ts_input_name] == df_previous_true[ts_input_name]


def test_backfilling_call_id(prepare_ts, deploy_schedule):
    """Test that the retrieved backfill call is the correct one.

    Args:
        prepare_ts (_type_): _description_
        deploy_schedule (_type_): _description_
    """
    # NEED TO DEPLOY SCHEDULE AT BACKFILL PERIOD TO ENSURE BACKFILLING IS PERFORMED!
    # BUT: IMPOSSIBLE TO NOW CALL ID A PRIORI !!?
    from datetime import timedelta
    # Assert that correct historical scheduled call is extracted at time when doing backfilling
    deployed_time_start = deploy_schedule["deploy_start"]
    all_calls = deploy_schedule["calls"]
    deployed_time_end = deployed_time_start + timedelta(minute=1) # scheduled every 1 min

    mask_start = all_calls["scheduled_time"] >= deployed_time_start
    mask_end = all_calls["scheduled_time"] < deployed_time_end

    # Check that extracted schedule from a given historical time is correct
    initial_call_id = all_calls[mask_start & mask_end]["id"].iloc[0]

    get_func = prepare_ts.client.functions.retrieve(external_id=prepare_ts.data["function_name"])
    get_schedule_id = prepare_ts.client.functions.schedules.list(
                name=prepare_ts.data["function_name"]).to_pandas().id[0]
    get_calls = get_func.list_calls(
                schedule_id=get_schedule_id, limit=-1).to_pandas()
    true_initial_call_id = get_calls["id"].iloc[0] # first call was time of backfilling - this is the one to compare with

    assert initial_call_id == true_initial_call_id


def test_backfilling_unchanged(prepare_ts):
    """Test that no changes have been made to output signal during backfilling
    when no changes in input signal.

    Args:
        prepare_ts (_type_): _description_
    """
    ts_input_name = "VAL_17-FI-9101-286:VALUE.COPY"
    ts_output_name = "test_CF"

    calculation = prepare_ts["calculation_function"]

    _, _, \
        backfill_dates, num_dates_old, num_dates_new = prepare_ts.check_backfilling(ts_input_name, testing=True)

    # Assert that all dates within overlapping period are equal after backfilling
    assert all(num_dates_new.index.isin(num_dates_old.index))
    assert len(backfill_dates) == 0

    df_out_before = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name)
    data = prepare_ts.get_orig_timeseries(eval(calculation))
    df_out_after = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name)

    assert df_out_before == df_out_after

def test_backfilling_insert(prepare_ts):
    """Test that number of datapoints in output signal has increased after backfilling
    when datapoints have been inserted in input signal .


    Args:
        prepare_ts (_type_): _description_
    """
    ts_input_name = "VAL_17-FI-9101-286:VALUE.COPY"
    ts_output_name = "test_CF"

    calculation = prepare_ts["calculation_function"]

    df_out_before = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name).to_pandas()
    # Insert data into copied input
    now = pd.Timestamp.now()
    new_data = [
        (datetime(now.year, now.month, now.day, now.hour-2, now.minute), 111),
        (datetime(now.year, now.month, now.day, now.hour-1, now.minute), 333)
    ]
    prepare_ts.client.time_series.data.insert(new_data, external_id=ts_input_name)

    data = prepare_ts.get_orig_timeseries(eval(calculation))

    df_out_after = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name).to_pandas()

    assert len(df_out_after) > len(df_out_before)

    mask_timestamp1_before = df_out_before.index == new_data[0][0]
    mask_timestamp2_before = df_out_before.index == new_data[1][0]
    mask_timestamp1_after = df_out_after.index == new_data[0][0]
    mask_timestamp2_after = df_out_after.index == new_data[1][0]

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
        deploy_schedule (_type_): _description_
    """
    ts_input_name = "VAL_17-FI-9101-286:VALUE.COPY"
    ts_output_name = "test_CF"

    calculation = prepare_ts["calculation_function"]

    df_out_before = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name).to_pandas()
    # Delete data from copied input
    now = pd.Timestamp.now()
    start_delete = datetime(now.year, now.month, now.day, now.hour-2, now.minute)
    end_delete = datetime(now.year, now.month, now.day, now.hour, now.minute)

    prepare_ts.client.time_series.data.delete_range(start=start_delete, end=end_delete, external_id=ts_input_name)

    data = prepare_ts.get_orig_timeseries(eval(calculation))

    df_out_after = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name).to_pandas()

    start_before = df_out_before.index >= start_delete
    end_before = df_out_before.index < end_delete
    start_after = df_out_after.index >= start_delete # inclusive start date
    end_after = df_out_after.index < end_delete # exclusive end date

    # Assert that num datapoints inside deleted range has decreased
    assert len(df_out_after[start_after & end_after]) < len(df_out_before[start_before & end_before])
    # Assert that num datapoints outsie deleted range remains unchanged
    assert len(df_out_after[~(start_after & end_after)]) == len(df_out_before[~(start_before & end_before)])











