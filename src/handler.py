from datetime import datetime, timedelta
import time
import pytz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cognite.client.data_classes import TimeSeries
import sys
import ast
import json


def handle(client, data):
    """Calculate drainage rate per timestamp and per day from tank,
    using Lowess filtering on volume percentage data from the tank.
    Large positive derivatives of signal are excluded to ignore
    human interventions (filling) of tank.
    Data of drainage rate helps detecting leakages.

    Args:
        client (CogniteClient): client used to authenticate cognite session
        data (dict): data input to the handle

    Returns:
        pd.DataFrame: dataframe with drainage rate and trend (derivative)
    """
    # STEP 0: Unfold data
    if data['cdf_env'] in ["dev", "test", "prod"]:
        ts_input_name = data['ts_input_name']
        ts_output_name = data["ts_output_name"]
        end_date = pd.Timestamp.now()
        # from the start (00:00:00) of end_date
        start_date = pd.to_datetime(end_date.date())

    data["start_date"] = start_date
    data["end_date"] = end_date

    # STEP 1: Retrieve time series and function schedules
    ts_orig = client.time_series.list(
        name=ts_input_name).to_pandas()  # original time series (vol percentage)
    ts_orig_extid = ts_orig.external_id[0]

    ts_leak = client.time_series.list(
        name=ts_output_name).to_pandas()  # transformed time series (leakage)
    # Check if transformed time series already exists
    data["ts_exists"] = not ts_leak.empty

    my_schedule_id, all_calls = get_schedules_and_calls(client, data)
    data["schedule_id"] = my_schedule_id
    data["scheduled_calls"] = all_calls

    # STEP 2: Create new time series, if not already exists
    create_timeseries(client, data)
    orig_signal = str({ts_input_name: json.dumps(None)})

    # STEP 3: Possibly perform backfilling
    print(f"Timestamp.now: {end_date}")
    # TODO: Change to 23 hours and 45 minutes.
    # NB: When running on schedule, now() is 2 hours BEFORE specified hour!
    if end_date.hour == 7 and end_date.minute < 15 and data["ts_exists"]:
        orig_signal = check_backfilling(client, ts_orig_extid, data)

    # STEP 4: Retrieve original time series for current date
    df_orig = retrieve_orig_ts(client, ts_orig_extid, data)

    # STEP 5: Run transformations
    try:
        # Dynamically import from correct folder
        import importlib
        # globals().update(importlib.import_module(
        #     f"{data['function_name']}.transformation").__dict__)
        run_transformation = importlib.import_module(
            f"{data['function_name']}.transformation").__dict__["run_transformation"]
    except:
        raise NotImplementedError(
            f"Folder {data['function_name']} for doing transformation {data['function_name']} does not exist. Can't perform transformation.")
    # NB: df_new is dataframe with only one column (transformed signal) with column label given by output name
    df_new = run_transformation(df_orig, data)

    # STEP 6: Insert transformed signal for new time range
    client.time_series.data.insert_dataframe(df_new)
    # client.time_series.data.insert_dataframe(mean_df)

    if not data["ts_exists"]:  # return full original signal
        orig_signal = df_orig.copy()
        orig_signal = orig_signal[ts_input_name].to_json()

    # Store original signal (for backfilling)
    return orig_signal  # new_df[[ts_output_name]].to_json()


def get_schedules_and_calls(client, data):
    my_func = client.functions.retrieve(external_id=data["function_name"])
    try:
        my_schedule_id = client.functions.schedules.list(
            name=data["schedule_name"]).to_pandas().id[0]
        all_calls = my_func.list_calls(
            schedule_id=my_schedule_id, limit=-1).to_pandas()
    except:  # No schedule exist
        my_schedule_id = None
        all_calls = pd.DataFrame()
    return my_schedule_id, all_calls


def create_timeseries(client, data):
    if not data["ts_exists"] and not data["scheduled_calls"].empty:
        # Schedule is set up before initial write has been done locally. Abort schedule!
        client.functions.schedules.delete(id=data["my_schedule_id"])
        print(f"Cognite Functions can't do initial transformation. Make sure to first run handler.py locally before deploying a schedule for your Cognite Function. \
                 Deleting ... \nSchedule with id {data['my_schedule_id']} has been deleted.")
        sys.exit()
    elif not data["ts_exists"]:
        print("Output time series does not exist. Creating ...")
        # client.time_series.delete(external_id=ts_output_name)
        client.time_series.create(TimeSeries(
            name=data["ts_output_name"], external_id=data["ts_output_name"], data_set_id=data['dataset_id']))


def retrieve_orig_ts(client, ts_orig_extid, data):
    ts_input_name = data["ts_input_name"]
    ts_output_name = data["ts_output_name"]

    start_date = data["start_date"]
    end_date = data["end_date"]

    # If no data in output time series AND no scheduled call has been made yet, run cognite function from first available date of original time series until date with last updated datapoint
    if not data["ts_exists"]:
        first_date_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                           aggregates="average",
                                                           granularity="1m",
                                                           limit=1).to_pandas().index[0]
        start_date = first_date_orig

    df = pd.DataFrame()
    # If no datapoints for current date, search backwards until date with last updated datapoint
    while df.empty:
        ts_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                   aggregates="average",
                                                   granularity="1m",
                                                   start=pd.to_datetime(
                                                       start_date),
                                                   end=pd.to_datetime(
                                                       end_date),
                                                   )

        df = ts_orig.to_pandas()
        start_date = pd.to_datetime(start_date - timedelta(days=1)
                                    ).date()  # start of previous date
        end_date = pd.to_datetime(start_date + timedelta(days=1))
        if df.empty:
            print(f"No data for current date. Reversing to date: {start_date}")

    df = df.rename(columns={ts_orig_extid + "|average": ts_input_name})

    df['time_sec'] = (df.index - datetime(1970, 1, 1)
                      ).total_seconds()  # tot seconds since epoch'
    return df


def check_backfilling(client, ts_orig_extid, data):
    ts_input_name = data["ts_input_name"]
    ts_output_name = data["ts_output_name"]
    end_date = data["end_date"]
    start_date = end_date - timedelta(days=data["backfill_period"])
    data['backfill'] = True

    # Search through prev 7 days of original time series for backfilling
    ts_orig_all = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                   aggregates="average",
                                                   granularity="1m",
                                                   start=start_date,
                                                   end=pd.to_datetime(
                                                       end_date),
                                                   limit=-1,
                                                   ).to_pandas()

    ts_orig_all = ts_orig_all.rename(
        columns={ts_orig_extid + "|average": ts_input_name})

    ts_orig_dates = pd.DataFrame(
        {ts_input_name: pd.to_datetime(ts_orig_all.index.date)})

    # ---------------- get_yesterday_orig_signal() ------------------
    my_func = client.functions.retrieve(external_id=data["function_name"])
    scheduled_calls = data["scheduled_calls"]

    if scheduled_calls.empty:  # No schedule exist --> nothing to compare with to do backfilling!
        return ts_orig_all[[ts_input_name]].to_json()

    now = pd.Timestamp.now()  # provided in local time
    start_time = datetime(now.year, now.month, now.day-1,
                          7, 0)  # -1 to get previous day
    print("start time: ", pytz.utc.localize(start_time))
    start_time = pytz.utc.localize(
        start_time).timestamp() * 1000  # convert to local time
    end_time = datetime(now.year, now.month, now.day-1, 7, 15)
    # *1000 -> millisec since epoch
    end_time = pytz.utc.localize(end_time).timestamp() * 1000

    mask_start = scheduled_calls["scheduled_time"] >= start_time
    mask_end = scheduled_calls["scheduled_time"] < end_time

    try:
        last_backfill_id = scheduled_calls[mask_start & mask_end]["id"].iloc[0]
    except:  # No scheduled call from yesterday --> nothing to compare with to do backfilling!
        print(
            f"No schedule from yesterday. Can't backfill. Returning original signal from last {data['backfill_period']} days.")
        return ts_orig_all[[ts_input_name]].to_json()

    last_backfill_call = my_func.retrieve_call(id=last_backfill_id)
    print(
        f"Retrieving scheduled call from yesterday with id {last_backfill_id}. Backfilling time series for last {data['backfill_period']} days ...")

    output_dict = ast.literal_eval(last_backfill_call.get_response())[
        data["ts_input_name"]]

    start_30_day_period = pd.to_datetime(
        last_backfill_call.scheduled_time, unit="ms") - timedelta(days=30)
    end_30_day_period = pd.to_datetime(
        last_backfill_call.scheduled_time, unit="ms")

    output_df = pd.DataFrame.from_dict([output_dict]).T
    output_df.index = pd.to_datetime(
        output_df.index.astype(np.int64), unit="ms").date  # astype(int)*1e7 for testing
    output_df.index.name = "Date"
    # Column created with standard value 0 ...
    yesterday_df = output_df.rename(columns={0: data["ts_input_name"]})
    print("Yesterday index: ", yesterday_df.index)

    # -----------------

    if not yesterday_df.empty:  # empty if no scheduled call from yesterday

        num_dates_old = yesterday_df.groupby(yesterday_df.index).count()
        num_dates_old.index = pd.to_datetime(num_dates_old.index)
        # ----------------

        # 1. For each write, store number of data points in ORIGINAL signal for each date - store in file associated with dataset ID.
        # number of datapoints for each date
        num_dates_new = ts_orig_dates.groupby(ts_input_name)[
            ts_input_name].count()
        num_dates_new = pd.DataFrame(num_dates_new)
        print("Today index: ", num_dates_new.index)

        num_dates_new.index.name = "Date"

        # Truncate original values to backfilled period
        num_dates_old = num_dates_old[num_dates_old.index >=
                                      num_dates_new.index[0]]

        missing_dates = num_dates_new[~num_dates_new.index.isin(
            num_dates_old.index)].index
        missing_dates = pd.DataFrame({ts_input_name:
                                      np.zeros(len(missing_dates), dtype=np.int32)}, index=missing_dates)

        # New df with zero count for missing dates
        num_dates_old = pd.concat([num_dates_old, missing_dates]).sort_index()

        # 2. Backfilling: Redo transformations if num datapoints have INCREASED or DECREASED for any dates
        increased_dates = num_dates_new[num_dates_new[ts_input_name] >
                                        num_dates_old[ts_input_name]].index
        print(f"Backfilling. Dates with NEW data: {increased_dates}")

        decreased_dates = num_dates_new[num_dates_new[ts_input_name] <
                                        num_dates_old[ts_input_name]].index
        print(f"Backfilling. Dates with DELETED data: {decreased_dates}")
        backfill_dates = increased_dates.union(decreased_dates, sort=True)

        # 3. Redo transformations for modified dates
        for date in backfill_dates:
            start_date = pd.to_datetime(date)
            end_date = pd.to_datetime(date+timedelta(days=1))
            # NB: need to change time frame in data
            data["start_date"] = start_date
            data["end_date"] = end_date

            df_orig = retrieve_orig_ts(client, ts_orig_extid, data)
            df_new = run_transformation(df_orig, data)

            client.time_series.data.insert_dataframe(df_new)

    # return recent original signal
    return ts_orig_all[[ts_input_name]].to_json()


if __name__ == '__main__':
    from initialize import initialize_client
    from dotenv import load_dotenv
    import os

    cdf_env = "dev"
    if cdf_env not in ["dev", "test", "prod"]:
        token = True
    else:
        token = False

    client = initialize_client(cdf_env, cache_token=token)

    load_dotenv("../handler-data.env")

    in_name = "VAL_11-LT-95034A:X.Value"
    out_name = "VAL_11-LT-95034A:X.CDF.D.AVG.LeakValue"

    tank_volume = 1400
    derivative_value_excl = 0.002
    # start_date = datetime(2023, 3, 21, 1, 0, 0)
    func_name = "avg_drainage_rate"
    sched_name = f"{func_name}_schedule"

    data_dict = {'tot_days': 0, 'tot_minutes': 15,  # convert date to str to make it JSON serializable
                 'ts_input_name': in_name, 'ts_output_name': out_name,
                 'derivative_value_excl': derivative_value_excl, 'tank_volume': tank_volume,
                 # NB: change dataset id when going to dev/test/prod!
                 'cdf_env': cdf_env, 'dataset_id': int(os.getenv("DATASET_ID")),
                 'backfill': False, 'backfill_period': 7,
                 'function_name': func_name, 'schedule_name': sched_name,
                 'lowess_frac': 0.001, 'lowess_delta': 0.01}

    # client.time_series.delete(external_id=str(os.getenv("TS_OUTPUT_NAME")))
    new_df = handle(client, data_dict)
