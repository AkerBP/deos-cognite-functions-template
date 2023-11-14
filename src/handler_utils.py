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


def get_orig_timeseries(client, data_dicts, transform_func):
    end_date = pd.Timestamp.now() #datetime(2023, 11, 14, 16, 30)
    # from the start (00:00:00) of end_date
    start_date = pd.to_datetime(end_date.date())
    print(f"Timestamp.now: {end_date}")

    for ts in data_dicts.keys():
        data = data_dicts[ts]
        # STEP 0: Unfold data
        ts_input_name = data['ts_input_name']
        ts_output_name = data["ts_output_name"]

        data["start_date"] = start_date
        data["end_date"] = end_date

        # STEP 1: Retrieve time series and function schedules
        ts_orig = client.time_series.list(
            name=ts_input_name).to_pandas()  # original time series (vol percentage)

        data["ts_orig_extid"] = ts_orig.external_id[0]

        ts_leak = client.time_series.list(
            name=ts_output_name).to_pandas()  # transformed time series (leakage)
        # Check if transformed time series already exists
        data["ts_exists"] = not ts_leak.empty

        my_schedule_id, all_calls = get_schedules_and_calls(client, data)
        data["schedule_id"] = my_schedule_id
        data["scheduled_calls"] = all_calls

        # STEP 2: Create new time series, if not already exists
        create_timeseries(client, data)
        df_orig_backfill = str({ts_input_name: json.dumps(None)})

        # STEP 3: Possibly perform backfilling
        # TODO: Change to 23 hours and 45 minutes.
        # NB: When running on schedule, now() is 2 hours BEFORE specified hour!
        if end_date.hour == 16 and end_date.minute >= 30 and end_date.minute < 45 and data["ts_exists"]:
            df_orig_backfill = check_backfilling(client, data, transform_func)

        # STEP 4: Retrieve original time series for current date
        df_orig_today = retrieve_orig_ts(client, data)

        if not data["ts_exists"]:  # return full original signal
            df_orig_backfill = df_orig_today.copy()
            df_orig_backfill = df_orig_backfill[data["ts_input_name"]].to_json()

        data["df_orig_today"] = df_orig_today
        data["df_orig_backfill"] = df_orig_backfill

    return data_dicts


def get_schedules_and_calls(client, data):
    my_func = client.functions.retrieve(external_id=data["function_name"])
    try:
        my_schedule_id = client.functions.schedules.list(
            name=data["function_name"]).to_pandas().id[0]
        all_calls = my_func.list_calls(
            schedule_id=my_schedule_id, limit=-1).to_pandas()
    except:  # No schedule exist
        my_schedule_id = None
        all_calls = pd.DataFrame()
    return my_schedule_id, all_calls


def create_timeseries(client, data):
    if not data["ts_exists"] and not data["scheduled_calls"].empty:
        # Schedule is set up before initial write has been done locally. Abort schedule!
        client.functions.schedules.delete(id=data["schedule_id"])
        print(f"Cognite Functions can't do initial transformation. Make sure to first run handler.py locally before deploying a schedule for your Cognite Function. \
                 Deleting ... \nSchedule with id {data['schedule_id']} has been deleted.")
        sys.exit()
    elif not data["ts_exists"]:
        print("Output time series does not exist. Creating ...")
        # client.time_series.delete(external_id=ts_output_name)
        client.time_series.create(TimeSeries(
            name=data["ts_output_name"], external_id=data["ts_output_name"], data_set_id=data['dataset_id']))


def retrieve_orig_ts(client, data):
    ts_input_name = data["ts_input_name"]
    ts_output_name = data["ts_output_name"]
    ts_orig_extid = data["ts_orig_extid"]

    start_date = data["start_date"]
    end_date = data["end_date"]

    # If no data in output time series, run cognite function from first available date of original time series until date with last updated datapoint
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


def check_backfilling(client, data, transform_func):
    """Runs a backfilling for last data["backfill_days"] days of input time series.

    Args:
        client (Cognite.Client): client for authentication with Cognite
        data (dict): input data to Cognite Function
        transform_func (function): transformations/calculations for a Cognite Function

    Returns:
        (dict): jsonified version of original signal (last 7 day period)
    """
    ts_input_name = data["ts_input_name"]
    ts_output_name = data["ts_output_name"]
    ts_orig_extid = data["ts_orig_extid"]

    end_date = data["end_date"]
    start_date = end_date - timedelta(days=data["backfill_days"])
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
        {"Date": pd.to_datetime(ts_orig_all.index.date),
         ts_input_name: ts_orig_all[ts_input_name]},
        index=pd.to_datetime(ts_orig_all.index))

    # ---------------- get_yesterday_orig_signal() ------------------
    my_func = client.functions.retrieve(external_id=data["function_name"])
    scheduled_calls = data["scheduled_calls"]

    if scheduled_calls.empty:
        print("No schedule called yet. Nothing to compare with to do backfilling!")
        return ts_orig_all[[ts_input_name]].to_json()

    # ----------------
    now = pd.Timestamp.now() #datetime(2023, 11, 14, 16, 30)  # provided in local time
    # ----------------
    start_time = datetime(now.year, now.month, now.day-1,
                          16, 30)  # -1 to get previous day
    start_time = pytz.utc.localize(
        start_time).timestamp() * 1000  # convert to local time
    end_time = datetime(now.year, now.month, now.day-1, 16, 45)
    end_time = pytz.utc.localize(end_time).timestamp() * 1000

    mask_start = scheduled_calls["scheduled_time"] >= start_time
    mask_end = scheduled_calls["scheduled_time"] < end_time

    try:
        last_backfill_id = scheduled_calls[mask_start & mask_end]["id"].iloc[0]
    except:  # No scheduled call from yesterday --> nothing to compare with to do backfilling!
        print(
            f"No schedule from yesterday. Can't backfill. Returning original signal from last {data['backfill_days']} days.")
        return ts_orig_all[[ts_input_name]].to_json()

    last_backfill_call = my_func.retrieve_call(id=last_backfill_id)
    print(
        f"Retrieving scheduled call from yesterday with id {last_backfill_id}. Backfilling time series for last {data['backfill_days']} days ...")

    output_dict = ast.literal_eval(last_backfill_call.get_response())[
        data["ts_input_name"]]

    output_df = pd.DataFrame.from_dict([output_dict]).T
    output_df.index = pd.to_datetime(
        output_df.index.astype(np.int64), unit="ms")
    output_df["Date"] = output_df.index.date  # astype(int)*1e7 for testing

    # Column created with standard value 0 ...
    yesterday_df = output_df.rename(columns={0: ts_input_name})
    # -----------------

    if not yesterday_df.empty:  # empty if no scheduled call from yesterday
        print("Dates from yesterday's signal: ", yesterday_df.index.values)
        print("Dates from today's signal: ", ts_orig_dates.index.values)
        # 1. Only include overlapping parts of signal from today and yesterday
        backfill_date_start = ts_orig_dates.index[0]
        backfill_date_stop = yesterday_df.index[-1]
        yesterday_df = yesterday_df[yesterday_df.index >= backfill_date_start]
        ts_orig_dates = ts_orig_dates[ts_orig_dates.index <= backfill_date_stop]

        # 2. Store number of data points in ORIGINAL signal for each date, for yesterday and today
        num_dates_old = yesterday_df.groupby(yesterday_df["Date"]).count()
        num_dates_old.index = pd.to_datetime(num_dates_old.index)
        num_dates_old = num_dates_old.rename(columns={ts_input_name: "Datapoints"})

        num_dates_new = ts_orig_dates.groupby(ts_orig_dates["Date"]).count()
        num_dates_new = num_dates_new.rename(columns={ts_input_name: "Datapoints"})

        missing_dates = num_dates_new[~num_dates_new.index.isin(
            num_dates_old.index)].index
        missing_dates = pd.DataFrame({"Datapoints":
                                      np.zeros(len(missing_dates), dtype=np.int32)}, index=missing_dates)

        # New df with zero count for missing dates
        num_dates_old = pd.concat([num_dates_old, missing_dates]).sort_index()
        # print("num_dates_old: ", num_dates_old)
        # print("num_dates_new: ", num_dates_new)

        # 3. Only backfill if num datapoints have INCREASED or DECREASED for any dates
        increased_dates = num_dates_new[num_dates_new["Datapoints"] >
                                        num_dates_old["Datapoints"]].index
        print(f"Backfilling. Dates with NEW data: {increased_dates.values}")

        decreased_dates = num_dates_new[num_dates_new["Datapoints"] <
                                        num_dates_old["Datapoints"]].index
        print(f"Backfilling. Dates with DELETED data: {decreased_dates.values}")
        backfill_dates = increased_dates.union(decreased_dates, sort=True)

        # 4. Redo transformations for modified dates
        for date in backfill_dates:
            start_date = pd.to_datetime(date)
            end_date = pd.to_datetime(date+timedelta(days=1))
            # NB: need to change time frame in data
            data["start_date"] = start_date
            data["end_date"] = end_date

            df_orig = retrieve_orig_ts(client, data)
            df_new = transform_func(df_orig, data)

            client.time_series.data.insert_dataframe(df_new)

    # return recent original signal
    return ts_orig_all[[ts_input_name]].to_json()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv("../authentication-ids.env")
    print(os.getenv("CLIENT_ID"))