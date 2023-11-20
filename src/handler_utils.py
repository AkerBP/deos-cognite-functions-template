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

class PrepareInputTimeSeries:
    def __init__(self, inputs, client, data_dicts):
        self.client = client
        self.inputs = inputs
        self.data_dicts = data_dicts


    def get_orig_timeseries(self, client, data_dicts, transform_func):
        """Get original time series signals used to compute new output signal from tranform_func

        Args:
            client (CogniteClient): client to authenticate with cognite
            data_dicts (dict): dictionary with input data and parameters used in transformation,
                                including global properties and sub-dictionaries (e.g. key-values for specific time series)
            transform_func (function): transformation function from run_transformation

        Returns:
            dict: updated data dictionaries
        """
        end_date = pd.Timestamp.now() #datetime(2023, 11, 14, 16, 30)
        # from the start (00:00:00) of end_date
        start_date = pd.to_datetime(end_date.date())
        data_dicts["start_time"] = start_date
        data_dicts["end_time"] = end_date
        print(f"Timestamp.now: {end_date}")

        ts_output = data_dicts["ts_output"]

        for ts_out_name in ts_output.keys():
            ts_leak = client.time_series.list(
                    name=ts_out_name).to_pandas()  # transformed time series (leakage)
            # Check if transformed time series already exists
            ts_output[ts_out_name]["exists"] = not ts_leak.empty

        my_schedule_id, all_calls = get_schedules_and_calls(client, data_dicts)
        data_dicts["schedule_id"] = my_schedule_id
        data_dicts["scheduled_calls"] = all_calls

        # Create new time series, if not already exists
        create_timeseries(client, data_dicts)

        data_dicts["ts_input_today"] = {ts_name: [] for ts_name in data_dicts["ts_input"].keys()} # stores original signal only for current date
        data_dicts["ts_input_backfill"] = {ts_name: [] for ts_name in data_dicts["ts_input"].keys()} # stores original signal for entire backfill period (to be compared when doing next backfilling)

        ts_inputs = data_dicts['ts_input']
        ts_outputs = data_dicts['ts_output']

        ts_output_names = [name for name in ts_outputs.keys()]
        if len(ts_inputs.keys()) > len(ts_outputs.keys()): # multiple input time series used to compute one output time series
            ts_output_names = ts_output_names*len(ts_inputs.keys())

        for ts_in, ts_out in zip(ts_inputs.keys(), ts_output_names):
            data_in = ts_inputs[ts_in]
            data_out = ts_outputs[ts_out]

            # STEP 2: Retrieve time series and function schedules
            ts_orig = client.time_series.list(
                name=ts_in).to_pandas()  # original time series (vol percentage)

            data_in["ts_orig_extid"] = ts_orig.external_id[0]

            ts_input_backfill = str(json.dumps(None))

            # STEP 3: Identify backfill candidates
            backfill_dates = []
            # TODO: Change to 23 hours and 45 minutes.
            # NB: When running on schedule, now() is 2 hours BEFORE specified hour!
            if end_date.hour == 13 and end_date.minute >= 30 and end_date.minute < 45 and data_out["exists"]:
                ts_input_backfill, backfill_dates = check_backfilling(client, data_dicts, ts_in)

            data_dicts["ts_input_backfill"][ts_in] = ts_input_backfill
            # STEP 4: Perform backfilling on dates with discrepancies in datapoints
            for date in backfill_dates:
                data_dicts["start_time"] = pd.to_datetime(date)
                data_dicts["end_time"] = pd.to_datetime(date+timedelta(days=1))

                for ts_in, ts_out in zip(ts_inputs.keys(), ts_output_names):
                    df_orig_today = retrieve_orig_ts(client, data_dicts, ts_in, ts_out)
                    data_dicts["ts_input_today"][ts_in] = df_orig_today[ts_in]

                df_new = transform_func(data_dicts)

                client.time_series.data.insert_dataframe(df_new)

        # STEP 5: After backfilling, retrieve original signal for intended transformation period (i.e., today)
        for ts_in, ts_out in zip(ts_inputs.keys(), ts_output_names):
            data_dicts["start_time"] = start_date
            data_dicts["end_time"] = end_date

            df_orig_today = retrieve_orig_ts(client, data_dicts, ts_in, ts_out)
            data_dicts["ts_input_today"][ts_in] = df_orig_today[ts_in]

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
        ts_input = data["ts_input"]
        ts_output = data["ts_output"]

        if len(ts_input.keys()) > len(ts_output.keys()): # Assume all input time series are from same asset
            asset_ids = [client.time_series.list(name=list(ts_input.keys())[0])[0].asset_id]
        else: # same num input and output signals -->
            asset_ids = [client.time_series.list(name=ts_name)[0].asset_id for ts_name in ts_input]

        for ts_out_name, asset_id in zip(ts_output.keys(), asset_ids):
            if not ts_output[ts_out_name]["exists"] and not data["scheduled_calls"].empty:
                # Schedule is set up before initial write has been done locally. Abort schedule!
                client.functions.schedules.delete(id=data["schedule_id"])
                print(f"Cognite Functions can't do initial transformation. Make sure to first run handler.py locally before deploying a schedule for your Cognite Function. \
                        \nDeleting ... \nSchedule with id {data['schedule_id']} has been deleted.")
                sys.exit()
            elif not ts_output[ts_out_name]["exists"]:
                print(f"Output time series {ts_out_name} does not exist. Creating ...")
                client.time_series.create(TimeSeries(
                    name=ts_out_name, external_id=ts_out_name, data_set_id=data['dataset_id'], asset_id=asset_id))


    def retrieve_orig_ts(client, data_dicts, ts_in, ts_out):
        data_in = data_dicts["ts_input"][ts_in]
        data_out = data_dicts["ts_output"][ts_out]
        ts_orig_extid = data_in["ts_orig_extid"]

        start_date = data_dicts["start_time"]
        end_date = data_dicts["end_time"]
        # If no data in output time series, run cognite function from first available date of original time series until date with last updated datapoint
        if not data_out["exists"]:
            first_date_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                            aggregates="average",
                                                            granularity=f"{data_dicts['granularity']}s",
                                                            limit=1).to_pandas().index[0]
            start_date = first_date_orig

        df = pd.DataFrame()
        # If no datapoints for current date, search backwards until date with last updated datapoint
        while df.empty:
            ts_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                    aggregates="average",
                                                    granularity=f"{data_dicts['granularity']}s",
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
                print(f"No data for time series '{ts_in}' for current date. Reversing to date: {start_date}")

        df = df.rename(columns={ts_orig_extid + "|average": ts_in})

        df['time_sec'] = (df.index - datetime(1970, 1, 1)
                        ).total_seconds()  # tot seconds since epoch'
        return df

    def align_time_series(ts_df, data):
        latest_start_date = np.max([ts_df[i].index[0] for i in range(len(ts_df))])
        earliest_end_date = np.min([ts_df[i].index[-1] for i in range(len(ts_df))])

        for i in range(len(ts_df)): # omit dates where some of time series have nan values
            ts_df[i] = ts_df[i][ts_df[i].index >= latest_start_date]
            ts_df[i] = ts_df[i][ts_df[i].index <= earliest_end_date]

        time_index = pd.date_range(start=latest_start_date, end=earliest_end_date, freq=f"{data['granularity']}s")

        for i in range(len(ts_df)):
            ts_df[i] = ts_df[i].reindex(time_index, copy=False) # missing internal dates are filled with nan

        return ts_df

    def get_ts_df(data):
        """List input time series' as dataframes

        Args:
            data (dict): input parameters for Cognite Function

        Returns:
            (list): list of time series dataframes
        """
        ts_data = data["ts_input_today"]
        ts_data = [ts_data[name] for name in ts_data]
        return ts_data

    def check_backfilling(client, data_dicts, ts_input_name):
        """Runs a backfilling for last data["backfill_days"] days of input time series.

        Args:
            client (Cognite.Client): client for authentication with Cognite
            data (dict): input data to Cognite Function

        Returns:
            (dict): jsonified version of original signal (last 7 day period)
        """
        data = data_dicts["ts_input"][ts_input_name]
        ts_orig_extid = data["ts_orig_extid"]

        end_date = data_dicts["end_time"]
        start_date = end_date - timedelta(days=data_dicts["backfill_days"])
        # data['backfill'] = True
        backfill_dates = []

        # Search through prev 7 days of original time series for backfilling
        ts_orig_all = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                    aggregates="average",
                                                    granularity=f"{data_dicts['granularity']}s",
                                                    start=start_date,
                                                    end=pd.to_datetime(
                                                        end_date),
                                                    limit=-1,
                                                    ).to_pandas()

        ts_orig_all = ts_orig_all.rename(
            columns={ts_orig_extid + "|average": ts_input_name})

        my_func = client.functions.retrieve(external_id=data_dicts["function_name"])
        scheduled_calls = data_dicts["scheduled_calls"]

        # ----------------
        now = pd.Timestamp.now() #datetime(2023, 11, 14, 16, 30)  # provided in local time
        # ----------------
        start_time = datetime(now.year, now.month, now.day-1,
                            13, 30)  # -1 to get previous day
        start_time = pytz.utc.localize(
            start_time).timestamp() * 1000  # convert to local time
        end_time = datetime(now.year, now.month, now.day-1, 13, 45)
        end_time = pytz.utc.localize(end_time).timestamp() * 1000

        mask_start = scheduled_calls["scheduled_time"] >= start_time
        mask_end = scheduled_calls["scheduled_time"] < end_time

        try:
            last_backfill_id = scheduled_calls[mask_start & mask_end]["id"].iloc[0]
        except:  # No scheduled call from yesterday --> nothing to compare with to do backfilling!
            print(
                f"No schedule from yesterday. Can't backfill. Returning original signal from last {data_dicts['backfill_days']} days.")
            return ts_orig_all[[ts_input_name]].to_json(), backfill_dates

        last_backfill_call = my_func.retrieve_call(id=last_backfill_id)
        print(
            f"Retrieving scheduled call from yesterday with id {last_backfill_id}. Backfilling time series for last {data_dicts['backfill_days']} days ...")

        output_dict = ast.literal_eval(last_backfill_call.get_response())[
            ts_input_name]

        output_df = pd.DataFrame.from_dict([output_dict]).T
        output_df.index = pd.to_datetime(
            output_df.index.astype(np.int64), unit="ms")
        output_df["Date"] = output_df.index.date  # astype(int)*1e7 for testing

        # YESTERDAY's signal spanning backfilling period
        yesterday_df = output_df.rename(columns={0: ts_input_name})
        # TODAY's signal spanning backfilling period
        ts_orig_dates = pd.DataFrame(
            {"Date": pd.to_datetime(ts_orig_all.index.date),
            ts_input_name: ts_orig_all[ts_input_name]},
            index=pd.to_datetime(ts_orig_all.index))

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

            # 3. Only backfill if num datapoints have INCREASED or DECREASED for any dates
            increased_dates = num_dates_new[num_dates_new["Datapoints"] >
                                            num_dates_old["Datapoints"]].index
            print(f"Backfilling dates with NEW data: {increased_dates.values}")

            decreased_dates = num_dates_new[num_dates_new["Datapoints"] <
                                            num_dates_old["Datapoints"]].index
            print(f"Backfilling dates with DELETED data: {decreased_dates.values}")
            backfill_dates = increased_dates.union(decreased_dates, sort=True)

        # return recent original signal
        return ts_orig_all[[ts_input_name]].to_json(), backfill_dates


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv("../authentication-ids.env")
    print(os.getenv("CLIENT_ID"))