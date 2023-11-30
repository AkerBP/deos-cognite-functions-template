import sys
import os
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

# parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if parent_path not in sys.path:
#     sys.path.append(parent_path)

from transformation_utils import RunTransformations

class PrepareTimeSeries:
    """Class to organize input time series and prepare output time series
    for transformations with Cognite Functions.
    """
    def __init__(self, ts_input_names, ts_output_names, client, data_dicts):
        self.client = client
        self.ts_input_names = ts_input_names
        self.ts_output_names = ts_output_names
        self.data = data_dicts

        self.update_ts("ts_input")
        self.update_ts("ts_output")

    def update_ts(self, field, val=0):
        if field == "ts_input":
            self.data["ts_input"] = {str(name):{"exists":isinstance(name,str)} for name in self.ts_input_names} # include boolean to check if input is an already existing time series from CDF
        elif field == "ts_output":
            self.data["ts_output"] = {name:{} for name in self.ts_output_names}
        else:
            self.data[field] = val

    def get_orig_timeseries(self, calc_func):
        """Get original time series signals used to compute new output signal from tranform_func

        Args:
            calc_func (function): calculation that transforms time series.
                                should take a data dictionary 'data' and
                                list of time series dataframes 'ts_df' as input,
                                i.e., calc_func(data, *ts_df)

        Returns:
            dict: updated data dictionaries
        """
        client = self.client

        end_date = pd.Timestamp.now() #datetime(2023, 11, 14, 16, 30)
        # from the start (00:00:00) of end_date
        start_date = pd.to_datetime(end_date.date())
        self.data["start_time"] = start_date
        self.data["end_time"] = end_date
        print(f"Timestamp.now: {end_date}")

        ts_inputs = self.data["ts_input"]
        ts_outputs = self.data["ts_output"]

        for ts_out_name in ts_outputs.keys():
            ts_leak = client.time_series.list(
                    name=ts_out_name).to_pandas()  # transformed time series (leakage)
            # Check if transformed time series already exists
            ts_outputs[ts_out_name]["exists"] = not ts_leak.empty

        my_schedule_id, all_calls = self.get_schedules_and_calls()
        self.data["schedule_id"] = my_schedule_id
        self.data["scheduled_calls"] = all_calls

        # Create new time series, if not already exists
        self.create_timeseries()

        self.data["ts_input_today"] = {ts_name: [] for ts_name in ts_inputs.keys()} # stores original signal only for current date
        self.data["ts_input_backfill"] = {ts_name: [] for ts_name in ts_inputs.keys()} # stores original signal for entire backfill period (to be compared when doing next backfilling)

        ts_output_names = [name for name in ts_outputs.keys()]
        if len(ts_inputs.keys()) > len(ts_outputs.keys()): # multiple input time series used to compute one output time series
            ts_output_names = ts_output_names*len(ts_inputs.keys())

        for ts_in, ts_out in zip(ts_inputs.keys(), ts_output_names):

            #TODO: CHECK FIRST IF INPUT DATA IS TIME SERIES OR SEPARATELY PROVIDED SCALAR/ARRAY
            data_in = ts_inputs[ts_in]
            data_out = ts_outputs[ts_out]

            if not data_in["exists"]:
                continue # skip to next input

            # STEP 2: Retrieve time series and function schedules
            ts_orig = client.time_series.list(
                name=ts_in).to_pandas()  # original time series (vol percentage)

            try:
                data_in["ts_orig_extid"] = ts_orig.external_id[0]
            except:
                raise KeyError(f"Input time series {ts_in} does not exist.")

            ts_input_backfill = str(json.dumps(None))

            # STEP 3: Identify backfill candidates
            backfill_dates = []
            # TODO: Change to 23 hours and 45 minutes.
            # NB: When running on schedule, now() is 2 hours BEFORE specified hour!
            if end_date.hour == self.data["backfill_hour"] and \
            end_date.minute >= self.data["backfill_min_start"] and \
            end_date.minute < self.data["backfill_min_end"] \
            and data_out["exists"]:
                ts_input_backfill, backfill_dates = self.check_backfilling(ts_in)

            self.data["ts_input_backfill"][ts_in] = ts_input_backfill
            # STEP 4: Perform backfilling on dates with discrepancies in datapoints
            for date in backfill_dates:
                self.data["start_time"] = pd.to_datetime(date)
                self.data["end_time"] = pd.to_datetime(date+timedelta(days=1))

                for ts_in, ts_out in zip(ts_inputs.keys(), ts_output_names):
                    df_orig_today = self.retrieve_orig_ts(ts_in, ts_out)
                    self.data["ts_input_today"][ts_in] = df_orig_today[ts_in]

                ts_df = self.get_ts_df()
                ts_df = self.align_time_series(ts_df)

                transform_timeseries = RunTransformations(self.data, ts_df)
                ts_out = transform_timeseries(calc_func)
                df_out = transform_timeseries.store_output_ts(ts_out)

                client.time_series.data.insert_dataframe(df_out)

        # STEP 5: After backfilling, retrieve original signal for intended transformation period (i.e., today)
        for ts_in, ts_out in zip(ts_inputs.keys(), ts_output_names):
            if not ts_inputs[ts_in]["exists"]:
                self.data["ts_input_today"][ts_in] = float(ts_in)
                continue # skip to next input

            self.data["start_time"] = start_date
            self.data["end_time"] = end_date

            df_orig_today = self.retrieve_orig_ts(ts_in, ts_out)
            self.data["ts_input_today"][ts_in] = df_orig_today[ts_in]

        return self.data


    def get_schedules_and_calls(self):
        data = self.data
        client = self.client

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


    def create_timeseries(self):
        client = self.client
        data = self.data
        ts_input = data["ts_input"]
        ts_output = data["ts_output"]

        if len(ts_input.keys()) > len(ts_output.keys()): # Assume all input time series are from same asset
            asset_ids = [client.time_series.list(name=list(ts_input.keys())[0])[0].asset_id]
        else: # same num input and output signals --> output time series linked to asset of associated input time series
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

        return asset_ids


    def retrieve_orig_ts(self, ts_in, ts_out):
        client = self.client
        data = self.data
        data_in = self.data["ts_input"][ts_in]
        data_out = self.data["ts_output"][ts_out]
        ts_orig_extid = data_in["ts_orig_extid"]

        start_date = data["start_time"]
        end_date = data["end_time"]
        # If no data in output time series, run cognite function from first available date of original time series until date with last updated datapoint
        if not data_out["exists"]:
            first_date_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                            aggregates="average",
                                                            granularity=f"{data['granularity']}s",
                                                            limit=1).to_pandas().index[0]
            start_date = first_date_orig

        df = pd.DataFrame()
        # If no datapoints for current date, search backwards until date with last updated datapoint
        while df.empty:
            ts_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                    aggregates="average",
                                                    granularity=f"{data['granularity']}s",
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

        return df

    def align_time_series(self, ts_all):
        ts_df = [[i, ts] for i, ts in enumerate(ts_all) if not isinstance(ts, float)]
        ts_scalars = [[i, ts] for i, ts in enumerate(ts_all) if isinstance(ts, float)]

        latest_start_date = np.max([ts_df[i][1].index[0] for i in range(len(ts_df))])
        earliest_end_date = np.min([ts_df[i][1].index[-1] for i in range(len(ts_df))])

        for i in range(len(ts_df)): # omit dates where some of time series have nan values
            ts_df[i][1] = ts_df[i][1][ts_df[i][1].index >= latest_start_date]
            ts_df[i][1] = ts_df[i][1][ts_df[i][1].index <= earliest_end_date]

        time_index = pd.date_range(start=latest_start_date, end=earliest_end_date, freq=f"{self.data['granularity']}s")

        ts_all = [0]*len(ts_all)
        for i in range(len(ts_df)):
            ts_df[i][1] = ts_df[i][1].reindex(time_index, copy=False) # missing internal dates are filled with nan
            ts_all[ts_df[i][0]] = ts_df[i][1]

        for i in range(len(ts_scalars)):
            ts_scalars[i][1] = pd.DataFrame(ts_scalars[i][1]*np.ones(len(ts_df[0][1])), index=ts_df[0][1].index) # can simply choose one of the dfs, they have the same index at this point anyway
            ts_all[ts_scalars[i][0]] = ts_scalars[i][1]

        return ts_all

    def get_ts_df(self):
        """List input time series' as dataframes

        Args:
            data (dict): input parameters for Cognite Function

        Returns:
            (list): list of time series dataframes (and potentially scalars)
        """
        ts_data = self.data["ts_input_today"]
        ts_data = [ts_data[name] if isinstance(ts_data[name], float) else pd.DataFrame(ts_data[name]) for name in ts_data]
        return ts_data

    def check_backfilling(self, ts_input_name, testing=False):
        """Runs a backfilling for last data["backfill_days"] days of input time series.

        Args:
            ts_input_name (str)

        Returns:
            (dict): jsonified version of original signal (last self.data['backfill_days'] days period)
        """
        client = self.client
        data = self.data
        ts_input = data["ts_input"][ts_input_name]
        ts_orig_extid = ts_input["ts_orig_extid"]

        end_date = data["end_time"]
        start_date = end_date - timedelta(days=data["backfill_days"])
        backfill_dates = []

        # Search through prev 7 days of original time series for backfilling
        ts_orig_all = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                    aggregates="average",
                                                    granularity=f"{data['granularity']}s",
                                                    start=start_date,
                                                    end=pd.to_datetime(
                                                        end_date),
                                                    limit=-1,
                                                    ).to_pandas()

        ts_orig_all = ts_orig_all.rename(
            columns={ts_orig_extid + "|average": ts_input_name})

        my_func = client.functions.retrieve(external_id=data["function_name"])
        scheduled_calls = data["scheduled_calls"]

        # ----------------
        now = pd.Timestamp.now() #datetime(2023, 11, 14, 16, 30)  # provided in local time
        # ----------------

        if testing: # when testing backfilling, we only move some minutes back in time, not an entire day
            backfill_day = now.day
        else:
            backfill_day = now.day-1
        start_time = datetime(now.year, now.month, backfill_day,
                            data["backfill_hour"], data["backfill_min_start"])  # -1 to get previous day
        start_time = pytz.utc.localize(
            start_time).timestamp() * 1000  # convert to local time
        end_time = datetime(now.year, now.month, backfill_day, data["backfill_hour"], data["backfill_min_end"])
        end_time = pytz.utc.localize(end_time).timestamp() * 1000

        try:
            mask_start = scheduled_calls["scheduled_time"] >= start_time
            mask_end = scheduled_calls["scheduled_time"] < end_time
            last_backfill_id = scheduled_calls[mask_start & mask_end]["id"].iloc[0]
        except:  # No scheduled call from yesterday --> nothing to compare with to do backfilling!
            print(
                f"Input {ts_input_name}: No schedule from yesterday. Can't backfill. Returning original signal from last {data['backfill_days']} days.")
            return ts_orig_all[[ts_input_name]].to_json(), backfill_dates

        last_backfill_call = my_func.retrieve_call(id=last_backfill_id)
        print(
            f"Retrieving scheduled call from yesterday with id {last_backfill_id}. Backfilling time series for last {data['backfill_days']} days ...")

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

        if testing:
            return ts_orig_all, yesterday_df, backfill_dates, \
                    num_dates_old, num_dates_new
        # return recent original signal
        return ts_orig_all[ts_input_name].to_json(), backfill_dates


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv("../authentication-ids.env")
    print(os.getenv("CLIENT_ID"))