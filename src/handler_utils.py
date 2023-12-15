import sys
import os
from datetime import datetime, timedelta
from typing import Tuple
import pandas as pd
import numpy as np
import ast
import json
import logging

from cognite.client.data_classes import TimeSeries
from cognite.client._cognite_client import CogniteClient
from cognite.client.exceptions import CogniteAPIError

logger = logging.getLogger(__name__)

# parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if parent_path not in sys.path:
#     sys.path.append(parent_path)

from transformation_utils import RunTransformations

class FunctionDeployError(Exception):
    pass

class PrepareTimeSeries:
    """Class to organize input time series and prepare output time series
    for transformations with Cognite Functions.
    """
    def __init__(self, ts_input_names: list, ts_output_names: list,
                 client: CogniteClient, data_dicts: dict):
        """Provide client and data dictionary for deployment of Cognite Function

        Args:
            ts_input_names (list): names for input time series
            ts_output_names (list): names for output time series
            client (CogniteClient): instantiated CogniteClient
            data_dicts (dict): data dictionary used for Cognite Function
        """
        self.client = client
        self.ts_input_names = ts_input_names
        self.ts_output_names = ts_output_names
        self.data = data_dicts

        self.update_ts("ts_input")
        self.update_ts("ts_output")

    def update_ts(self, field: str, val=None):
        """Update provided field in data dictionary for Cognite Functions.

        Args:
            field (str): key in dictionary to update
            val (optional): value assigned to the field. Defaults to None.
        """
        if field == "ts_input":
            self.data["ts_input"] = {str(name):{"exists":isinstance(name,str)} for name in self.ts_input_names} # include boolean to check if input is an already existing time series from CDF
        elif field == "ts_output":
            self.data["ts_output"] = {name:{"exists":False} for name in self.ts_output_names}
        else:
            self.data[field] = val

    def get_orig_timeseries(self, calc_func) -> dict:
        """Get original time series signals used to compute new output signal,
        performing backfilling if scheduled.

        Args:
            calc_func (function): calculation that transforms time series.
                                should take a data dictionary 'data' and
                                list of time series dataframes 'ts_df' as input,
                                i.e., calc_func(data, *ts_df), and return a
                                pd.Series object of transformed data points
                                for a given date range.

        Returns:
            (dict): updated data dictionaries
        """
        client = self.client

        end_date = pd.Timestamp.now()
        # start_date = pd.to_datetime(end_date.date())
        start_date = end_date - timedelta(minutes=self.data["backfill_min_end"] - self.data["backfill_min_start"])
        if "start_time" in self.data.keys():
            start_date = self.data["start_time"] # overwrite default start time defined by schedule (relevant for certain cases, e.g., date-specific aggregated)

        self.data["start_time"] = start_date
        self.data["end_time"] = end_date

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

            data_in = ts_inputs[ts_in]
            data_out = ts_outputs[ts_out]

            if not data_in["exists"]:
                continue # input not in CDF, provided separately, skip to next input

            # STEP 2: Retrieve time series and function schedules
            ts_orig = client.time_series.list(
                name=ts_in).to_pandas()  # original time series (vol percentage)

            try:
                data_in["ts_orig_extid"] = ts_orig.external_id[0]
            except:
                raise KeyError(f"Input time series {ts_in} does not exist.")

            ts_input_backfill = json.dumps(None)

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
                    if not ts_inputs[ts_in]["exists"]:
                        self.data["ts_input_today"][ts_in] = float(ts_in)
                        continue

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
                continue # input not from CDF, skip to next input

            self.data["start_time"] = start_date
            self.data["end_time"] = end_date

            df_orig_today = self.retrieve_orig_ts(ts_in, ts_out)
            self.data["ts_input_today"][ts_in] = df_orig_today[ts_in]

        self.data["ts_input_backfill"] = json.dumps(self.data["ts_input_backfill"])
        return self.data


    def get_schedules_and_calls(self) -> Tuple[int, pd.DataFrame]:
        """Return ID of schedule that is currently running for this function,
        and a list of all calls made to this schedule.

        Returns:
            (int): id of schedule
            (pd.DataFrame): table of calls made
        """

        data = self.data
        client = self.client

        my_func = client.functions.retrieve(external_id=data["function_name"])
        if my_func is None:
            err = f"No function with external_id={data['function_name']} exists."
            logger.warning(err)
            raise FunctionDeployError(err)

        try:
            my_schedule_id = client.functions.schedules.list(
                name=data["function_name"]).to_pandas().id[0]
            all_calls = my_func.list_calls(
                schedule_id=my_schedule_id, limit=-1).to_pandas()
        except:  # No schedule exist
            my_schedule_id = None
            all_calls = pd.DataFrame()

        return my_schedule_id, all_calls


    def create_timeseries(self) -> list:
        """Create Time Series object for output time series if not exist.

        Returns:
            list: asset ids of associated input time series
        """
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
                print(f"Cognite Functions schedules can't do initial transformation. Make sure to first run handler.py locally before deploying a schedule for your Cognite Function. \
                        \nDeleting ... \nSchedule with id {data['schedule_id']} has been deleted.")
                sys.exit()
            elif not ts_output[ts_out_name]["exists"]:
                print(f"Output time series {ts_out_name} does not exist. Creating ...")
                client.time_series.create(TimeSeries(
                    name=ts_out_name, external_id=ts_out_name, data_set_id=data['dataset_id'], asset_id=asset_id))

        return asset_ids


    def retrieve_orig_ts(self, ts_in: str, ts_out: str) -> pd.DataFrame:
        """Return input time series over given date range, averaged over a given granularity.

        Args:
            ts_in (str): name of input time series
            ts_out (str): name of output time series (to be produced)

        Returns:
            pd.DataFrame: input time series
        """
        client = self.client
        data = self.data
        data_in = self.data["ts_input"][ts_in]
        data_out = self.data["ts_output"][ts_out]
        ts_orig_extid = data_in["ts_orig_extid"]

        start_date = data["start_time"]
        end_date = data["end_time"]

        try:
            # If no data in output time series, run cognite function from first available date of original time series until date with last updated datapoint
            if not data_out["exists"]:
                first_date_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                                aggregates="average",
                                                                granularity=f"{data['granularity']}s",
                                                                limit=1).to_pandas().index[0]
                start_date = first_date_orig

            df = pd.DataFrame()
            # If no datapoints for current interval, search backwards until first interval with valid datapoints
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

        except CogniteAPIError as error:
            msg = "Unable to read time series. Deployment key is missing capability 'timeseries:READ'"
            logger.error(msg)
            raise CogniteAPIError(msg, error.code, error.x_request_id) from None


    def align_time_series(self, ts_all: list) -> list:
        """Align input time series to cover same data range.

        Args:
            ts_all (list): list of input time series (pd.DataFrame or scalars)

        Returns:
            (pd.DataFrame): dataframe of time series truncated by common dates
        """
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

        ts_all = pd.concat(ts_all, axis=1) # concatenate along date

        return ts_all

    def get_ts_df(self) -> list:
        """List input time series' from CDF as dataframes and scalars as floats

        Returns:
            (list): list of time series dataframes (and potentially scalars)
        """
        ts_data = self.data["ts_input_today"]
        ts_data = [ts_data[name] if isinstance(ts_data[name], float) else pd.DataFrame(ts_data[name]) for name in ts_data]
        return ts_data

    def check_backfilling(self, ts_input_name: str, testing: bool = False) -> Tuple[dict, list]:
        """Runs a backfilling for last self.data["backfill_days"] days of input time series.

        Args:
            ts_input_name (str): name of input time series
            testing (bool): If running unit test or not. Defaults to False.

        Returns:
            (dict): jsonified version of original signal for last self.data['backfill_days'] days
            (list): dates to backfill data
        """
        client = self.client
        data = self.data
        ts_input = data["ts_input"][ts_input_name]
        ts_orig_extid = ts_input["ts_orig_extid"]

        end_date = data["end_time"]
        start_date = end_date - timedelta(days=data["backfill_days"])
        backfill_dates = []

        # Search through prev X=data["backfill_days"] days of original time series for backfilling
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
        now = pd.Timestamp.now().date() #datetime(2023, 11, 14, 16, 30)  # provided in local time
        # ----------------

        if testing: # when testing backfilling, we only move some minutes back in time, not an entire day
            backfill_day = now.day
            backfill_month = now.month
            backfill_year = now.year
        else:
            backfill_date = (now - timedelta(days=1)) #(now - timedelta(hours=1))
            backfill_day = backfill_date.day
            backfill_month = backfill_date.month
            backfill_year = backfill_date.year

        SEC_SINCE_EPOCH = datetime(1970, 1, 1, 0, 0)
        start_time = datetime(backfill_year, backfill_month, backfill_day,
                            data["backfill_hour"], data["backfill_min_start"])  # pd.Timestamp.now().hour-1, data["backfill_min_start"])
        start_time = (start_time - SEC_SINCE_EPOCH).total_seconds() * 1000  # convert to local time

        end_time = datetime(backfill_year, backfill_month, backfill_day,
                            data["backfill_hour"], data["backfill_min_end"])
        end_time = (end_time - SEC_SINCE_EPOCH).total_seconds() * 1000

        try:
            mask_start = scheduled_calls["scheduled_time"] >= start_time
            mask_end = scheduled_calls["scheduled_time"] < end_time
            last_backfill_id = scheduled_calls[mask_start & mask_end]["id"].iloc[0]
        except:  # No scheduled call from yesterday --> nothing to compare with to do backfilling!
            print(
                f"Input {ts_input_name}: No schedule from yesterday. Can't backfill. Returning original signal from last {data['backfill_days']} days.")
            orig_as_dict = ast.literal_eval(ts_orig_all[ts_input_name].to_json())
            return orig_as_dict, backfill_dates

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
        orig_as_dict = ast.literal_eval(ts_orig_all[ts_input_name].to_json())
        return orig_as_dict, backfill_dates


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv("../authentication-ids.env")
    print(os.getenv("CLIENT_ID"))