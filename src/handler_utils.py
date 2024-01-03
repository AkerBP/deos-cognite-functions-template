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

        ts_inputs = self.data["ts_input"]
        ts_outputs = self.data["ts_output"]

        # TODO TODO TODO TODO ----------------------
        end_date = pd.Timestamp.now()# - timedelta(days=1) #TODO: SUBTRACTING ONE DAY ONLY FOR TESTING !
        # TODO TODO TODO TODO ----------------------

        start_date = end_date - timedelta(minutes=int(self.data["cron_interval_min"]))

        self.data["start_time"] = start_date
        self.data["end_time"] = end_date

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

        self.data["ts_input_data"] = {ts_name: [] for ts_name in ts_inputs.keys()} # stores original signal only for current date
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
                data_in["orig_extid"] = ts_orig.external_id[0]
            except:
                raise KeyError(f"Input time series {ts_in} does not exist.")

            ts_input_backfill = json.dumps(None)

            # STEP 3: Identify backfill candidates
            backfill_dates = []
            # TODO: Change to 23 hours and 45 minutes.
            # NB: When running on schedule, now() is 2 hours BEFORE specified hour!
            if self.data["testing"]:
                if int(str(end_date.minute)[-1]) == 0:
                    ts_input_backfill, backfill_dates = self.check_backfilling(ts_in, testing=True)
            elif end_date.hour == self.data["backfill_hour"] and \
                end_date.minute >= self.data["backfill_min_start"] and \
                end_date.minute < self.data["backfill_min_end"] \
                and data_out["exists"]:
                    ts_input_backfill, backfill_dates = self.check_backfilling(ts_in)

            self.data["ts_input_backfill"][ts_in] = ts_input_backfill
            # STEP 4: Perform backfilling on dates with discrepancies in datapoints
            if "period" in self.data["aggregate"] and "type" in self.data["aggregate"] \
                and len(backfill_dates) > 0:
                # If aggregates, run backfilling for last X aggregated periods
                min_backfill_date = min(backfill_dates)
                start_year = min_backfill_date.year
                start_month = 0
                start_day = 0
                start_hour = 0
                start_minute = 0
                if self.data["aggregate"]["period"] == "year":
                    pass
                elif self.data["aggregate"]["period"] == "month":
                    start_month = min_backfill_date.month
                elif self.data["aggregate"]["period"] == "day":
                    start_month = min_backfill_date.month
                    start_day = min_backfill_date.day
                elif self.data["aggregate"]["period"] == "hour":
                    start_month = min_backfill_date.month
                    start_day = min_backfill_date.day
                    start_hour = min_backfill_date.hour
                elif self.data["aggregate"]["period"] == "minute":
                    start_month = min_backfill_date.month
                    start_day = min_backfill_date.day
                    start_hour = min_backfill_date.hour
                    start_minute = min_backfill_date.minute
                else:
                    raise NotImplementedError(f"Backfilling not implemented for aggregates of period {self.data['aggregate']['period']}. Supported periods are: year, month, day, hour, minute.")

                self.data["start_time"] = datetime(start_year, start_month, start_day, start_hour, start_minute)

            else: # If no aggregates, run backfilling on each date separately
                for date in backfill_dates:
                    self.data["start_time"] = pd.to_datetime(date)
                    self.data["end_time"] = pd.to_datetime(date+timedelta(days=1))

                    for ts_in, ts_out in zip(ts_inputs.keys(), ts_output_names):
                        if not ts_inputs[ts_in]["exists"]:
                            self.data["ts_input_data"][ts_in] = float(ts_in)
                            continue

                        df_orig = self.retrieve_orig_ts(ts_in, ts_out)
                        self.data["ts_input_data"][ts_in] = df_orig[ts_in]

                    ts_df = self.get_ts_df()
                    ts_df = self.align_time_series(ts_df)

                    transform_timeseries = RunTransformations(self.data, ts_df)
                    ts_out = transform_timeseries(calc_func)
                    df_out = transform_timeseries.store_output_ts(ts_out)

                    client.time_series.data.insert_dataframe(df_out)

        # STEP 5: After backfilling, retrieve original signal for intended transformation period
        for ts_in, ts_out in zip(ts_inputs.keys(), ts_output_names):
            if not ts_inputs[ts_in]["exists"]:
                self.data["ts_input_data"][ts_in] = float(ts_in)
                continue # input not from CDF, skip to next input

            df_orig = self.retrieve_orig_ts(ts_in, ts_out)

            if "period" in self.data["aggregate"] and "type" in self.data["aggregate"]:
                start_time_schedule = self.data["start_time"]
                start_time_aggregate = self.get_aggregated_start_time()

                orig_extid = self.data["ts_input"][ts_in]["orig_extid"]
                # Retrieve all datapoints from aggregated period NOT part of current schedule
                df_orig_prev = client.time_series.data.retrieve(external_id=orig_extid,
                                                                aggregates="average",
                                                                granularity=f"{self.data['granularity']}s",
                                                                start=start_time_aggregate,
                                                                end=start_time_schedule).to_pandas()
                df_orig_prev = df_orig_prev.rename(columns={orig_extid + "|average": ts_in})
                df_orig_prev = df_orig_prev.iloc[:-1] # Omit last element as this is first element in df_orig

                df_orig_prev.index = pd.to_datetime(df_orig_prev.index)
                df_orig.index = pd.to_datetime(df_orig.index)

                # df_orig = df_orig.join(df_orig_prev)
                df_orig = pd.concat([df_orig_prev, df_orig]) # join scheduled period with remaining aggregated period
                df_orig = df_orig.apply(lambda row: getattr(row, self.data["aggregate"]["type"])(), axis=1)

                df_orig = pd.DataFrame(df_orig, columns=[ts_in])

            self.data["ts_input_data"][ts_in] = df_orig[ts_in]

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
        orig_extid = data_in["orig_extid"]

        start_date = data["start_time"]
        end_date = data["end_time"]

        try:
            # If no data in output time series, run cognite function from first available date of original time series until date with last updated datapoint
            if not data_out["exists"]:
                start_date = client.time_series.data.retrieve(external_id=orig_extid,
                                                                aggregates="average",
                                                                granularity=f"{data['granularity']}s",
                                                                limit=1).to_pandas().index[0]
                self.data["start_time"] = start_date


            ts_orig = client.time_series.data.retrieve(external_id=orig_extid,
                                                    aggregates="average",
                                                    granularity=f"{data['granularity']}s",
                                                    start=pd.to_datetime(
                                                        start_date),
                                                    end=pd.to_datetime(
                                                        end_date),
                                                    )

            df = ts_orig.to_pandas()
            if df.empty:
                print(f"No data for time series '{ts_in}' for interval: [{start_date}, {end_date}]. Skipping calculations.")

            df = df.rename(columns={orig_extid + "|average": ts_in})
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
        """List all input time series as dataframes objects

        Returns:
            (list): list of time series dataframes
        """
        ts_data = self.data["ts_input_data"]
        ts_data = [ts_data[name] if isinstance(ts_data[name], float) else pd.DataFrame(ts_data[name]) for name in ts_data]
        return ts_data

    def get_aggregated_start_time(self) -> datetime:
        """For aggregated calculations, return start time of current aggregated period.

        Raises:
            NotImplementedError: Aggregation not supported for periods other than "second", "minute", "hour", "day", "month" or "year".

        Returns:
            (datetime): start time of aggregated period
        """
        end_time_previous = self.data["start_time"] # end time of previous scheduled call

        if self.data["aggregate"]["period"] == "second":
            start_time_previous = datetime(end_time_previous.year,
                                            end_time_previous.month,
                                            end_time_previous.day,
                                            end_time_previous.hour,
                                            end_time_previous.minute,
                                            end_time_previous.second)
        elif self.data["aggregate"]["period"] == "minute":
            start_time_previous = datetime(end_time_previous.year,
                                            end_time_previous.month,
                                            end_time_previous.day,
                                            end_time_previous.hour,
                                            end_time_previous.minute)
            # start_time_previous = end_time_previous - PrepTS.data["cron_interval_min"]
        elif self.data["aggregate"]["period"] == "hour":
            start_time_previous = datetime(end_time_previous.year,
                                            end_time_previous.month,
                                            end_time_previous.day,
                                            end_time_previous.hour)
        elif self.data["aggregate"]["period"] == "day":
            start_time_previous = datetime(end_time_previous.year,
                                            end_time_previous.month,
                                            end_time_previous.day)
        elif self.data["aggregate"]["period"] == "month":
            start_time_previous = datetime(end_time_previous.year,
                                            end_time_previous.month)
        elif self.data["aggregate"]["period"] == "year":
            start_time_previous = datetime(end_time_previous.year)
        else:
            raise NotImplementedError(f"Cognite Functions Template does not support calculations for aggregation period '{self.data['aggregate']['period']}'")

        return start_time_previous

    def check_backfilling(self, ts_input_name: str, testing: bool = False) -> Tuple[dict, list]:
        """Runs a backfilling for last self.data["backfill_period"] days of input time series.

        Args:
            ts_input_name (str): name of input time series
            testing (bool): If running unit test or not. Defaults to False.

        Returns:
            (dict): jsonified version of original signal for last self.data['backfill_period'] days
            (list): dates to backfill data
        """
        client = self.client
        data = self.data
        ts_input = data["ts_input"][ts_input_name]
        orig_extid = ts_input["orig_extid"]

        # ----------------
        #now = pd.Timestamp.now() #datetime(2023, 11, 14, 16, 30)  # provided in local time
        # ----------------

        end_date = data["end_time"]

        start_date, _ = self.backfill_period_start(end_date, data, False)

        backfill_dates = []

        # Search through prev X=data["backfill_period"] time period of original time series for backfilling
        ts_orig_all = client.time_series.data.retrieve(external_id=orig_extid,
                                                    aggregates="average",
                                                    granularity=f"{data['granularity']}s",
                                                    start=start_date,
                                                    end=pd.to_datetime(
                                                        end_date),
                                                    limit=-1,
                                                    ).to_pandas()

        ts_orig_all = ts_orig_all.rename(
            columns={orig_extid + "|average": ts_input_name})

        my_func = client.functions.retrieve(external_id=data["function_name"])
        scheduled_calls = data["scheduled_calls"]

        backfill_date, backfill_hour = self.backfill_period_start(end_date, data, False)

        backfill_day = backfill_date.day
        backfill_month = backfill_date.month
        backfill_year = backfill_date.year

        SEC_SINCE_EPOCH = datetime(1970, 1, 1, 0, 0)
        start_time = datetime(backfill_year, backfill_month, backfill_day,
                            backfill_hour, data["backfill_min_start"])  # pd.Timestamp.now().hour-1, data["backfill_min_start"])
        start_time = (start_time - SEC_SINCE_EPOCH).total_seconds() * 1000  # convert to local time

        end_time = datetime(backfill_year, backfill_month, backfill_day,
                            backfill_hour, data["backfill_min_end"])
        end_time = (end_time - SEC_SINCE_EPOCH).total_seconds() * 1000

        try:
            mask_start = scheduled_calls["scheduled_time"] >= start_time
            mask_end = scheduled_calls["scheduled_time"] < end_time
            last_backfill_id = scheduled_calls[mask_start & mask_end]["id"].iloc[0]
        except:  # No scheduled call from yesterday --> nothing to compare with to do backfilling!
            print(
                f"Input {ts_input_name}: No schedule from yesterday. Can't backfill. Returning original signal from last {data['backfill_period']} days.")
            orig_as_dict = ast.literal_eval(ts_orig_all[ts_input_name].to_json())
            return orig_as_dict, backfill_dates

        last_backfill_call = my_func.retrieve_call(id=last_backfill_id)
        print(
            f"Retrieving scheduled call from yesterday with id {last_backfill_id}. Backfilling time series for last {data['backfill_period']} days ...")

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

        # if testing:
        #     return ts_orig_all, yesterday_df, backfill_dates, \
        #             num_dates_old, num_dates_new
        # return recent original signal
        orig_as_dict = ast.literal_eval(ts_orig_all[ts_input_name].to_json())
        return orig_as_dict, backfill_dates


    def backfill_period_start(self, end_date, data, get_previous_schedule=False):
        """Utility function for getting start of backfilling period,
        taking into account aggregates (if relevant).

        Args:
            end_date (datetime): date to run backfilling up to (typically pd.Timestamp.now)
            data (dict): input parameters for deployment of cognite function
            get_previous_schedule (bool): true if we want to retrieve previous schedule running backfilling
        Returns:
            backfill_date (datetime): date marking start of backfilling
            backfill_hour (int): hour of day at which to run backfilling
        """
        from dateutil.relativedelta import relativedelta

        backfill_hour = data["backfill_hour"]#backfill_date.hour

        if get_previous_schedule:
            backfill_period = 1
        else:
            backfill_period = data["backfill_period"]

        if data["testing"] and get_previous_schedule: # when testing backfilling, we only move some minutes back in time, not an entire day
            backfill_date = (end_date - timedelta(minutes=10))

        elif "period" in data["aggregate"] and "type" in data["aggregate"]:
            # perform backfilling for each aggregated period
            if data["aggregate"]["period"] == "year":
                backfill_date = (end_date - relativedelta(year=backfill_period))
            elif data["aggregate"]["period"] == "month":
                backfill_date = (end_date - relativedelta(month=backfill_period))
            elif data["aggregate"]["period"] == "day":
                backfill_date = (end_date - timedelta(days=backfill_period))
            elif data["aggregate"]["period"] == "hour":
                backfill_date = (end_date - timedelta(hours=backfill_period))
                backfill_hour = backfill_date.hour

        else:
            backfill_date = (end_date - timedelta(days=backfill_period)) # If not aggregates, backfill one day by default

        return backfill_date, backfill_hour


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv("../authentication-ids.env")
    print(os.getenv("CLIENT_ID"))