import sys
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
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
        end_date = pd.Timestamp.now(tz="CET").floor("1s").tz_convert("UTC") #- timedelta(days=2) #TODO: SUBTRACTING ONE DAY ONLY FOR TESTING !
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
            backfill_periods = []

            if self.data["testing"]:
                if int(str(end_date.minute)[-1]) == 0:
                    ts_input_backfill, backfill_periods = self.check_backfilling(ts_in, testing=True)
            elif end_date.hour == self.data["backfill_hour"] and \
                end_date.minute >= self.data["backfill_min_start"] and \
                end_date.minute < self.data["backfill_min_end"] \
                and data_out["exists"]: # TODO: Change to 23 hours and 45 minutes.
                    ts_input_backfill, backfill_periods = self.check_backfilling(ts_in)

            self.data["ts_input_backfill"][ts_in] = ts_input_backfill

            # STEP 4: Perform backfilling on dates with discrepancies in datapoints
            for period in backfill_periods:
                if "period" in self.data["aggregate"] and "type" in self.data["aggregate"]:
                    # If aggregates, for each "changed" period, run backfilling over a single aggregation period
                    start_time, end_time = self.get_aggregated_start_end_time(period)

                    self.data["start_time"] = start_time
                    self.data["end_time"] = end_time

                else:
                    self.data["start_time"] = pd.to_datetime(period) # period assumed to be a date
                    self.data["end_time"] = pd.to_datetime(period+timedelta(days=1))

                self.run_backfilling(ts_inputs, ts_output_names, calc_func)

            """if "period" in self.data["aggregate"] and "type" in self.data["aggregate"] \
                and len(backfill_periods) > 0:
                # If aggregates, run backfilling for last X aggregated periods
                start_time, end_time = self.get_aggregated_start_end_time(backfill_periods)

                self.data["start_time"] = start_time
                self.data["end_time"] = end_time

                self.run_backfilling(ts_inputs, ts_output_names, calc_func)

            else: # If no aggregates, run backfilling on each date separately
                for date in backfill_periods:
                    self.data["start_time"] = pd.to_datetime(date)
                    self.data["end_time"] = pd.to_datetime(date+timedelta(days=1))

                    self.run_backfilling(ts_inputs, ts_output_names, calc_func)
            """
            # STEP 5: After backfilling, retrieve original signal for intended transformation period
            if not ts_inputs[ts_in]["exists"]:
                self.data["ts_input_data"][ts_in] = float(ts_in)
                continue # input not from CDF, skip to next input

            # Start and end times may have been modified by backfilling. Retrieve original times.
            self.data["start_time"] = start_date
            self.data["end_time"] = end_date

            df_orig = self.retrieve_orig_ts(ts_in, ts_out)

            if "period" in self.data["aggregate"] and "type" in self.data["aggregate"]:
                start_time_schedule = self.data["start_time"]
                start_time_aggregate, _ = self.get_aggregated_start_end_time() # only need start time, not end time

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

                df_orig = pd.concat([df_orig_prev, df_orig]) # join scheduled period with remaining aggregated period
                # df_orig = df_orig.apply(lambda row: getattr(row, self.data["aggregate"]["type"])(), axis=1)

                df_orig = pd.DataFrame(df_orig, columns=[ts_in])

            self.data["ts_input_data"][ts_in] = df_orig[ts_in]
            if self.data["ts_input_backfill"][ts_in] == "null": # if no backfilling for input signal, return original signal
                self.data["ts_input_backfill"][ts_in] = ast.literal_eval(df_orig[ts_in].to_json())

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

        end_date = data["end_time"]

        # Get start of period to backfill for
        start_date, _ = self.backfill_period_start(end_date, False)

        backfill_periods = []

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

        # Get date and hour of when most recent backfilling occured
        backfill_date, backfill_hour = self.backfill_period_start(end_date, True)

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
            return orig_as_dict, backfill_periods

        last_backfill_call = my_func.retrieve_call(id=last_backfill_id)
        print(
            f"Retrieving scheduled call from yesterday with id {last_backfill_id}. Backfilling time series for last {data['backfill_period']} days ...")

        output_dict = ast.literal_eval(last_backfill_call.get_response())[
            ts_input_name]

        output_df = pd.DataFrame.from_dict([output_dict]).T
        output_df.index = pd.to_datetime(
            output_df.index.astype(np.int64), unit="ms")
        # output_df["Date"] = output_df.index.date  # astype(int)*1e7 for testing

        agg_period = {"second":"S", "minute":"T", "hour":"H", "day":"D", "month":"M", "year":"Y"}
        aggregate = data["aggregate"]
        if "period" in aggregate:
            sampling_period = agg_period[aggregate["period"]]
        else:
            sampling_period = "D" # if no aggregates, we run backfilling for dates by default

        # PREVIOUS period's signal spanning backfilling period
        previous_df = output_df.rename(columns={0: ts_input_name})
        # CURRENT period's signal spanning backfilling period
        ts_orig_current = ts_orig_all.copy()

        if not previous_df.empty:  # empty if no scheduled call from previous period
            print("Dates from previous signal: ", previous_df.index.values)
            print("Dates from current signal: ", ts_orig_current.index.values)
            # 1. Only include overlapping parts of signal from current and previous periods
            backfill_date_start = ts_orig_current.index[0]
            backfill_date_stop = previous_df.index[-1]
            previous_df = previous_df[previous_df.index >= backfill_date_start]
            ts_orig_current = ts_orig_current[ts_orig_current.index <= backfill_date_stop]

            # 2. Store number of data points in ORIGINAL signal for each aggregated period, for both previous period (num_periods_previous) and current period (num_periods_current)
            num_periods_previous = previous_df.resample(sampling_period).count()
            # num_periods_previous = previous_df.groupby(previous_df["Date"]).count()
            num_periods_previous.index = pd.to_datetime(num_periods_previous.index)
            num_periods_previous = num_periods_previous.rename(columns={ts_input_name: "Datapoints"})

            num_periods_current = ts_orig_current.resample(sampling_period).count()
            # num_periods_current = ts_orig_current.groupby(ts_orig_current["Date"]).count()
            num_periods_current = num_periods_current.rename(columns={ts_input_name: "Datapoints"})

            missing_periods = num_periods_current[~num_periods_current.index.isin(
                num_periods_previous.index)].index
            missing_periods = pd.DataFrame({"Datapoints":
                                        np.zeros(len(missing_periods), dtype=np.int32)}, index=missing_periods)

            # New df with zero count for missing periods
            num_periods_previous = pd.concat([num_periods_previous, missing_periods]).sort_index()

            # 3. Only backfill if num datapoints have INCREASED or DECREASED for any periods
            increased_periods = num_periods_current[num_periods_current["Datapoints"] >
                                            num_periods_previous["Datapoints"]].index
            print(f"Backfilling periods with NEW data: {increased_periods.values}")

            decreased_periods = num_periods_current[num_periods_current["Datapoints"] <
                                            num_periods_previous["Datapoints"]].index
            print(f"Backfilling periods with DELETED data: {decreased_periods.values}")
            backfill_periods = increased_periods.union(decreased_periods, sort=True)

        orig_as_dict = ast.literal_eval(ts_orig_all[ts_input_name].to_json())

        return orig_as_dict, backfill_periods

    def run_backfilling(self, ts_inputs, ts_output_names, calc_func):
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

        self.client.time_series.data.insert_dataframe(df_out)
        return

    def get_aggregated_start_end_time(self, start_backfill_time=None) -> Tuple[datetime, datetime]:
        """Return start and end time of aggregated period.
        In case of backfilling, the end time is one delta period
        ahead of last backfilled aggregated period.

        Args:
            start_backfill_time (datetime64, optional): Datetime period to start backfilling from. Defaults to None.

        Raises:
            NotImplementedError: In case of invalid aggregation period.

        Returns:
            Tuple[datetime, datetime]: Start time and end time of aggregation period.
        """
        if start_backfill_time is not None:
            agg_period = {self.data["aggregate"]["period"]+"s": 1}

            """start_backfill_time = min(start_backfill_period)
            agg_delta = relativedelta(**agg_period) # delta of aggregation period
            end_backfill_time = max(backfill_period)+agg_delta # add aggregation period delta to include last backfilling period
            """
            agg_delta = relativedelta(**agg_period)
            end_backfill_time = start_backfill_time + agg_delta

        else:
            start_backfill_time = self.data["start_time"]
            end_backfill_time = self.data["end_time"]

        start_time = datetime(start_backfill_time.year, 1, 1)
        end_time = datetime(end_backfill_time.year, 1, 1)

        if self.data["aggregate"]["period"] == "year":
            pass
        elif self.data["aggregate"]["period"] == "month":
            start_time = start_time.replace(month=start_backfill_time.month)
            end_time = end_time.replace(month=end_backfill_time.month)
        elif self.data["aggregate"]["period"] == "day":
            start_time = start_time.replace(month=start_backfill_time.month,
                                            day=start_backfill_time.day)
            end_time = end_time.replace(month=end_backfill_time.month,
                                        day=end_backfill_time.day)
        elif self.data["aggregate"]["period"] == "hour":
            start_time = start_time.replace(month=start_backfill_time.month,
                                            day=start_backfill_time.day,
                                            hour=start_backfill_time.hour)
            end_time = end_time.replace(month=end_backfill_time.month,
                                        day=end_backfill_time.day,
                                        hour=end_backfill_time.hour)
        elif self.data["aggregate"]["period"] == "minute":
            start_time = start_time.replace(month=start_backfill_time.month,
                                            day=start_backfill_time.day,
                                            hour=start_backfill_time.hour,
                                            minute=start_backfill_time.minute)
            end_time = end_time.replace(month=end_backfill_time.month,
                                        day=end_backfill_time.day,
                                        hour=end_backfill_time.hour,
                                        minute=end_backfill_time.minute)

        else:
            raise NotImplementedError(f"Backfilling not implemented for aggregates of period {self.data['aggregate']['period']}. Supported periods are: year, month, day, hour, minute.")

        start_time = start_time.replace(tzinfo=self.data["start_time"].tzinfo)
        end_time = end_time.replace(tzinfo=self.data["end_time"].tzinfo)

        end_time = min(end_time, self.data["end_time"]) # cap end time at current timestamp

        return start_time, end_time


    def backfill_period_start(self, end_date, get_previous_schedule=False):
        """Utility function for getting start of backfilling period,
        taking into account aggregates (if relevant).

        Args:
            end_date (datetime): date to run backfilling up to (typically pd.Timestamp.now)
            get_previous_schedule (bool): true if we want to retrieve previous schedule running backfilling
        Returns:
            backfill_date (datetime): date marking start of backfilling
            backfill_hour (int): hour of day at which to run backfilling
        """
        data = self.data

        backfill_hour = data["backfill_hour"]#backfill_date.hour

        if get_previous_schedule:
            backfill_period = 1
        else:
            backfill_period = data["backfill_period"]

        if data["testing"] and get_previous_schedule: # when testing backfilling, we only move some minutes back in time, not an entire day
            backfill_date = (end_date - timedelta(minutes=10))

        elif "period" in data["aggregate"] and "type" in data["aggregate"]:
            # perform backfilling for each aggregated period
            period = {data["aggregate"]["period"]+"s": backfill_period} # because relativedelta parameters end with s (e.g., hours not hour)

            backfill_date = (end_date - relativedelta(**period))

            if data["aggregate"]["period"] == "hour":
                backfill_hour = backfill_date.hour

        else:
            backfill_date = (end_date - timedelta(days=backfill_period)) # If not aggregates, backfill one day by default

        return backfill_date, backfill_hour


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv("../authentication-ids.env")
    print(os.getenv("CLIENT_ID"))