import sys
import os
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from pytz import utc
from typing import Tuple
import pandas as pd
import numpy as np
import ast
import json
import logging
import re
from io import BytesIO
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

from cognite.client.data_classes import TimeSeries
from cognite.client._cognite_client import CogniteClient
from cognite.client.exceptions import CogniteAPIError

logger = logging.getLogger(__name__)

<<<<<<<< HEAD:examples/prepare_timeseries.py
from run_transformation import RunTransformations
========
from transform_timeseries import RunTransformations
>>>>>>>> 17f24d61bdea89a32f0272d860b4311b8a3c6673:cf_template/prepare_timeseries.py
from utilities import dataframe_to_bytes, get_external_id_from_name
from utilities import AGG_PERIOD


class PrepareTimeSeries:
    """Class to organize input time series and prepare output time series
    for transformations with Cognite Functions.
    """
    def __init__(self, ts_input_names: list, ts_output: dict,
                 client: CogniteClient, data_dicts: dict):
        """Provide client and data dictionary for deployment of Cognite Function

        Args:
            ts_input_names (list): names for input time series
            ts_output (dict): metadata for output time series
            client (CogniteClient): instantiated CogniteClient
            data_dicts (dict): data dictionary used for Cognite Function
        """
        self.client = client
        self.ts_input_names = ts_input_names
        self.ts_output = ts_output
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
            self.data["ts_output"] = {name:{"exists":False,
                                            "description":desc,
                                            "unit":u} for name, desc, u in zip(self.ts_output["names"],
                                                                                self.ts_output["description"],
                                                                                self.ts_output["unit"])}
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

        end_date = pd.Timestamp.now(tz="CET").floor("1s").tz_convert("UTC") #- timedelta(days=2) #TODO: SUBTRACTING ONE DAY ONLY FOR TESTING !
        start_date = end_date - timedelta(minutes=int(self.data["cron_interval_min"]))

        self.data["start_time"] = start_date
        self.data["end_time"] = end_date

        for ts_out_name in ts_outputs.keys():
            ts_leak = client.time_series.list(
                    name=ts_out_name).to_pandas()  # transformed time series (leakage)
            # Check if transformed time series already exists
            ts_outputs[ts_out_name]["exists"] = not ts_leak.empty

        # Create new time series, if not already exists
        self.create_timeseries()

        self.data["ts_input_data"] = {ts_name: [] for ts_name in ts_inputs.keys()} # stores original signal only for current date

        ts_output_names = list(ts_outputs.keys())
        if len(ts_inputs.keys()) > len(ts_output_names): # multiple input time series used to compute one output time series
            ts_output_names = ts_output_names*len(ts_inputs.keys())

        # Find external ids
        for ts_in, ts_out in zip(ts_inputs.keys(), ts_output_names):
            if not ts_inputs[ts_in]["exists"]:
                continue # input not in CDF, provided separately, skip to next input

            try:
                ts_inputs[ts_in]["orig_extid"] = get_external_id_from_name(client, ts_in)
            except:
                raise KeyError(f"Input time series {ts_in} does not exist.")

        all_orig_extid = [ts_inputs[ts_in]["orig_extid"] for ts_in in ts_inputs.keys() if ts_inputs[ts_in]["exists"]]

        # Find latest datetime for which all inputs have a value for:
        df_orig = pd.DataFrame()
        nonan = False
        num_days = 1
        while not nonan:
            start_date = end_date - timedelta(days=num_days)
            df_orig = client.time_series.data.retrieve_dataframe_in_tz(external_id=all_orig_extid,
                                                                    granularity=self.data["granularity"],
                                                                    aggregates="average",
                                                                    start=pd.to_datetime(start_date, utc=True),
                                                                    end=pd.to_datetime(end_date, utc=True))
            num_days += 1
            nonan = df_orig.notna().all(axis=1).any()
            if not nonan:
                print(f"No/nan data found for period [{start_date}, {end_date}]. Rewinding 1 day ...")

        latest_end = df_orig.dropna().index.max()
        latest_start = latest_end - timedelta(minutes=int(self.data["cron_interval_min"]))

        self.data["ts_input"] = ts_inputs # update dictionary

        # MAIN RETRIEVAL LOOP
        for ts_in, ts_out in zip(ts_inputs.keys(), ts_output_names):

            data_in = ts_inputs[ts_in]
            data_out = ts_outputs[ts_out]

            if not data_in["exists"]:
                self.data["ts_input_data"][ts_in] = float(ts_in)
                continue # input not in CDF, provided separately, skip to next input

            # STEP 1: Identify backfill candidates
            backfill_periods = []

            if end_date.hour == self.data["backfill_hour"] and \
               end_date.minute >= self.data["backfill_min_start"] and \
                end_date.minute < self.data["backfill_min_end"] \
                and data_out["exists"]: # TODO: Change to 23 hours and 45 minutes.
                    backfill_periods = self.check_backfilling(ts_in)

            # STEP 2: Perform backfilling on dates with discrepancies in datapoints
            for period in backfill_periods:
                if "aggregate" in self.data["optional"]:
                    # If aggregates, for each "changed" period, run backfilling over a single aggregation period
                    start_time, end_time = self.get_aggregated_start_end_time(period)

                    self.data["start_time"] = start_time
                    self.data["end_time"] = end_time

                else:
                    self.data["start_time"] = pd.to_datetime(period) # period assumed to be a date
                    self.data["end_time"] = pd.to_datetime(period+timedelta(days=1))

                self.run_backfilling(calc_func)

            # STEP 3: After backfilling, retrieve original signal for intended transformation period
            # Start and end times may have been modified by backfilling. Retrieve original times.
            self.data["start_time"] = latest_start#start_date
            self.data["end_time"] = latest_end#end_date

            df_orig = self.retrieve_orig_ts(ts_in, ts_out, latest_start, latest_end)

            if "aggregate" in self.data["optional"]: # append data part of aggregation period prior to schedule start to data part of shcedule
                start_time_schedule = self.data["start_time"]
                start_time_aggregate, _ = self.get_aggregated_start_end_time() # only need start time, not end time

                # If schedule starts exactly at aggregated period - nothing to concatenate with scheduled period
                if start_time_aggregate < start_time_schedule:
                    orig_extid = self.data["ts_input"][ts_in]["orig_extid"]
                    # Retrieve all datapoints from aggregated period NOT part of current schedule
                    df_orig_prev = client.time_series.data.retrieve_dataframe_in_tz(external_id=orig_extid,
                                                                                    aggregates="average",
                                                                                    granularity=self.data["granularity"],
                                                                                    start=start_time_aggregate,
                                                                                    end=start_time_schedule)

                    df_orig_prev = df_orig_prev.rename(columns={orig_extid + "|average": ts_in})
                    # df_orig_prev = df_orig_prev.iloc[:-1] # Omit last element as this is first element in df_orig

                    df_orig_prev.index = pd.to_datetime(df_orig_prev.index, utc=True)
                    df_orig.index = pd.to_datetime(df_orig.index, utc=True)

                    df_orig = pd.concat([df_orig_prev, df_orig]) # join scheduled period with remaining aggregated period
                    # df_orig = df_orig.apply(lambda row: getattr(row, self.data["aggregate"]["type"])(), axis=1)

                    df_orig = pd.DataFrame(df_orig, columns=[ts_in])

            # STEP 4: Store time series signal in data dictionary
            self.data["ts_input_data"][ts_in] = df_orig[ts_in]

        return self.data


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
            if not ts_output[ts_out_name]["exists"]:
                print(f"Output time series {ts_out_name} does not exist. Creating ...")
                client.time_series.create(TimeSeries(
                                                    name=ts_out_name,
                                                    external_id=ts_out_name,
                                                    data_set_id=data['dataset_id'],
                                                    asset_id=asset_id,
                                                    unit=ts_output[ts_out_name]["unit"],
                                                    description=ts_output[ts_out_name]["description"]))

        return asset_ids


    def retrieve_orig_ts(self, ts_in: str, ts_out: str, latest_start=None, latest_end=None) -> pd.DataFrame:
        """Return input time series over given date range, averaged over a given granularity.

        Args:
            ts_in (str): name of input time series
            ts_out (str): name of output time series (to be produced)
            latest_start (datetime): latest possible start time where all inputs have values
            latest_end (datetime): latest possible end time where all inputs have values

        Returns:
            pd.DataFrame: input time series
        """
        client = self.client
        data = self.data
        data_in = self.data["ts_input"][ts_in]
        data_out = self.data["ts_output"][ts_out]
        orig_extid = data_in["orig_extid"]

        if latest_start is not None and latest_end is not None:
            start_date = latest_start
            end_date = latest_end
        else:
            start_date = data["start_time"]
            end_date = data["end_time"]

        try:
            # If no data in output time series, run cognite function from first available date of original time series until date with last updated datapoint
            if not data_out["exists"]:
                if "historic_start_time" in data["optional"]: # retrieve original signal from specified date
                    new_start = data["optional"]["historic_start_time"]
                    start_date = datetime(new_start["year"], new_start["month"], new_start["day"],
                                          tzinfo=timezone.utc) # Set timezone to UTC by default
                else: # retrieve signal from first historic value
                    start_date = client.time_series.data.retrieve_dataframe(external_id=orig_extid,
                                                                            aggregates="average",
                                                                            granularity=data["granularity"]).index[0]
                    start_date = utc.localize(start_date)

                self.data["start_time"] = start_date

            df = client.time_series.data.retrieve_dataframe_in_tz(external_id=orig_extid,
                                                                    aggregates="average",
                                                                    granularity=data["granularity"],
                                                                    start=pd.to_datetime(
                                                                        start_date, utc=True),
                                                                    end=pd.to_datetime(
                                                                        end_date, utc=True),
                                                                    )

            if df.empty:
                print(f"No data for time series '{ts_in}' for interval: [{start_date}, {end_date}]. Rewinding to look for latest datapoint ...")

            while df.empty:
                try:
                    end_date_new = client.time_series.data.retrieve_dataframe_in_tz(external_id=orig_extid,
                                                                                aggregates="average",
                                                                                granularity=data["granularity"],
                                                                                start=pd.to_datetime(end_date - timedelta(days=1), utc=True), # Go back 1 day to search for latest datapoint
                                                                                end=pd.to_datetime(end_date, utc=True)).index[-1]
                except:
                    end_date = end_date - timedelta(days=1)
                    print(f"Checking period [{end_date - timedelta(days=1)}, {end_date}] ...")
                    continue

                start_date = end_date_new - timedelta(minutes=int(data["cron_interval_min"]))

                self.data["start_time"] = start_date
                self.data["end_time"] = end_date_new

                df = client.time_series.data.retrieve_dataframe_in_tz(external_id=orig_extid,
                                                                    aggregates="average",
                                                                    granularity=data["granularity"],
                                                                    start=pd.to_datetime(
                                                                        start_date, utc=True),
                                                                    end=pd.to_datetime(
                                                                        end_date_new, utc=True),
                                                                    )
                print(f"Found latest datapoint: {end_date_new}. New period: [{start_date}, {end_date_new}].")

            df = df.rename(columns={orig_extid + "|average": ts_in})
            return df

        except CogniteAPIError as error:
            msg = "Unable to read time series. Deployment key is missing capability 'timeseries:READ'"
            logger.error(msg)
            raise CogniteAPIError(msg, error.code, error.x_request_id) from None


    def align_time_series(self, ts_all: list) -> list:
        """Convert all inputs to dataframes,
        and align input time series to cover same data range.

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

        time_index = pd.date_range(start=latest_start_date, end=earliest_end_date, freq=self.data["granularity"][:-1]+AGG_PERIOD[self.data["granularity"][-1]])

        ts_all = [0]*len(ts_all)
        for i in range(len(ts_df)):
            ts_df[i][1] = ts_df[i][1].reindex(time_index, copy=False) # missing internal dates are filled with nan
            ts_all[ts_df[i][0]] = ts_df[i][1]

        for i in range(len(ts_scalars)):
            ts_scalars[i][1] = pd.DataFrame(ts_scalars[i][1]*np.ones(len(ts_df[0][1])), index=ts_df[0][1].index, columns=[f"scalar{i}"]) # can simply choose one of the dfs, they have the same index at this point anyway
            ts_all[ts_scalars[i][0]] = ts_scalars[i][1]

        ts_all = pd.concat(ts_all, axis=1) # concatenate along date

        return ts_all

    def get_ts_df(self) -> list:
        """List all input time series as dataframes (if time series from CDF) or scalars (if single scalar provided)

        Returns:
            (list): list of formatted time series
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
        print(f"Checking signal {ts_input_name} for periods to backfill ...")
        client = self.client
        data = self.data
        ts_input = data["ts_input"][ts_input_name]
        orig_extid = ts_input["orig_extid"]

        end_date = data["end_time"]

        # Get start of period to backfill for
        # start_date, _ = self.backfill_period_start(end_date, False)

        start_date = end_date - relativedelta(months=1)
        backfill_periods = []

        # Search through prev X=data["backfill_period"] time period of original time series for backfilling
        ts_orig_all = client.time_series.data.retrieve_dataframe_in_tz(external_id=orig_extid,
                                                                    aggregates="average",
                                                                    granularity=data["granularity"],
                                                                    start=start_date,
                                                                    end=pd.to_datetime(
                                                                        end_date, utc=True))

        ts_orig_all = ts_orig_all.rename(
            columns={orig_extid + "|average": ts_input_name})
        # Excel does not support tz-aware datetimes. Convert to string to preserve timezone info
        ts_orig_all.index = ts_orig_all.index.astype(str)
        # CURRENT period's signal spanning backfilling period
        ts_orig_current = ts_orig_all.copy()
        ts_orig_current.index = pd.to_datetime(ts_orig_current.index) # restore tz-aware datetime after writing

        if testing:
            file_name = "VAL_17-FI-9101-286VALUE_COPY"
        else:
            file_name = re.sub(r'[\\/:*?"<>|]', "", ts_input_name)

        # Convert time series to bytes object
        current_bytes = dataframe_to_bytes(ts_orig_all)

        try:
            previous_bytes = client.files.download_bytes(external_id=file_name) # Data from previous backfilling period in bytes
        except: # No data File exist in CDF yet - upload once and for all
            print(f"No historic data stored yet. Can't perform backfilling. Uploading data to CDF File {file_name}.xlsx ...")
            # ts_orig_all.to_excel(f"{file_name}.xlsx", index=True) # TODO WORKS ???
            client.files.upload_bytes(content=BytesIO(current_bytes), external_id=file_name, name=f"{file_name}.xlsx",
                                data_set_id=data["dataset_id"], mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") # mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            # client.files.upload(f"{file_name}.xlsx", external_id=file_name, name=f"{file_name}.xlsx",
            #                     data_set_id=data["dataset_id"], mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") # mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            return backfill_periods

        if "aggregate" in data["optional"]:
            aggregate = data["optional"]["aggregate"]
            sampling_period = AGG_PERIOD[aggregate["period"]]
        else:
            sampling_period = "D" # if no aggregates, we run backfilling for dates by default

        # PREVIOUS period's signal spanning backfilling period
        previous_df = pd.read_excel(BytesIO(previous_bytes), index_col=0)
        previous_df.index = pd.to_datetime(previous_df.index) # restore tz-aware datetime

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
        print(f"Periods with NEW data: {increased_periods.values}")

        decreased_periods = num_periods_current[num_periods_current["Datapoints"] <
                                        num_periods_previous["Datapoints"]].index
        print(f"Periods with DELETED data: {decreased_periods.values}")
        backfill_periods = increased_periods.union(decreased_periods, sort=True)

        # Overwriting file not possible, so need to remove first
        client.files.delete(external_id=file_name)
        client.files.upload_bytes(content=BytesIO(current_bytes), external_id=file_name, name=f"{file_name}.xlsx",
                                data_set_id=data["dataset_id"], mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        return backfill_periods

    def run_backfilling(self):
        """Runs backfilling for time interval given by
        self.data["start_date"] and self.data["end_date"].
        """
        ts_input = self.data["ts_input"]
        ts_output = self.data["ts_output"]
        calc_func = self.data["calculation_function"]

        for ts_in, ts_out in zip(ts_input.keys(), ts_output.keys()):
            if not ts_input[ts_in]["exists"]:
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
        agg = self.data["optional"]["aggregate"]

        if start_backfill_time is not None:
            start_time_exact = start_backfill_time
            agg_period = {agg["period"]+"s": 1}

            """start_backfill_time = min(start_backfill_period)
            agg_delta = relativedelta(**agg_period) # delta of aggregation period
            end_backfill_time = max(backfill_period)+agg_delta # add aggregation period delta to include last backfilling period
            """
            agg_delta = relativedelta(**agg_period)
            end_time_exact = start_time_exact + agg_delta

        else:
            start_time_exact = self.data["start_time"]
            end_time_exact = self.data["end_time"]

        start_time = datetime(start_time_exact.year, 1, 1)
        end_time = datetime(end_time_exact.year, 1, 1)

        if agg["period"] == "year":
            pass
        elif agg["period"] == "month":
            start_time = start_time.replace(month=start_time_exact.month)
            end_time = end_time.replace(month=end_time_exact.month)
        elif agg["period"] == "day":
            start_time = start_time.replace(month=start_time_exact.month,
                                            day=start_time_exact.day)
            end_time = end_time.replace(month=end_time_exact.month,
                                        day=end_time_exact.day)
        elif agg["period"] == "hour":
            start_time = start_time.replace(month=start_time_exact.month,
                                            day=start_time_exact.day,
                                            hour=start_time_exact.hour)
            end_time = end_time.replace(month=end_time_exact.month,
                                        day=end_time_exact.day,
                                        hour=end_time_exact.hour)
        elif agg["period"] == "minute":
            start_time = start_time.replace(month=start_time_exact.month,
                                            day=start_time_exact.day,
                                            hour=start_time_exact.hour,
                                            minute=start_time_exact.minute)
            end_time = end_time.replace(month=end_time_exact.month,
                                        day=end_time_exact.day,
                                        hour=end_time_exact.hour,
                                        minute=end_time_exact.minute)

        else:
            raise NotImplementedError(f"Backfilling not implemented for aggregates of period {self.data['aggregate']['period']}. Supported periods are: year, month, day, hour, minute.")

        start_time = start_time.replace(tzinfo=self.data["start_time"].tzinfo)
        end_time = end_time.replace(tzinfo=self.data["end_time"].tzinfo)

        end_time = min(end_time, self.data["end_time"]) # cap end time at current timestamp

        return pd.to_datetime(start_time), pd.to_datetime(end_time)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv("../authentication-ids.env")
    print(os.getenv("CLIENT_ID"))