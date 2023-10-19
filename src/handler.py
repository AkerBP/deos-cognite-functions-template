from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cognite.client.data_classes import TimeSeries
from statsmodels.nonparametric.smoothers_lowess import lowess
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
    tank_volume = data['tank_volume']
    derivative_value_excl = data['derivative_value_excl']

    if data['cdf_env'] not in ["dev", "test", "prod"]:
        # end_date = datetime(year=2022, month=6, day=26, hour=1,
        #                     minute=0, second=0)
        data['tot_days'] = 10
        data['tot_minutes'] = 0
        ts_input_name = "PI-70445:X.Value"
        ts_output_name = "PI-70445:X.CDF.AVG.LeakRate"
        data["ts_input_name"] = ts_input_name
        data["ts_output_name"] = ts_output_name
        # only if new time series already uploaded
        end_date = pd.Timestamp.now() - pd.Timedelta(days=449)
        start_date = end_date - \
            timedelta(days=data['tot_days'], minutes=data['tot_minutes'])
    else:
        ts_input_name = data['ts_input_name']
        ts_output_name = data["ts_output_name"]
        end_date = pd.Timestamp.now()
        # from the start (00:00:00) of end_date
        start_date = pd.to_datetime(end_date.date())

    data["start_date"] = start_date
    data["end_date"] = end_date

    ts_orig = client.time_series.list(
        name=ts_input_name).to_pandas()  # original time series (vol percentage)
    ts_orig_extid = ts_orig.external_id[0]

    ts_leak = client.time_series.list(
        name=ts_output_name).to_pandas()  # transformed time series (leakage)
    # Check if transformed time series already exists
    data["ts_exists"] = not ts_leak.empty

    # Check for backfilling - run during last schedule of the day!
    # first_date_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
    #                                                    aggregates="average",
    #                                                    granularity="1m",
    #                                                    limit=1).to_pandas().index[0]
    # first_date_orig = pd.to_datetime(first_date_orig).date()
    # last_date_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
    #                                                   aggregates="average",
    #                                                   granularity="1m",
    #                                                   limit=1).to_pandas().index[-1]
    # last_date_orig = pd.to_datetime(last_date_orig).date()

    orig_signal = str({ts_output_name: json.dumps(None)})
    if end_date.hour == 15 and end_date.minute >= 0 and data["ts_exists"]:
        orig_signal = check_backfilling(client, ts_orig_extid, data)

    # STEP 1: Load time series from name and aggregate
    df = retrieve_orig_ts(client, ts_orig_extid, data)

    # STEP 2: Filter signal
    plot_filter = False
    df = filter_ts(df, plot_filter, data)

    # STEP 3: Create new time series, if not already exists
    if not data["ts_exists"]:
        print("Output time series does not exist. Creating ...")
        # client.time_series.delete(external_id=ts_output_name)
        client.time_series.create(TimeSeries(
            name=ts_output_name, external_id=ts_output_name, data_set_id=data['dataset_id']))

    # STEP 4: Calculate daily average drainage rate [% of tank vol / sec]
    daily_avg_drainage, df = calc_daily_avg_drain(df, data)

    # external ID is column name
    mean_df = pd.DataFrame({ts_output_name: daily_avg_drainage})
    mean_df.index = pd.to_datetime(mean_df.index)

    new_df = pd.merge(df, mean_df, on="Date")
    new_df["draining_rate [L/min]"] = new_df["derivative_excl_filling"] * \
        tank_volume/100  # drainage rate per TIME STAMP (not avg per day)

    # STEP 5: Insert transformed signal for new time range
    client.time_series.data.insert_dataframe(mean_df)

    if not data["ts_exists"]:
        # Write num datapoints for each date to csv-file, for backfilling purpose
        # df_orig_dates = df.groupby("Date")[ts_input_name].count()
        # df_orig_dates.to_csv("num_data_dates_orig_ts.csv")
        orig_signal = df.copy()
        orig_signal = orig_signal[ts_input_name].to_json()

    # Store original signal (for backfilling)
    return orig_signal  # new_df[[ts_output_name]].to_json()


def retrieve_orig_ts(client, ts_orig_extid, data):
    ts_input_name = data["ts_input_name"]
    ts_output_name = data["ts_output_name"]

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


def filter_ts(df, plot_filter, data):
    ts_input_name = data["ts_input_name"]
    vol_perc = df[ts_input_name]

    if __name__ == "__main__" and not data["backfill"]:
        import time
        print("Running lowess smoothing ...")
        t_start = time.perf_counter()

    smooth = lowess(vol_perc, df['time_sec'], is_sorted=True,
                    frac=0.01, it=0, delta=0)

    if __name__ == "__main__" and not data["backfill"]:
        t_stop = time.perf_counter() - t_start
        print("... Finished!")
        print(f"Lowess smoothing took {round(t_stop, 1)} seconds.")
    df_smooth = pd.DataFrame(smooth, columns=["time_sec", "smooth"])

    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'time_stamp'})
    df = pd.merge(df, df_smooth, on='time_sec')
    df.set_index('time_stamp', drop=True, append=False,
                 inplace=True, verify_integrity=False)

    if plot_filter:
        df.plot(x="time_sec", y=[ts_input_name, "smooth"])
        plt.show()

    return df


def calc_daily_avg_drain(df, data):
    try:
        df["derivative"] = np.gradient(df['smooth'], df["time_sec"])
    except:
        raise IndexError(
            "No datapoints found for selected date range. Cannot compute drainage rate.")

    derivative_value_excl = data['derivative_value_excl']
    df['derivative_excl_filling'] = df["derivative"].apply(
        lambda x: 0 if x > derivative_value_excl or pd.isna(x) else x)  # not interested in large INLET fluxes

    df.reset_index(inplace=True)
    df.index = pd.to_datetime(df['time_stamp'])
    df['Date'] = df.index.date
    df["Date"] = pd.to_datetime(df["Date"])

    mean_drainage_day = df.groupby('Date')['derivative_excl_filling'].mean(
    )*data['tank_volume']/100  # avg drainage rate per DAY

    return mean_drainage_day, df


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

    # ----------------
    # Retrieve last 30-day period signal (get response from associated function call!)
    yesterday_df = get_yesterday_original_signal(client, data)

    if not yesterday_df.empty:  # empty if no scheduled call from yesterday

        num_dates_old = yesterday_df.groupby(yesterday_df.index).count()
        num_dates_old.index = pd.to_datetime(num_dates_old.index)
        # ----------------

        # 1. For each write, store number of data points in ORIGINAL signal for each date - store in file associated with dataset ID.
        # number of datapoints for each date
        num_dates_new = ts_orig_dates.groupby(ts_input_name)[
            ts_input_name].count()
        num_dates_new = pd.DataFrame(num_dates_new)

        num_dates_new.index.name = "Date"

        # Truncate original values to backfilled period
        num_dates_old = num_dates_old[num_dates_old.index >=
                                      num_dates_new.index[0]]

        missing_dates = num_dates_new[~num_dates_new.index.isin(
            num_dates_old.index)].index
        missing_dates = pd.DataFrame({ts_input_name:
                                      np.zeros(len(missing_dates), dtype=np.int32)}, index=missing_dates)
        print(yesterday_df)
        print("---")
        print(num_dates_new)
        print(f"missing: {missing_dates}")
        # New df with zero count for missing dates
        num_dates_old = pd.concat([num_dates_old, missing_dates]).sort_index()

        # print(f"num orig dates: {num_dates_old}")
        # print(f"num orig dates new: {num_dates_new}")
        # 2. If number of datapoints INCREASES for certain dates (= backfilled), redo transformations for these dates.
        backfill_dates = num_dates_new[num_dates_new[ts_input_name] >
                                       num_dates_old[ts_input_name]].index
        print(f"backfill_dates: {backfill_dates}")
        # 3. Redo transformations for modified dates
        for date in backfill_dates:
            start_date = pd.to_datetime(date)
            end_date = pd.to_datetime(date+timedelta(days=1))
            df_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                       aggregates="average",
                                                       granularity="1m",
                                                       start=start_date,
                                                       end=end_date
                                                       ).to_pandas()
            # NB: need to change time frame in data
            data["start_date"] = start_date
            data["end_date"] = end_date
            df_orig = retrieve_orig_ts(client, ts_orig_extid, data)
            df_orig = filter_ts(df_orig, False, data)

            daily_avg_drainage, df = calc_daily_avg_drain(df_orig, data)

            mean_df = pd.DataFrame({ts_output_name: daily_avg_drainage})
            mean_df.index = pd.to_datetime(mean_df.index)

            client.time_series.data.insert_dataframe(mean_df)

    # return recent original signal
    return ts_orig_all[[ts_input_name]].to_json()


def get_yesterday_original_signal(client, data):
    my_func = client.functions.retrieve(external_id=data["function_name"])
    my_schedule_id = client.functions.schedules.list(
        name=data["schedule_name"]).to_pandas().id[0]
    all_calls = my_func.list_calls(
        schedule_id=my_schedule_id, limit=-1).to_pandas()

    now = pd.Timestamp.now()
    start_time = datetime(now.year, now.month, now.day-1,
                          23, 45).timestamp() * 1000  # -1 to get previous day
    end_time = datetime(now.year, now.month, now.day, 0,
                        0).timestamp() * 1000  # *1000 -> millisec since epoch

    mask_start = all_calls["scheduled_time"] >= start_time
    mask_end = all_calls["scheduled_time"] < end_time

    try:
        last_backfill_id = all_calls[mask_start & mask_end]["id"].iloc[0]
    except:
        return pd.DataFrame()

    last_backfill_call = my_func.retrieve_call(id=last_backfill_id)

    output_dict = ast.literal_eval(last_backfill_call.get_response())[
        data["ts_output_name"]]

    start_30_day_period = pd.to_datetime(
        last_backfill_call.scheduled_time, unit="ms") - timedelta(days=30)
    end_30_day_period = pd.to_datetime(
        last_backfill_call.scheduled_time, unit="ms")

    output_df = pd.DataFrame.from_dict([output_dict]).T
    output_df.index = pd.to_datetime(
        output_df.index.astype(int), unit="ms").date  # astype(int)*1e7 for testing
    output_df.index.name = "Date"
    # Column created with standard value 0 ...
    output_df = output_df.rename(columns={0: data["ts_input_name"]})

    return output_df


if __name__ == '__main__':
    from initialize import initialize_client

    cdf_env = "dev"
    if cdf_env not in ["dev", "test", "prod"]:
        token = True
    else:
        token = False

    client = initialize_client(cdf_env, cache_token=token)

    ts_name = "VAL_11-LT-95034A:X.Value"
    ts_output_name = "VAL_11-LT-95034A:X.CDF.D.AVG.LeakValue"
    dataset_id = 1832663593546318  # 3197476513083188  #
    function_name = "avg-drainage-rate-new"
    schedule_name = "avg-drainage-rate-schedule-new"

    tank_volume = 1400
    derivative_value_excl = 0.002
    # start_date = datetime(2023, 3, 21, 1, 0, 0)

    data_dict = {'tot_days': 0, 'tot_minutes': 15,  # convert date to str to make it JSON serializable
                 'ts_input_name': ts_name, 'ts_output_name': ts_output_name,
                 'derivative_value_excl': derivative_value_excl, 'tank_volume': tank_volume,
                 # NB: change dataset id when going to dev/test/prod!
                 'cdf_env': cdf_env, 'dataset_id': dataset_id,
                 'backfill': False, 'backfill_period': 7,
                 'function_name': function_name, 'schedule_name': schedule_name}

    # out = get_yesterday_original_signal(client, data_dict)
    # print(out)
    # client.time_series.delete(external_id="PI-70445:X.CDF.AVG.LeakRate")
    new_df = handle(client, data_dict)
