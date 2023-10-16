from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cognite.client.data_classes import TimeSeries
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats


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

    if data['test_run']:
        # Run UaT of initial write, continuous write, etc ...
        pass

    elif data['cdf_env'] not in ["dev", "test", "prod"]:
        # end_date = datetime(year=2022, month=6, day=26, hour=1,
        #                     minute=0, second=0)
        data['tot_days'] = 3
        data['tot_minutes'] = 0
        ts_input_name = "PI-70445:X.Value"
        # only if new time series already uploaded
        end_date = pd.Timestamp.now() - pd.Timedelta(days=449)
        start_date = end_date - \
            timedelta(days=data['tot_days'], minutes=data['tot_minutes'])
    else:
        ts_input_name = data['ts_input_name']
        end_date = pd.Timestamp.now()
        # from the start (00:00:00) of end_date
        start_date = pd.to_datetime(end_date.date())

    ts_output_name = data['ts_output_name']
    dataset_id = data['dataset_id']

    # STEP 1: Load time series from name and aggregate
    ts_orig = client.time_series.list(
        name=ts_input_name).to_pandas()  # original time series (vol percentage)
    ts_orig_extid = ts_orig.external_id[0]

    ts_leak = client.time_series.list(
        name=ts_output_name).to_pandas()  # transformed time series (leakage)
    ts_exists = not ts_leak.empty  # Check if transformed time series already exists

    ts_leak_data = client.time_series.data.retrieve(
        external_id=ts_output_name, limit=1).to_pandas()
    # If no data in output time series, run cognite function from first available date of original time series until date with last updated datapoint
    if ts_leak_data.empty:
        first_date_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                           aggregates="average",
                                                           granularity="1m",
                                                           limit=1).to_pandas().index[0]
        start_date = first_date_orig

    # Search through original time series for backfilling,
    if not ts_leak_data.empty:
        ts_orig_all = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                       aggregates="average",
                                                       granularity="1m",
                                                       end=pd.to_datetime(
                                                           end_date),
                                                       limit=-1,
                                                       ).to_pandas()
        ts_leak_all = client.time_series.data.retrieve(
            external_id=ts_output_name, limit=-1).to_pandas()

        ts_orig_dates = pd.to_datetime(ts_orig_all.index.date)
        ts_leak_dates = ts_leak_all.index
        missing_dates = ts_orig_dates[~ts_orig_dates.isin(
            ts_leak_dates)].unique()  # Dates in input signal NOT in transformed signal - indicates values have been backfilled

        # TODO: Do transformations for missing dates (recently backfilled)

        # TODO:
        # 1. For each write, store number of data points in original signal for each date - store in file associated with dataset ID.
        # number of datapoints for each date
        num_orig_dates_new = ts_orig_dates.groupby(ts_orig_dates.index).count()
        num_orig_dates_new.to_csv("num_data_dates_orig_ts.csv")
        nume_orig_dates_old = client.files.retrieve()
        # 2. If number of datapoints INCREASES for certain dates (= backfilled), redo transformations for these dates.
        backfill_dates = num_orig_dates_new[num_orig_dates_new >
                                            nume_orig_dates_old].index
        # 3. (Update content of this file for each new write.)
        for date in backfill_dates:
            pass
            # Redo transformations

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
        print(f"End date: {end_date}")
        start_date = pd.to_datetime(start_date - timedelta(days=1)
                                    ).date()  # start of previous date
        end_date = pd.to_datetime(start_date + timedelta(days=1))

    df = df.rename(columns={ts_orig_extid + "|average": ts_input_name})
    # df = df.iloc[:800000]
    df['time_sec'] = (df.index - datetime(1970, 1, 1)
                      ).total_seconds()  # tot seconds since epoch'
    vol_perc = df[ts_input_name]

    return
    # CHECK IF ANY MISSING VALUES -> indicates lost connection
    if False:
        time_deltas = np.diff(df["time_sec"])
        # "correct" time period between data points
        mode = stats.mode(time_deltas).mode
        # skipped period in time (NB: assuming datapoints ingested evenly)
        period_skips = [(df["time_sec"][idx], dt)
                        for idx, dt in enumerate(time_deltas) if dt != mode]
        # period_skips = ~all(dt == mode for dt in time_deltas)

        if any(period_skips):
            timestamp_skips = []
            # timestamp_skips = []
            for i in range(len(period_skips)):
                # number of consecutive time stamps missing for skipped period i
                # -1 since no timestamp skips if period_skips[i][2] = mode
                num_skipped_i = round(period_skips[i][2]/mode) - 1
                for j in range(num_skipped_i):
                    timestamp_skips.append(period_skips[i][1] + j*mode)
            # 1. Insert nan values for missing time steps
            # TODO
            nan_values = np.empty(len(timestamp_skips))
            nan_values[:] = np.nan
            df_miss = pd.DataFrame([nan_values], index=timestamp_skips)
            df = pd.concat([df, df_miss])
            # 2. Interpolate nan values between two closest true values
            # TODO
            df = df.interpolate(method="linear")

        # No backfilling ??

    # STEP 2: Filter signal
    if ts_leak_data.empty:  # Relax the filtering to save computations
        delta = 0*len(vol_perc)
    else:
        delta = 0
    if __name__ == "__main__":
        import time
        print("Running lowess smoothing ...")
        t_start = time.perf_counter()
    smooth = lowess(vol_perc, df['time_sec'], is_sorted=True,
                    frac=0.01, it=0, delta=delta)
    if __name__ == "__main__":
        t_stop = time.perf_counter() - t_start
        print("... Finished!")
        print(f"Lowess smoothing took {round(t_stop, 1)} seconds.")
    df_smooth = pd.DataFrame(smooth, columns=["time_sec", "smooth"])

    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'time_stamp'})
    df = pd.merge(df, df_smooth, on='time_sec')
    df.set_index('time_stamp', drop=True, append=False,
                 inplace=True, verify_integrity=False)

    if False:
        df.plot(x="time_sec", y=[ts_input_name, "smooth"])
        plt.show()

    # STEP 3: Create new time series, if not already exists
    if not ts_exists:
        print("Output time series does not exist. Creating ...")
        # client.time_series.delete(external_id=ts_output_name)
        client.time_series.create(TimeSeries(
            name=ts_output_name, external_id=ts_output_name, data_set_id=dataset_id))

    # STEP 4: Calculate daily average drainage rate [% of tank vol / sec]
    try:
        df["derivative"] = np.gradient(df['smooth'], df["time_sec"])
    except:
        raise IndexError(
            "No datapoints found for selected date range. Cannot compute drainage rate.")

    derivative_value_excl = data['derivative_value_excl']
    df['derivative_excl_filling'] = df["derivative"].apply(
        lambda x: 0 if x > derivative_value_excl or pd.isna(x) else x)  # not interested in large INLET fluxes

    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['time_stamp']).dt.date
    df["Date"] = pd.to_datetime(df["Date"])

    mean_drainage_day = df.groupby('Date')['derivative_excl_filling'].mean(
    )*tank_volume/100  # avg drainage rate per DAY
    # external ID is column name
    # {ts_output_name: mean_drainage_day}
    mean_df = pd.DataFrame({ts_output_name: mean_drainage_day})
    mean_df.index = pd.to_datetime(mean_df.index)

    new_df = pd.merge(df, mean_df, on="Date")
    new_df["draining_rate [L/min]"] = new_df["derivative_excl_filling"] * \
        tank_volume/100  # drainage rate per TIME STAMP

    # insert transformed signal for new time range
    client.time_series.data.insert_dataframe(mean_df)

    return new_df[[ts_output_name]].to_json()  # jsonify output signal


if __name__ == '__main__':
    from initialize import initialize_client

    cdf_env = "dev"
    client = initialize_client(cdf_env, cache_token=False)

    ts_name = "VAL_11-LT-95034A:X.Value"
    tank_volume = 1400
    derivative_value_excl = 0.002
    # start_date = datetime(2023, 3, 21, 1, 0, 0)

    data_dict = {'tot_days': 0, 'tot_minutes': 15,  # convert date to str to make it JSON serializable
                 'ts_input_name': ts_name, 'ts_output_name': "VAL_11-LT-95034A:X.CDF.D.AVG.LeakValue",
                 'derivative_value_excl': derivative_value_excl, 'tank_volume': tank_volume,
                 'cdf_env': cdf_env, 'dataset_id': 1832663593546318}  # NB: change dataset id when going to dev/test/prod!

    new_df = handle(client, data_dict)
