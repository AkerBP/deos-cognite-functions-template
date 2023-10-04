from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cognite.client import CogniteClient
from cognite.client.data_classes import TimeSeries
from statsmodels.nonparametric.smoothers_lowess import lowess


def handle(client: CogniteClient, data: dict) -> pd.DataFrame:
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

    if data['run_sandbox']:
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

    ts_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                               aggregates="average",
                                               granularity="1m",
                                               start=start_date,
                                               end=end_date)

    df = ts_orig.to_pandas()
    df = df.rename(columns={ts_orig_extid + "|average": ts_input_name})

    # STEP 2: Filter signal
    df['time_sec'] = (df.index - datetime(1970, 1, 1)
                      ).total_seconds()  # tot seconds since epoch
    vol_perc = df[ts_input_name]
    smooth = lowess(vol_perc, df['time_sec'], is_sorted=True, frac=0.01, it=0)
    df_smooth = pd.DataFrame(smooth, columns=["time_sec", "smooth"])

    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'time_stamp'})
    df = pd.merge(df, df_smooth, on='time_sec')
    df.set_index('time_stamp', drop=True, append=False,
                 inplace=True, verify_integrity=False)

    # STEP 3: Create new time series, if not already exists
    if not ts_exists:
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

    run_sandbox = True
    client = initialize_client(run_sandbox, cache_token=True)

    ts_name = "VAL_11-LT-95034A:X.Value"
    tank_volume = 1400
    derivative_value_excl = 0.002
    # start_date = datetime(2023, 3, 21, 1, 0, 0)

    data_dict = {'tot_days': 0, 'tot_minutes': 15,  # convert date to str to make it JSON serializable
                 'ts_input_name': ts_name, 'ts_output_name': "VAL_11-LT-95034A:X.CDF.D.AVG.LeakValueTest",
                 'derivative_value_excl': derivative_value_excl, 'tank_volume': tank_volume,
                 'run_sandbox': True, 'dataset_id': 3197476513083188}  # NB: change dataset id when going to dev/test/prod!

    new_df = handle(client, data_dict)
