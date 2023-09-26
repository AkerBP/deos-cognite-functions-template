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
        end_date = pd.Timestamp.now() - pd.Timedelta(days=449)
    else:
        ts_input_name = data['ts_input_name']
        end_date = pd.Timestamp.now()

    start_date = end_date - \
        timedelta(days=data['tot_days'], minutes=data['tot_minutes'])

    ts_output_name = data['ts_output_name']
    dataset_id = data['dataset_id']

    # STEP 1: Load time series from name and aggregate

    ts_orig = client.time_series.list(
        name=ts_input_name).to_pandas()  # find time series by name
    ts_orig_extid = ts_orig.external_id[0]  # extract its external id

    # load NEW time series
    # already aggregated, and we want full time series, so no start/enddate provided
    ts_leak = client.time_series.list(
        name=ts_output_name).to_pandas()
    try:
        ts_leak_extid = ts_leak.external_id[0]
        first_update = False
        ts_leak = client.time_series.data.retrieve(external_id=ts_leak_extid)
        df_leak = ts_leak.to_pandas()
        df_leak = df_leak.rename(
            columns={ts_leak_extid + "|average": ts_input_name})
        df_leak.index = pd.to_datetime(df_leak.index)  # .dt.date

        first_update = False

        # load ORIGINAL time series - start and end dates defines most recent schedule
        ts_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                   aggregates="average",
                                                   granularity="15m",
                                                   start=start_date,
                                                   end=end_date)
    except:
        first_update = True  # time series not found
        # load ORIGINAL time series - load all data so far (no start/end dates) as this is first time we transform time series
        end_date = pd.Timestamp.now()
        if data['run_sandbox']:
            end_date = end_date - pd.Timedelta(days=452)
        ts_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                   aggregates="average",
                                                   granularity="15m",
                                                   end=end_date)

    df = ts_orig.to_pandas()
    df = df.rename(columns={ts_orig_extid + "|average": ts_input_name})

    # STEP 2: Filter signal
    # total seconds elapsed of each data point since 1970
    df['time_sec'] = (df.index - datetime(1970, 1, 1)).total_seconds()
    vol_perc = df[ts_input_name]
    smooth = lowess(vol_perc, df['time_sec'], is_sorted=True, frac=0.01, it=0)
    df_smooth = pd.DataFrame(smooth, columns=["time_sec", "smooth"])

    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'time_stamp'})
    # merge smooth signal into origianl dataframe
    df = pd.merge(df, df_smooth, on='time_sec')
    df.set_index('time_stamp', drop=True, append=False,
                 inplace=True, verify_integrity=False)

    # STEP 3: Create new time series
    if first_update:
        # client.time_series.delete(external_id=ts_output_name)
        ts_output = client.time_series.create(TimeSeries(
            name=ts_output_name, external_id=ts_output_name, data_set_id=dataset_id))

    # STEP 4: Calculate daily average drainage rate
    # Unit: vol_percentage/time [% of tank vol / sec]
    try:
        df["derivative"] = np.gradient(df['smooth'], df["time_sec"])
    except:
        raise IndexError(
            "No datapoints found for selected date range. Cannot compute drainage rate.")
    # replace when derivative is greater than alfa
    derivative_value_excl = data['derivative_value_excl']
    df['derivative_excl_filling'] = df["derivative"].apply(
        lambda x: 0 if x > derivative_value_excl or pd.isna(x) else x)

    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['time_stamp']).dt.date
    df["Date"] = pd.to_datetime(df["Date"])

    mean_drainage_day = df.groupby('Date')['derivative_excl_filling'].mean(
    )*tank_volume/100  # avg drainage rate per DAY
    # Use external ID as column name
    mean_df = pd.DataFrame({ts_output_name: mean_drainage_day})
    mean_df.index = pd.to_datetime(mean_df.index)

    # Append previously calculated time series
    # get last date of previous calculated time series
    if not first_update:
        last_date = df_leak.index.max()
        # only select newly calculated values (avoid duplicates)
        # mean_df.index = (mean_df.index - datetime(1970, 1, 1)).total_seconds()
        # mean_df["Date"] = pd.to_datetime(mean_df.index).dt.date
        df_new = mean_df.loc[mean_df.index > last_date]
        df_full = pd.concat([df_leak, df_new])
        # datetime index required for inserting dataframe into time series
    else:
        df_full = mean_df  # no merging if first update

    df_full.index = pd.to_datetime(df_full.index)
    new_df = pd.merge(df, mean_df, on="Date")

    new_df["draining_rate [L/min]"] = new_df["derivative_excl_filling"] * \
        tank_volume/100  # drainage rate per TIME STAMP

    ts_inserted = client.time_series.data.insert_dataframe(df_full)

    return new_df[[ts_output_name]].to_json()  # , ts_output, ts_inserted


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
    # # Create function
    # get_df = client.functions.create(
    #     name="calc-drainage-rate",
    #     # external_id="load-time-series",
    #     function_handle=handle
    # )
    # # Call function
    # func_info = {'function_id': 'calc-drainage-rate'}
    # call_get_df = get_df.call(data=data_dict, function_call_info=func_info)
