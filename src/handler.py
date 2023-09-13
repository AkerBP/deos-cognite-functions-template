from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cognite.client import CogniteClient
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import seasonal_decompose

from initialize import initialize_client

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
    start_date = data['start_date']
    end_date = start_date + timedelta(days=data['tot_days'])
    ts_name = data['ts_name']

    # STEP 1: Load time series from name and aggregate

    ts_all = client.time_series.search(name=ts_name) # find time series by name
    cdf_ext_id = ts_all[0].external_id # extract its external id
    df_cdf = client.time_series.data.retrieve(external_id=cdf_ext_id,
                                        aggregates="average",
                                        granularity="1m",
                                        start=start_date,
                                        end=end_date) # load time series by external id

    df = df_cdf.to_pandas()
    df = df.rename(columns = {cdf_ext_id + "|average": ts_name})

    # STEP 2: Filter signal
    df['time_sec'] = (df.index - datetime(1970,1,1)).total_seconds() # total seconds elapsed of each data point since 1970
    vol_perc = df[ts_name]
    smooth = lowess(vol_perc, df['time_sec'], is_sorted=True, frac=0.01, it=0)
    df_smooth = pd.DataFrame(smooth, columns=["time_sec", "smooth"])

    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'time_stamp'})
    df = pd.merge(df, df_smooth, on='time_sec') # merge smooth signal into origianl dataframe
    df.set_index('time_stamp', drop=True, append=False, inplace=True, verify_integrity=False)

    # STEP 3: Calculate derivative
    # df["time_diff"] = df.index.to_series().diff().dt.total_seconds()
    # df["smooth_diff"] = df["smooth"].diff()
    # derivative = df['smooth_diff'] / df['time_diff']
    df["derivative"] = np.gradient(df['smooth'], df["time_sec"]) # Unit: vol_percentage/time [% of tank vol / sec]
    # replace when derivative is greater than alfa
    derivative_value_excl = data['derivative_value_excl']
    df['derivative_excl_filling'] = df["derivative"].apply(lambda x: 0 if x > derivative_value_excl or pd.isna(x) else x)

    # STEP 4: Calculate daily average drainage rate
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['time_stamp']).dt.date
    #df['Time'] = pd.to_datetime(df['time_stamp']).dt.time
    mean_drainage_day = df.groupby('Date')['derivative_excl_filling'].mean()*tank_volume/100
    mean_df = pd.DataFrame({'mean_derivative_by_day': mean_drainage_day})

    new_df = pd.merge(df, mean_df, on="Date")
    new_df["draining_rate [L/min]"] = new_df["derivative_excl_filling"]*tank_volume/100

    return new_df


if __name__ == '__main__':
    client, status = initialize_client()

    ts_name = "VAL_11-LT-95034A:X.Value"
    tank_volume = 1400
    derivative_value_excl = 0.002
    start_date = datetime(2023, 3, 21, 1, 0, 0)

    data_dict = {'start_date':start_date, 'tot_days':25, 'ts_name':ts_name,
                'derivative_value_excl':derivative_value_excl, 'tank_volume':tank_volume}
    handle(client, data_dict)
    # Create function
    get_df = client.functions.create(
        name="calc-drainage-rate",
        #external_id="load-time-series",
        function_handle=handle
    )
    # Call function
    func_info = {'function_id':'calc-drainage-rate'}
    call_get_df = get_df.call(data=data_dict, function_call_info=func_info)