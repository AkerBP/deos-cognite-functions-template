import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import numpy as np

def run_transformation(df, data):
    if 'plot_filter' in data:
        plot_filter = data["plot_filter"]
    else:
        plot_filter = False
    df = filter_ts(df, plot_filter, data)

    # STEP 5: Calculate daily average drainage rate [% of tank vol / sec]
    daily_avg_drainage, df = calc_daily_avg_drain(df, data)

    # external ID is column name
    mean_df = pd.DataFrame({data["ts_output_name"]: daily_avg_drainage})
    mean_df.index = pd.to_datetime(mean_df.index)

    new_df = pd.merge(df, mean_df, on="Date")
    new_df["draining_rate [L/min]"] = new_df["derivative_excl_filling"] * \
        data["tank_volume"] / \
        100  # drainage rate per TIME STAMP (not avg per day)

    return mean_df


def filter_ts(df, plot_filter, data):
    ts_input_name = data["ts_input_name"]
    vol_perc = df[ts_input_name]

    if __name__ == "__main__" and not data["backfill"]:
        import time
        print("Running lowess smoothing ...")
        t_start = time.perf_counter()

    if "lowess_frac" in data:
        frac = data["lowess_frac"]
    else:
        frac = 0.01
    if "lowess_delta" in data:
        delta = data["lowess_delta"]
    else:
        delta = 0
    smooth = lowess(vol_perc, df['time_sec'], is_sorted=True,
                    frac=frac, it=0, delta=delta*len(vol_perc))

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
