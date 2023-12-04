def main_calculation(data, *ts_data):
    ts_out = []
    for ts in ts_data:
        ts_df = ts.rolling(window=int(len(ts)/10)).mean()
        ts_out.append(ts_df.squeeze()) # convert to series
    return ts_out