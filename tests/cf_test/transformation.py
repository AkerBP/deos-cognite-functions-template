def calc_test(data, ts):
    return (ts - 10).squeeze()

def calc_multiple_outputs(data, *ts_data):
    ts_out = []
    for ts in ts_data:
        ts_df = ts.apply(lambda x: x if x > 2 else 1)
        ts_out.append(ts_df.squeeze()) # convert to series
    return ts_out