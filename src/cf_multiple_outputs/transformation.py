def calculation(data, *ts_data):
    ts_out = []
    for ts in ts_data:
        ts_out.append(ts.rolling(window=int(len(ts)/10)).mean())
    return ts_out