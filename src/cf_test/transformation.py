
def main_calculation(data, *ts_inputs):
    """Main function for transforming a set of input timeseries from
    whatever calculations you would like to perform.

    Args:
        data (dict): calculation-specfic parameters for Cognite Function
        ts_inputs (list | pd.DataFrame): list of inputs time series to transform, each one given as a pd.DataFrame

    Returns:
        (list): transformed time series given as pd.Series
    """
    # The following is just a dummy calculation
    ts_out = []
    for ts in ts_inputs:
        ts_df = ts.rolling(window=int(len(ts)/10)).mean()
        ts_out.append(ts_df.squeeze()) # Necessary to convert from pd.DataFrame to pd.Series!
    return ts_out
