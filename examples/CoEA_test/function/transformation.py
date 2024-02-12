import pandas as pd

def transformation(data_dict: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Returns the transformed df

    Args:
        data_dict (dict): Input parameters needed for the transformation.
        df (pd.DataFrame): Your original, untransfrmed data.

    Returns:
        pd.DataFrame: Your transformed timeseries.
    """
    new_df = df**2
    new_df = new_df.rename(columns={data_dict["ts_input_names"][0]: data_dict["ts_output"]["names"][0]})
    return new_df