import re
import pandas as pd
from io import BytesIO
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

from cognite.client import CogniteClient

class RegexDict:
    """Represent dictionary where mulitple keys can be associated
    with the same value using regex.
    """
    def __init__(self):
        self._data = {}

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getitem__(self, key):
        for pattern, value in self._data.items():
            if re.fullmatch(pattern, key):
                return value
        raise KeyError(key)

AGG_PERIOD = RegexDict()
AGG_PERIOD[r"second|s"] = "S"
AGG_PERIOD[r"minute|m"] = "min"
AGG_PERIOD[r"hour|h"] = "h"
AGG_PERIOD[r"day|d"] = "D"
AGG_PERIOD[r"month"] = "M"
AGG_PERIOD[r"year"] = "Y"

def get_external_id_from_name(client: CogniteClient, ts_name: str):
    """Get external id from provided name of time series

    Args:
        client (CogniteClient): authenticated cognite client
        ts_name (str): name of input time series

    Return:
        (str): external id of time series
    """
    return client.time_series.search(name=ts_name, limit=1)[0].external_id


def dataframe_to_bytes(df: pd.DataFrame):
    """Convert dataframe to bytes object

    Args:
        df (pd.DataFrame): dataframe object to convert

    Returns:
        str: excel-readable bytes object
    """
    bytes_object = BytesIO()
    wb = Workbook()
    ws = wb.active
    for r in dataframe_to_rows(df, index=True, header=True):
        ws.append(r)
    wb.save(bytes_object)

    return bytes_object.getvalue()

def first_letters_of_words(string: str):
    """Concatenate first letter of each word in a string
    to make an abbreviated form of the string.

    Args:
        string (str): string/sentence to make abbreviation from

    Returns:
        str: abbreviated form of string
    """
    words = string.split()
    first_letters = [word[0] for word in words if word[0].isalnum()]
    return ''.join(first_letters)

def dataset_abbreviation(client, optional, ds_id):
    if "dataset_abbr" in optional:
        dataset_abbr = optional["dataset_abbr"]
    else:
        dataset_name = client.data_sets.retrieve(id=ds_id).name
        dataset_abbr = first_letters_of_words(dataset_name)
    return dataset_abbr
