import re
import pandas as pd
from io import BytesIO
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

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
AGG_PERIOD[r"hour|h"] = "H"
AGG_PERIOD[r"day|d"] = "D"
AGG_PERIOD[r"month"] = "M"
AGG_PERIOD[r"year"] = "Y"

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
