import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ast
import os
import sys
from dotenv import load_dotenv

from cognite.client.testing import monkeypatch_cognite_client
from cognite.client.data_classes import TimeSeries
from cognite.client.data_classes import FileMetadata
from cognite.client.data_classes.functions import Function, FunctionCallList, FunctionCall
from cognite.client.data_classes.functions import FunctionSchedulesList
from cognite.client.data_classes.functions import FunctionSchedule

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# parent_path = parent_path + "\\src"
# print(parent_path + "\\src")
if parent_path not in sys.path:
    sys.path.append(parent_path)

from transformation_utils import RunTransformations
from handler_utils import PrepareTimeSeries
from initialize import initialize_client
from deploy_cognite_functions import deploy_cognite_functions, list_scheduled_calls
from generate_cf import generate_cf, write_handle, write_transformation, get_toml_dependencies

load_dotenv("../../handler-data.env")


DATA_DICT = {'testing': True,
            "ts_input_names": [str(os.getenv("TS_INPUT_NAME"))],
            "ts_output": {"names": ["test_deploy_cognite_functions"],
                            "description": [None],
                            "unit": [None]},
            'function_name': "cf_test",
            'schedule_name': "cf_test_schedule",
            'calculation_function': "main_test",
            'granularity': 60,
            'aggregate': {},
            "cron_interval_min": str(5),
            "backfill_period": 1,
            "backfill_hour": 2,
            "backfill_min_start": 3, "backfill_min_end": 8,
            "dataset_id": str(os.getenv("DATASET_ID")),
            "calc_params": {}}


CREATED_FILE = [
    FileMetadata(name="zip_handle.zip",
                 #external_id="zip_handle",
                 data_set_id=int(DATA_DICT["dataset_id"]))
]

CREATED_FUNCTION = [
    Function(
        external_id=DATA_DICT["function_name"],
        name=DATA_DICT["function_name"],

    )
]

RETRIEVED_FUNCTION = [
    Function(
        external_id=DATA_DICT["function_name"]
    )
]

CREATED_SCHEDULE = [
    FunctionSchedule(
        name=DATA_DICT["schedule_name"],
        # function_external_id=DATA_DICT["function_name"], # NB: this parameter will be removed in future!
        description=f"Calculation scheduled every {DATA_DICT['cron_interval_min']} minute",
        cron_expression=f"*/{DATA_DICT['cron_interval_min']} * * * *",
    )
]

@pytest.fixture
def cognite_client_mock():
    with monkeypatch_cognite_client() as client:
        client.files.upload.return_value = CREATED_FILE
        client.functions.create.return_value = CREATED_FUNCTION
        client.functions.retrieve.return_value = RETRIEVED_FUNCTION
        client.functions.schedules.create.return_value = CREATED_SCHEDULE
        return client


def test_create_cognite_function(cognite_client_mock):
    """Test that zip file is uploaded as File resource, and
    cognite function created as Function resource to CDF.
    Function is NOT called in this test.
    """
    # func = cognite_client_mock.functions.retrieve(external_id=data["function_name"])
    # if func is not None: # Since we are testing creation of function, we must delete any existing
    #     cognite_client_mock.functions.delete(external_id=data["function_name"])

    deploy_cognite_functions(DATA_DICT, cognite_client_mock,
                             single_call=False, scheduled_call=False)

    assert cognite_client_mock.functions.retrieve.call_count == 1
    # Assert that zip file uploaded to CDF as File resource
    assert cognite_client_mock.files.upload.call_count == 1
    uploaded_file_true = [file.dump() for file in CREATED_FILE]
    uploaded_file_returned = [file.dump() for file in cognite_client_mock.files.upload.call_args[0]]
    assert uploaded_file_returned == uploaded_file_true

    # Assert that cognite function created in CDF as Function resource
    assert cognite_client_mock.function.create.call_count == 1
    created_function_true = [func.dump() for func in CREATED_FUNCTION]
    created_function_returned = [func.dump() for func in cognite_client_mock.function.create.call_args[0]]
    assert created_function_returned == created_function_true

def test_single_call(cognite_client_mock):
    """Test that intended function has been called once when single call is specified
    """

    deploy_cognite_functions(DATA_DICT, cognite_client_mock,
                             single_call=True, scheduled_call=False)

    # Test correct retrieval of function
    assert cognite_client_mock.functions.retrieve.call_count == 1 # or 2 ???
    retrieved_function_true = [func.dump() for func in RETRIEVED_FUNCTION]
    retrieved_function_returned = [func.dump() for func in cognite_client_mock.function.retrieve.call_args[0]]
    assert retrieved_function_returned == retrieved_function_true

    # func = cognite_client_mock.functions.retrieve(external_id="cf_test")
    # single_call_id = func.list_calls().to_pandas().id[0]

    assert cognite_client_mock.functions.call.call_count == 1
    # TODO Don't think this will work ...
    assert cognite_client_mock.function.call.call_args[0] == DATA_DICT

def test_scheduled_call(cognite_client_mock):
    """Test that schedule is set up of intended function when scheduled call is specified
    """
    deploy_cognite_functions(DATA_DICT, cognite_client_mock,
                             single_call=False, scheduled_call=True)

    assert cognite_client_mock.functions.retrieve.call_count == 1 # or 3 ???

    assert cognite_client_mock.functions.schedules.create.call_count == 1

    created_schedule_true = [sch.dump() for sch in CREATED_SCHEDULE]
    created_schedule_returned = [sch.dump() for sch in cognite_client_mock.functions.schedules.create.call_args[0]]
    assert created_schedule_returned == created_schedule_true