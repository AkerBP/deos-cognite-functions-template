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
from deploy_cognite_functions import deploy_cognite_functions
from generate_cf import generate_cf, write_handle, write_transformation, get_toml_dependencies

@pytest.fixture
def cognite_client_mock():
    with monkeypatch_cognite_client() as client:
        return client

@pytest.fixture
def cf_test_path():
    cf_name = "test"
    cf_path = os.getcwd().replace("\\", "/") + f"/cf_{cf_name}"
    return cf_path

@pytest.fixture
def dummy_toml():
    return '''[tool.poetry]
    name = "cf_test"
    version = "0.1.0"
    description = ""
    readme = "README.md"

    [tool.poetry.dependencies]
    python = "^3.11"
    ipykernel = "^6.28.0"
    pytest = "^7.4.4"
    numpy = "^1.26.3"
    cognite-sdk = "^7.10.0"
    pandas = "^2.1.4"
    python-dotenv = "^1.0.0"
    dummy-python-package = "2.0.0"
    statsmodels = "^0.14.1"

    [build-system]
    requires = ["poetry-core"]
    build-backend = "poetry.core.masonry.api"
    '''

def test_make_files(cf_test_path):
    cf_name = "test"
    generate_cf(cf_name, add_packages=[])
    assert os.path.exists(cf_test_path)

    all_files = ["__init__.py", "handler.py", "transformation.py", "requirements.py"]
    for i, (root, dir, files) in enumerate(os.walk(cf_test_path)):
        assert files in all_files
        all_files.pop(i)

def test_handle_content(cf_test_path):
    # Assert that handle file contains right content
    expected_content = write_handle()

    with open(cf_test_path+"/handler.py", "r") as file:
        file_content = file.read()

    assert expected_content == file_content

def test_transformation_content():
    # Assert that transformation file contains right content
    expected_content = write_transformation()

    with open(cf_test_path+"/transformation.py", "r") as file:
        file_content = file.read()

    assert expected_content == file_content


def test_get_toml_dependencies(dummy_toml):
    if not os.path.exists("toml_test"):
        os.mkdir("test_toml")

    with open("test_toml/pyproject.toml", "w") as file:
        file.write(dummy_toml)
        file.close()

    get_toml_dependencies("test_toml", "test_toml/", include_version=True)

    expected_output = """
    python
    ipykernel
    pytest
    numpy
    cognite-sdk
    pandas
    python-dotenv
    dummy-python-package
    statsmodels
    """
    with open("test_toml/requirements.txt", "r") as file:
        file_output = file.read()

    # Remove leading empty lines
    file_output = file_output.lstrip("\n")

    assert file_output == expected_output