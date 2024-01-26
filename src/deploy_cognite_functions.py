import zipfile
import os
import pandas as pd
from typing import Tuple
import time

from cognite.client._cognite_client import CogniteClient
from cognite.client.data_classes import functions
from cognite.client.data_classes import ClientCredentials

from utilities import dataset_abbreviation

from dotenv import load_dotenv
load_dotenv("../authentication-ids.env")

def deploy_cognite_functions(data_dict: dict, client: CogniteClient,
                             single_call=True, scheduled_call=False):
    """General procedure to deploy Cognite Functions through schedule,
    using zip-file approach to link data to a designated dataset

    Args:
        data_dict (list): Dictionary with input data and parameters for the Cognite Function
        client (CogniteClient): instantiated Cognite Client
        single_call (bool, optional): If running a single call to the Cognite Function. This is necessary for first transformation of input time series. Defaults to True.
        scheduled_call (bool, optional): If running Cognite Function on schedule. Defaults to False.

    Raises:
        FileNotFoundError: If zip-file connected to associated dataset is not found.
    """
    cognite_function = client.functions.retrieve(external_id=f"{data_dict['function_name']}")

    if cognite_function is None: # function not exist, create ...
        folder = os.getcwd().replace("\\", "/")
        folder_cf = folder + "/" + data_dict["function_name"]

        zip_name = "zip_handle.zip"
        zip_path = f"{folder_cf}/{zip_name}"

        dataset_abbr = dataset_abbreviation(client, data_dict["optional"], data_dict["dataset_id"])
        if data_dict["function_name"] == f"{dataset_abbr}_test": # for unit testing
            parent = "/.."
        else:
            parent = ""

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            try:
                zipf.write(f'{folder_cf}/requirements.txt', arcname='requirements.txt')
                zipf.write(f'{folder_cf}{parent}/../handler_utils.py', arcname='handler_utils.py')
                zipf.write(f'{folder_cf}/handler.py', arcname='handler.py')
                zipf.write(f'{folder_cf}/transformation.py', arcname='transformation.py')
                zipf.write(f'{folder_cf}{parent}/../transformation_utils.py', arcname='transformation_utils.py')
                zipf.write(f'{folder_cf}{parent}/../utilities.py', arcname='utilities.py')
            except:
                raise FileNotFoundError(f"Make sure you have the following three required files in your Cognite Function folder {folder_cf}:\n" \
                                            "\trequirements.txt\n" \
                                            "\thandler.py\n" \
                                            "\ttransformation.py")
        zipf.close()

        uploaded = client.files.upload(path=zip_path, name=zip_name, data_set_id=data_dict["dataset_id"])

        client.functions.create(
                                name=f"{data_dict['function_name']}",
                                external_id=f"{data_dict['function_name']}",
                                # folder=".",
                                file_id=uploaded.id,
                                runtime="py311"
                            )

        cognite_function = client.functions.retrieve(external_id=f"{data_dict['function_name']}")
        print("Cognite Function created. Waiting for deployment status to be ready ...")
        while cognite_function.status != "Ready":
            # wait for function to be ready
            time.sleep(3) # wait three second to avoid too frequent loop iterations
            cognite_function.update()

        print("Ready for deployement.")

    # Single call to function to run initial transformation before scheduling
    if single_call:
        print("Calling Cognite Function individually ...")
        cognite_function.call(data=data_dict)
        print("... Done")

    # AFTER initial call, function can be scheduled
    if scheduled_call:
        cron_interval = data_dict["cron_interval_min"]

        now = pd.Timestamp.now(tz="CET").floor("1s").tz_convert("UTC")
        print("Preparing schedule to start sharp at next minute ...")
        while now.second > 0: # Align schedule to start at minute sharp, for sampling rate >= 1 min
            time.sleep(1)
            now = pd.Timestamp.now(tz="CET").floor("1s").tz_convert("UTC")

        print(f"Setting up Cognite Function schedule at time {now} ...")
        client.functions.schedules.create(
            name=f"{data_dict['schedule_name']}",
            cron_expression=f"*/{cron_interval} * * * *", # every cron_interval min
            function_id=cognite_function.id,
            client_credentials=ClientCredentials(client_id=str(os.getenv("CLIENT_ID")),
                                                 client_secret=str(os.getenv("CLIENT_SECRET"))),
            description=f"Calculation scheduled every {cron_interval} minute",
            data=data_dict,
        )
        print("... Done")