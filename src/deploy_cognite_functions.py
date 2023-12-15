import zipfile
import os
import pandas as pd
from typing import Tuple

from cognite.client._cognite_client import CogniteClient
from cognite.client.data_classes import functions

def deploy_cognite_functions(data_dict: dict, client: CogniteClient, cron_interval: str,
                             single_call=True, scheduled_call=False):
    """General procedure to deploy Cognite Functions through schedule,
    using zip-file approach to link data to a designated dataset

    Args:
        data_dict (list): Dictionary with input data and parameters for the Cognite Function
        client (CogniteClient): instantiated Cognite Client
        cron_interval (str): Minute-interval to run schedule on
        single_call (bool, optional): If running a single call to the Cognite Function. This is necessary for first transformation of input time series. Defaults to True.
        scheduled_call (bool, optional): If running Cognite Function on schedule. Defaults to False.

    Raises:
        FileNotFoundError: If zip-file connected to associated dataset is not found.
    """
    func_limits = functions.FunctionsLimits(timeout_minutes=60, cpu_cores=0.25, memory_gb=1, runtimes=["py39"], response_size_mb=2.5)
    cognite_function = client.functions.retrieve(external_id=f"{data_dict['function_name']}")

    if cognite_function is None: # function not exist, create ...
        import time
        folder = os.getcwd().replace("\\", "/")
        folder_cf = folder + "/" + data_dict["function_name"]

        zip_name = "zip_handle.zip"
        zip_path = f"{folder_cf}/{zip_name}"

        if data_dict["function_name"] == "cf_test": # for unit testing
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
            except:
                print(folder_cf)
                raise FileNotFoundError("Make sure you have the following three required files in your Cognite Function folder:\n" \
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
        print("Setting up Cognite Function schedule ...")
        client.functions.schedules.create(
            name=f"{data_dict['function_name']}",
            cron_expression=f"*/{cron_interval} * * * *", # every cron_interval min
            function_id=cognite_function.id,
            #client_credentials=client,
            description=f"Calculation scheduled every {cron_interval} minute",
            data=data_dict
        )
        print("... Done")

def list_scheduled_calls(data_dict: dict, client: CogniteClient) -> Tuple[int, pd.DataFrame]:
    """List all scheduled calls to Cognite Function

    Args:
        data_dict (dict): Dictionary with input data and parameters for the Cognite Function
        client (CogniteClient): client to authenticate with Cognite

    Returns:
        my_schedule_id (int): id of deployed schedule
        all_calls (pd.DataFrame): table of all calls made
    """
    import time
    my_func = client.functions.retrieve(external_id=data_dict["function_name"])
    try:
        my_schedule_id = client.functions.schedules.list(
                name=data_dict["function_name"]).to_pandas().id[0]
    except:
        raise NotImplementedError(f"No schedule for function {data_dict['function_name']} exists.")
    all_calls = my_func.list_calls(
                schedule_id=my_schedule_id, limit=-1).to_pandas()
    while all_calls.empty: # wait for first call
        time.sleep(1)
        all_calls = my_func.list_calls(
                schedule_id=my_schedule_id, limit=-1).to_pandas()
    print(f"Calls for Cognite Function '{my_func.name}':\n{all_calls.tail()}")

    return my_schedule_id, all_calls