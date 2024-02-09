import shutil
import argparse
import zipfile
import time
import os

import pandas as pd

from pathlib import Path
from cognite.client import CogniteClient
from cognite.client.data_classes import ClientCredentials
from cf_template.initialize_cdf_client import initialize_cdf_client
from typing import Union


class NewCogniteFunction:
    def __init__(
        self,
        cdf_env: str="dev",
        client: CogniteClient=None,
        description: str="",
        input_ts_names: list=None,
        input_ts_sampling_rate: str="1m",
        output_ts_names: list=None,
        output_ts_descriptions: list=None,
        output_ts_units: list=None,
        output_ts_agg_method: Union[None, str]=None,
        output_ts_agg_freq: Union[None, str]=None,
        dataset_id: Union[str, int]=None,
        name_of_function: Union[None, str]=None,
        name_of_schedule: str=None,
        cron_interval_in_minutes: str="15",
        backfill_period: int=7,
        backfill_hour: int=12,
        backfill_min_start: int=0,
        historic_start_time: str="2023-1-1 00:00",
        deployment_single_call: bool=False,
        deployment_scheduled_call: bool=True
    ) -> None:
        self.cdf_env = cdf_env
        self.client = client if client is not None else initialize_cdf_client(self.cdf_env)
        self.description = description
        self.input_ts_names = input_ts_names
        self.input_ts_sampling_rate = input_ts_sampling_rate
        self.output_ts_names = output_ts_names
        self.output_ts_descriptions = output_ts_descriptions
        self.output_ts_units = output_ts_units
        self.ts_output = {"names": self.output_ts_names,
                          "descriptions": self.output_ts_descriptions,
                          "units": self.output_ts_units}
        self.output_ts_agg_method=output_ts_agg_method
        self.output_ts_agg_freq = output_ts_agg_freq
        self.dataset_id = dataset_id
        self.name_of_schedule = name_of_schedule
        self.name_of_function = name_of_function
        self.cron_interval_in_minutes = cron_interval_in_minutes
        self.backfill_period = backfill_period
        self.backfill_hour = backfill_hour
        self.backfill_min_start = backfill_min_start
        self.historic_start_time = historic_start_time
        self.deployment_single_call = deployment_single_call
        self.deployment_scheduled_call = deployment_scheduled_call

        self.data_dict = {}

        self.function_path = Path().cwd().joinpath(self.name_of_function, "function")

    def validate(self) -> None:
        required_args = [
                self.input_ts_names,
                self.client,
                self.output_ts_names,
                self.dataset_id,
                self.name_of_schedule
            ]

        newline = "\n"
        assert not any(
            [var is None for var in required_args]
        ), f"You need to pass valid values to these args: {newline.join(required_args)}"

    def deploy_cf(self) -> None:
        self.validate()

        cf = self.client.functions.retrieve(external_id=self.name_of_function)
        if cf is None:
            zip_path = self.function_path.joinpath("function.zip")

            try:
                with zipfile.ZipFile(zip_path, "w") as f:
                    f.write(self.function_path.joinpath("requirements.txt"), arcname="requirements.txt")
                    f.write(self.function_path.joinpath("handler.py"), arcname="handler.py")
                    f.write(self.function_path.joinpath("transformation.py"), arcname="transformation.py")
                    f.write(self.function_path.parent.joinpath("prepare_timeseries.py"), arcname="prepare_timeseries.py")
                    f.write(self.function_path.parent.joinpath("transform_timeseries.py"), arcname="transform_timeseries.py")
                    f.write(self.function_path.parent.joinpath("utilities.py"), arcname="utilities.py")
            except FileNotFoundError:
                msg = """The following three files are required to deploy:
requirements.txt
handler.py
transformation.py
"""
                raise FileNotFoundError(msg)

            uploaded = self.client.files.upload(
                path=zip_path,
                name=zip_path.stem,
                data_set_id=self.dataset_id
            )

            self.client.functions.create(
                name=self.name_of_function,
                external_id=self.name_of_function,
                file_id=uploaded.id,
                runtime="py311",
                folder="function"
            )

            cf = self.client.functions.retrieve(external_id=self.name_of_function)
            print("Cognite Function created. Waiting for deployment status to be ready...")
            while cf.status != "Ready":
                time.sleep(3)
                cf.update()
            print("Ready for deployment.")

        if self.deployment_single_call:
            print("Calling Cognite Function individually.")
            cf.call(data=self.data_dict)
            print("Done")

        if self.deployment_scheduled_call:
            print("Preparing schedule to start sharp at next minute...", end="\r")

            now = pd.Timestamp.now(tz="CET").floor("1s").tz_convert("UTC")
            while now.second > 0:
                time.sleep(1)
                now = pd.Timestamp.now(tz="CET").floor("1s").tz_convert("UTC")

            print(f"Setting up Cognite Function at time {now}...", end="\r")

            self.client.functions.schedules.create(
                name=self.name_of_schedule,
                cron_expression=f"*/{self.cron_interval_in_minutes} * * * *",
                function_id=cf.id,
                client_credentials=ClientCredentials(
                    client_id=os.getenv("CLIENT_ID"),
                    client_secret=os.getenv("CLIENT_SECRET")
                ),
                description=f"Calculation scheduled every {self.cron_interval_in_minutes} minutes",
                data=self.data_dict
            )

            print("Done")
            return

def make_new_cf_structure(name: str) -> None:
    root_path = Path().cwd()
    function_path = root_path.joinpath(name)
    template_path = Path(__file__).parent.joinpath("template_dir")

    if not function_path.exists():
        shutil.copytree(template_path, function_path)
    else:
        raise RuntimeError(f"Cognite Function directory '{name}' already exists!")

    print(f"New Cognite Function Directory Created:\n")
    tree = f"""{name}/
├── Deployment.ipynb          <-- Utility notebook for setting up the deployment
└── function                  <-- The directory that will be uploaded to CDF
    ├── handler.py            <-- Entry point to your code for Cognite Functions
    ├── requirements.txt      <-- All runtime python dependencies go here
    └── transformation.py     <-- Separate file where you define the transformation
"""
    print(tree)
    return

def cli():
    """"""
    description = "Cognite Functions Template CLI"
    epilog = """This CLI provides an entry point to the
Cognite Functions Template.
"""
    parser = argparse.ArgumentParser(
        description=description,
        add_help=True,
        epilog=epilog
    )

    parser.add_argument("-n", "--name", type=str, help="Name of the new cognite function project.")

    args = parser.parse_args()

    if " " in args.name:
        cf_name = "_".join(args.name.split()).lower()
    else:
        cf_name = args.name

    make_new_cf_structure(cf_name)
