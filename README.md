# Template for deployment of Cognite Functions for time series calculations
## Introduction
For a concise guide for deployment of Cognite Functions, see [Cognite-Function-Demonstration](https://github.com/Aker-BP-OpsHub/Cognite-Function-Demonstration).

This project provides a template for using Cognite Functions as a tool for transforming and deploying time series to Cognite Data Fusion (CDF) for real-time analytics. Using Cognite's Python SDK, the framework supports transformations of single and multiple time series residing in CDF. Python has tons of libraries to satisfy your calculation setup. The idea is to automate and streamline the process of running semi-advanced calculations on time series. Cognite Charts already provides a user-friendly interface for performing basic to semi-advanced calculations, but has a quite limited set of available calculations. This Cognite Function Template is designed to be more versatile in the sense that you can import any data science model or algorithm available in the open-source Python library, providing endless possibilities for calculations.

The transformed time series is frequently and automatically updated by letting the Cognite Function run on a prescribed schedule. The setup also facilitates backfilling for quality assurance with the original signal. 
The new time series is published to the [Center of Excellence - Analytics](https://akerbp.fusion.cognite.com/akerbp-dev/data-sets/data-set/1832663593546318?cluster=api.cognitedata.com) dataset in CDF and can be analysed by SMEs and end-users in desired visualization tools connected to CDF.

We further detail how one goes by acquiring read/write access for CDF resources and dataset, and how to use Cognite Functions from the Python SDK to read, transform and write datasets for CDF. We detail the necessities for the three distinct phases of this process; development, testing and production. The project follows Microsoft's recommended template for Python projects: [https://github.com/microsoft/python-package-template/]. The repository is organized as follows (standard template files in parent folder are omitted).
```markdown
├── data
├── docs
|   ├── dev
|   ├── test
|   ├── prod
├── src
├── tests
├── authentication-data.env
└── handler_data.env
```
Here, `src` is the main "hub" for creating and deploying Cognite Functions, `tests` provides scripts for unit tests and UaT tests, and in `docs/development` you find detailed documentation of project development and deployment. Your tenant and client IDs and secrets used to authenticate should be stored in `authentication-data.env`.

**Disclaimer:**  In this project we assume that the time series data residing in CDF is of high quality, in alignment with data from the source system, although we are aware that the compression rate of signals in the source system and CDF may differ, which may cause differences in the data from CDF and the source system when aggregating the signals.

## Getting started
This section details the necessary steps to get ready for deployment of Cognite Functions using this project's template. To this end, we set up a virtual environment using the Poetry package manager.

1. Clone the repository using git and move to the cloned directory
```
git clone https://github.com/AkerBP-DataOps/deos-cognite-functions-template.git
cd deos-cognite-functions-template
```
2. Make sure to have [poetry](https://python-poetry.org/docs/) installed.
3. Set the location of the virtual environment to be inside the project repository
```
poetry config virtualenvs.in-project true
````
4. Install dependencies specified in `pyproject.toml`
```
poetry install
```
5. Activate the poetry environment
```
.venv/Scripts/activate
```
and select `.venv` as kernel in the interactive script `src/run_functions.ipynb`

## Authentication with Python SDK.
To read/write data from/to CDF, including Cognite Functions, you need to apply for read and write access to the relevant data resources. This is done by submitting a request using the following (form)[https://forms.office.com/Pages/ResponsePage.aspx?id=cEF-O0iDpEq_rgaj4YZ0aUVYsXTN0c9Dil0iHGZgj0lUOTBXVFlSWDlMUFk1WUNBS1lKWjZKWko2TyQlQCN0PWcu]. As a minimum to use this template, you need the following resource accesses: Functions:READ and Functions:WRITE.

Once access has been granted, we need to connect with the Cognite application. This section describes the process of authenticating with a Cognite client using client credentials.
- Create a user (or sign into your existing) account at [Cognite Hub](https://hub.cognite.com/). This will connect you to an Azure Active Directory tenant that is used to authenticate with CDF, which gives you read access to the time series dataset used in this project. All Aker BP accounts and guest accounts have by default access to the development environment of CDF (Cognite Fusion Dev).
- **NB: To deploy Cognite Functions on schedules, interactive authentication using the `OAuthInteractive` provider does no work!** Instead, you need to authenticate using a `OAuthClientCredentials` with a personal client secret. Authentication is done in `initialize.py`. Five parameters must be specified:
  1. `TENANT_ID`: ID of the Azure AD tenant where the user is signed in (default: `3b7e4170-8348-4aa4-bfae-06a3e1867469`)
  2. `CLIENT_ID`: ID of the application in Azure AD. This will be unique value available to users that have been granted write access to the dataset. It is found in your "Key vaults > *key_vault_name* > Secrets" service at [Microsoft Azure](https://portal.azure.com/#home) (reach out to the CDF team if you don't know the exact *key_vault_name*), where the relevant key has suffix ending "-ID"
  3. `CDF_CLUSTER`: Cluster where your CDF project is installed (here: `api`)
  4. `COGNITE_PROJECT`: Name of CDF project (here: `akerbp`)
  5. `CLIENT_SECRET`: A secret token required for deployement. This is found in your "Key vaults > Secrets" service at [Microsoft Azure](https://portal.azure.com/#home) where the relevant key has suffix ending "-SECRET"
The `TENANT_ID`, `CLIENT_ID` and `CLIENT_SECRET` must be defined in a file `authentication-ids.env` in the root folder of your project.

## Structure of project
The source code is located in the folder `time_series_calculation`.

```markdown
├── src
|   ├── README.md
|   ├── __init__.py
│   ├── CoEA_avg-drainage
│   │   ├── requirements.txt
│   │   ├── handler.py
│   │   ├── transformation.py
│   │   ├── poetry.lock
│   │   ├── pyproject.toml
│   │   ├── zip_handle.zip
│   ├── *dsA*_*func-X*
│   │   ├── requirements.txt
│   │   ├── handler.py
│   │   ├── transformation.py
│   │   ├── poetry.lock
│   │   ├── pyproject.toml
│   │   ├── zip_handle.zip
│   ├── *dsB*_*func-Y*
│   │   ├── ...
│   ├── handler_utils.py
│   ├── transformation_utils.py
│   ├── initialize.py
│   ├── generate_cf.py
│   ├── deploy_cognite_functions.py
│   ├── utilities.py
│   └── run_functions.ipynb
```
Here we find a script `prepare_timeseries.py` that takes an input time series and preprocesses it to make it ready for calculation. It also handles backfilling. A script `transform_timeseries.py` is devoted for the actual transfoormation of the time series. Utility functions are found in `utilities.py`. There are two main classes of the template:

**`PrepareTimeseries`** (in `prepare_timeseries.py`)
- Prepares one or a set of time series for transformation by retrieving and aligning input signals over a populated datetime range, handling NaNs, support for aggregations and backfilling
- Time series are collectively retrieved, structured and backfilled by the method `get_orig_timeseries`, importing functionality from other methods in the same class
- Transformed time series are eventually written to a devoted time series object in CDF

**`TransformationTimeseries`** (in `transform_timeseries.py`)
- Class that takes an organized dataframe of time series inputs (as prepared by `PrepareTimeseries`) and transforms the signals acording to the transformation defined in `transformation.py`

The only programmatic modification required by the end-user is setting up your calculation in the `transformation` function in `transformation.py`, where the function should return a pandas dataframe satisfying the following:
  - must have a pandas datetime index representing the timestamp for each value
  - the columns should be set to the output names defined in the data dictionary, i.e., `data["ts_output"].keys()`
If additional python packages are used in `transformation.py`, remember to include them in `requirements.txt`.


## The deployment process
This section details how the template is set up to deploy Cognite Functions.

### 1. Generate Cognite Function folder structure
The first step to deploy a Cognite Function is to create a folder structure to "host" it. A new Cognite Function `myname` is instantiated by running the function `generate_cf(cf_name, add_packages)` from the script `generate_cf.py`, where `cf_name` is the name of our Cognite Function, i.e., `cf_name=*myname*`, and `add_packages` specifies additional packages required to perform transformations defined in `transformation.py`. 

In addition to generating necessary scripts for deployment, `generate_cf.py` also sets up a dedicated Poetry environment for this function, including autogeneration of `requirements.txt`.  **NB: Internal consistency of `requirements.txt` is not guaranteed through this approach, so you may need to manually check for consistency in case of errors calling the Cognite Function.**
### 2. Create file
Next, we create a Cognite File to link with our Cognite Function. The file must point to a zip file `zip_handle.zip` in the subfolder `cf_*myname*` designated for the Cognite Function with name `myname`. The zip file contains the main entry `handler.py` with a function named `handle` inside it, and other necessary files to run `handler.py` (here: `requirements.txt`, `handler_utils.py` and `transformation.py`)
```
folder = os.getcwd().replace("\\", "/")
folder_cf = folder + "/" + data_dict["function_name"]
zip_name = "zip_handle.zip"

uploaded = client.files.upload(path=f"{folder_cf}/{zip_name}", name=zip_name, data_set_id=dataset_id)
```
The Cognite File is associated with a dataset with id `dataset_id` and uploaded to CDF.
### 3. Deployment
The next step is to create an instance of the `handle` function (located in the subfolder `*ds*_*func*`) to be deployed to CDF. 
```
client.functions.create(
    name=f"{data_dict['function_name']}",
    external_id=f"{data_dict['function_name']}",
    file_id=uploaded.id,
)
```
The `file_id` is assigned the id of the newly created zip file.
### 4. Set up schedule
Once deployed, we set up a schedule for our function. For example, if we want out Cognite Function to run every 15 minute, this is specified by setting `data_dict["cron_interval_min"] = str(15)`. If the output time series does not exist, the first call will transform all historic data points. This will likely be a lot of data, and since Cognite Functions have a limited runtime of 10-15 minutes, you have two options if the Cognite Function is unable to perform the initial transformation:  
- A.  Specify the start date for retrieving historic data through the optional argument "historic_start_time", which will overwrite the true historic start date of the data
- B.  Perform initial transformation locally by running `handle(client, data)` with instantiated Cognite `client` and `data` dictionary *with same input parameters as used to set up schedule*

The schedule is instantiated by
```
client.functions.schedules.create(
    name=f"{data_dict['schedule_name']}",
    cron_expression=f"*/{data_dict['cron_interval_min']} * * * *", # every cron_interval min
    function_id=cognite_function.id,
    client_credentials=ClientCredentials(client_id=str(os.getenv("CLIENT_ID")),
                                         client_secret=str(os.getenv("CLIENT_SECRET"))),
    description=f"Calculation scheduled every {data_dict['cron_interval_min']} minute",
    data=data_dict,
)
```
**Steps 2-4 above are collectively run by the function `deploy_cognite_functions.py`.**
### 5. Repeat calculation for other time series'
If you want to run the same calculation for another time series input, you simply create a new schedule for the same Cognite Function. To do so, in `run_functions.ipynb` modify `data_dict` with desired parameters (important to set a unique name of the schedule with the `schedule_name` key). Since the Cognite Function is already generated, you can skip running `generate_cf` and jump straight to running the initial transformation followed by setup of schedule by running `deploy_cognite_functions(data_dict, client, single_call=False, scheduled_call=True)`. You can also make a single call to the Cognite Function, without setting up a schedule, by running `deploy_cognite_functions(data_dict, client, single_call=True, scheduled_call=False)`.

## Testing
The integrity and quality of the data product is tested using several approaches. The `tests` folder represents the testing framework applied in the CDF test environment, and contains the following
```markdown
├── tests
|   ├── __init__.py
|   ├── conftest.py
|   ├── test_methods.py
|   └── utils.py
```
where `conftest.py` sets up necessary configurations of the tests, and `test_methods.py` contains the actual unit tests and UaTs. Test scenarios and results for the latter are documented in the file `docs/development/SIT-UaT-Test.xlsx`.

## Cognite Functions Template or Cognite Charts?
Although this template automates many of the sequential steps in the deployment process, it still requires the end-user to write his/her calculation in a Python script, specify parameters and deploy in a coding framework. We are looking into an alternative approach for transforming time series that uses the Cognite Charts user interface, hence reducing the level of abstraction. There are multiple functions already available in Charts, written in the `indsl` Python library, but if you are missing a desired calculation, it is possible to add it to the Charts environment. The idea is to let someone with sufficient coding knowledge write the calculation in the `indsl` package and have the Cognite team validate and deploy the function to Charts. Once deployed, the calculation is available for use by everyone. Thus, one can easily apply the calculation on any time series using use-friendly drag and drop functionality, avoiding having to rerun a set of code cells to deploy a new Cognite Function. We believe this approach will appeal more to SMEs. Work is needed, though, to map the feasibility (or tediousness) of contributing with own calculations in the `indsl` library.

On the other hand, there are also areas where the Cognite Functions Template is superior at the moment.
1. Aggregated functions
- Scheduled calculations in Charts are not consistent. The aggregated result depends on the scheduled period and when the scheduled calculation is intially called. The result from the last scheduled call that covers datapoints in the aggregated period will overwrite results from previous scheduled calls that also covers the aggregated period. The ultimate, correct aggregated result would be to assemble all individual aggregates over the aggregated period. This functionality, more specifically date-specific aggregates, is facilitated in the implementation of the Cognite Functions Template
2. Flexibility and adaptability
- Functionality of Charts is limited to the functions residing in the user interface. Although this usually covers most necessities for time series analysis, the Cognite Functions Template is more flexible and customizable in the sense that it facilitates problem-tailored calculations - only limited by the runtime capacity of Cognite Functions schedules
3. Time to production
- While you can contribute with new functions to the `indsl` library in Charts, the deployment has to go through pull requests with the Cognite team, delaying the time for performing time series analysis. With the Cognite Functions Template, the plan is that calculations on time series from a governed dataset is seamlessly validated by the associated domain/team. We believe this internal deployment process is more transparent and much faster than going through the Cognite team.

In the future, it might be possible to generate own Charts environments isolated from the "global" Charts environment (https://hub.cognite.com/developer-and-user-community-134/isolated-calculations-in-charts-2568). This will allow teams in your organization generate calculations that are specific for your environment without overwhelming or "polluting" the global Charts.
