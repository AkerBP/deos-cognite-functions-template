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

## Structure of project
The source code is located in the folder `time_series_calculation`.

```markdown
├── time_series_backend
|   ├── prepare_timeseries.py
|   ├── transform_timeseries.py
│   └── utilities.py
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
