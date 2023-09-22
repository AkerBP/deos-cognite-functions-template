# opshub-task1
## Introduction
Some tanks on Aker BPs assets are missing draining rate measurements. Draining rate is valuable data to detect leakages from tanks.
The goal of this project is to transform original time series data of fluid volume percentage to drainage rate from the tanks.
The new time series will be published as a new dataset in the Cognite Fusion Prod tenant and deployed in Grafana dashboards for quick analysis by the end-user.

The project seeks to demonstrate how one goes by acquiring read/write access for CDF datasets, and how to use Cognite Functions from the Python SDK to read, transform and write datasets for CDF. We detail the necessities for the three distinct phases of this process; development, testing and production. The project follows Microsoft's recommended template for Python projects: [https://github.com/microsoft/python-package-template/].

## Getting started
1. Clone the repository using git and move to the cloned directory
```
git clone https://github.com/vetlenev/opshub-task1.git
cd opshub-task1
```
2. Create a virtual environment using conda
```
conda create -n myenv
conda activate myenv
```
- The installation includes Cognite's Python SDK `cognite-sdk` (version 6.15.3), used to perform transformations for CDF directly through Python
- For advanced management of Python virtual environments, `poetry` is recommended for the installation. See (https://github.com/cognitedata/using-cognite-python-sdk) for more details

4. Authentication with Python SDK.
- First, create a user (or sign into your existing) account at (Cognite Hub)[https://hub.cognite.com/]. This will connect you to an Azure Active Directory tenant that is used to authenticate with the Cognite Fusion Prod tenant, which gives you read access to the time series dataset used in this project. All Aker BP accounts and Aker BP guest accounts have by default access to the development environment of CDF (Cognite Fusion Dev).
- To authenticate with the Cognite API we use the `OAuthClientCredentials` credentials provider. Authentication is done in the Jupyter file `run_functions.ipynb`. Four parameters must be specified:
  1. `TENANT_ID`: ID of the Azure AD tenant where the user is signed in (here: `3b7e4170-8348-4aa4-bfae-06a3e1867469`)
  2. `CLIENT_ID`: ID of the application in Azure AD (here: `779f2b3b-b599-401a-96aa-48bd29132a27`)
  3. `CDF_CLUSTER`: Cluster where your CDF project is installed (here: `api`)
  4. `COGNITE_PROJECT`: Name of CDF project (here: `akerbp`)
- With these, credentials are provided by
```
credentials = OAuthClientCredentials(
    token_url=f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token",
    client_id=CLIENT_ID,
    client_secret=??????????????????????????
    scopes=[f"https://{CDF_CLUSTER}.cognitedata.com/.default"],
)
```
- The client is configured as follows
```
config = ClientConfig(
    client_name="Cognite Academy course taker",
    project=COGNITE_PROJECT,
    base_url=f"https://{CDF_CLUSTER}.cognitedata.com",
    credentials=credentials,
)
```
- To instantiate a Cognite `client`, run `client = CogniteClient(config)`. Functionality of Python SDK can now be accessed through this client
- For an overview of read/write accesses granted for different resources and projects, see `client.iam.token.inspect()`

## Testing
The integrity and quality of the data product is tested using several approaches. 
- A framework for unit testing is found in the folder `tests`
- User Acceptance Testing (UaT), including plan and test scenarios, have been performed and are documented in the file `docs/development/SIT-UaT-Test`
- System Integration Testing (SIT) is not applicable for this project, because we are not using any external extractors or APIs for data processing

## Improvements for access request system
Completing all steps in this demonstration, from retrieving the original time series to writing the new time series back to CDF Prod, unfortunately takes an undeseriably long time and is subject to efficiency improvements. The main bottleneck is the process of granting necessary read and write accesses for CDF. 
- The form for requesting access is more comprehensive than necessary. It is not trivial what to fill out in some sections. Thus, we believe too much time is wasted mailing the CDF Operations team back and fourth for particular guidance. This process has potential for streamlining by, e.g., offering standard priviliges or prefilled forms tailored for particular work domains. For instance, propose a specific read/write access for data scientists satisfying their general work scope, facilitating automated request processing
- If you submit a form with the same title as another submitted form, it is considered as a duplicate and will be deleted. Hence, if you fill out something wrong and have to resubmit the form, it is crucial to rename the title. We think this issue should be communicated better by the CDF Ops team to avoid users waiting forever for their submission to be processed.
- The CDF Operations team is by the time of writing (September 2023) understaffed, where a response to your request form is expected to take multiple days, or up to a week. This is not sustainable for a company like Aker BP with lots of employees developing their work scope. The CDF Operations team needs expansion of their staff.

## Architecture Design Documentation
1. **Document Objective**
- This documentation aims at describing the process of integrating a new time series with a new dataset in CDF, including extraction from Cognite CLEAN, transformations using Cognite Functions in the Python SDK, and contextualization through a CDF resource model
- Procedures for granting access requests for reading data from Cognite Fusion Prod and writing data to Cognite Fusion Dev are also provided
2. **Problem Description**
- The goal is to create a new time series of daily average drainage rate from tanks
- Transformations are performed on an existing time series in CDF, which serves data of volume percentage of oil in tanks
- Cognite Functions are defined and called through a Jupyter file that creates the new time series and populates it with transformed data
- The new time series is assigned to a new, governed dataset
3. **General Graphical Overview**
- The figure below depicts the processes and systems involved (--- show figure from task description ---).
4. **Data Extraction**
- Data is extracted from the CLEAN storage in Cognite Fusion Prod as a time series CDF resource
- Here, the name of the time series is `ts_input_name="VAL_11-LT-95034A:X.VALUE"`, associated with the *PI time series* dataset
- The relevant time series is found by calling
  `ts = client.time_series.search(name=ts_input_name)`
- The time series can then be loaded by retrieving its external id:
  `client.time_series.data.retrieve(external_id=ts[0].external_id, aggegrated="average", granularity="1m", start=start_date, end=end_date)`
  Through additional arguments, we aggregate the average signal per minute between given start and end dates
5. **Transformations**
- Transformations are performed with Cognite Functions through (Cognite Python SDK)[https://cognite-sdk-python.readthedocs-hosted.com/en/latest/]
- The extracted time series is converted to a `pandas` DataFrame, `df`
- Then, the signal of volume percentage (`vol_perc`) is filtered using a Lowess smoother from the `statsmodels.nonparametric.smoothers_lowess` module:
  `smooth = lowess(vol_perc, df["time_sec"], is_sorted=True, frac=0.01, it=0)`,
   where `df["time_sec"]` is the time in seconds between successive data points. Please consult the (documentation)[https://github.com/statsmodels/statsmodels/blob/main/statsmodels/nonparametric/smoothers_lowess.py] for explanation of additional arguments
- Drainage rate is calculated as the derivative of the smoothed signal:
  `df[dXdt] = numpy.gradient(df["smooth"], df["time_sec"])`
- We are not interested in large, positive derivatives as these indicate filling of tank (human interventions). These are therefore truncated from the signal, and missing values are replaced by zeros:
  `df["dXdt_excl_filling"] = df[dXdt].apply(lambda x: 0 if x > alfa or pd.isna(x) else x)`,
  where `alfa=0.002` is the threshold value for truncation
- To get the daily average drainage, we group the data by date, aggregate by mean and convert from percentage to volume, i.e,
  `avg_drainage_day = df.groupby("Date")["dXdt_excl_filling"].mean()*tank_volume/100`
- Create a new instance of a time series CDF resource through
  `client.time_series.create(TimeSeries(name=ts_output_name, external_id=ts_output_name, data_set_id=ds_id))`
  where `ts_output_name = "VAL_11-LT-95034A:X.CDF.D.AVG.LeakValue"` is the name of the new time series. Data lineage and integrity is ensured by associating the time series to the dataset with id `ds_id`
- Populate the new time series with data of daily average drainage:
  `client.time_series.data.insert_dataframe(pd.DataFrame({ts_output_name: avg_drainage_day}))`. The data is inserted into the correct time series by specifying the name of the time series as header for the inputted dataframe
6. **Access control**
- To deploy data to the Cognite Fusion Dev evnironment, we need to submit an access request form for Cognite Data Fusion. Subsection 6a. describes how to fill out this form in order to get the necessary access rights for developing our dataset, and subsection 6b details how to authenticate with a Cognite Client.
  a. ***Read/write access request***
    - The form is found (here)[https://forms.office.com/Pages/ResponsePage.aspx?id=cEF-O0iDpEq_rgaj4YZ0aUVYsXTN0c9Dil0iHGZgj0lUOTBXVFlSWDlMUFk1WUNBS1lKWjZKWko2TyQlQCN0PWcu]
    - Set a unique *Title of the request* 
    - In *Area of the request* select Cognite Data Fusion (CDF)
    - In *Category* select New access or create new dataset
    - In *New access or create new dataset request* select Create new dataset
    - In *Justification of dataset*, mention that calculation of drainage rate will help detecting leakages
    - NOE MER
  
## Deployment Plan
1. **Deployment Scope**
- The scope of the deployment is to make the dataset continuously available in CDF to Aker BP employees in general, and the DEOS team in particular, for streamlined root cause analysis
2. **Premisses**
- The premisses for the deployment is increased value outtake in the form of increased uptime of assets. The dataset is projected to serve valuable data of leakage events from tanks for improved decision making
3. **Deployment Timeline**




## Contributing

This project welcomes contributions and suggestions. For details, visit the repository's [Contributor License Agreement (CLA)](https://cla.opensource.microsoft.com) and [Code of Conduct](https://opensource.microsoft.com/codeofconduct/) pages.
