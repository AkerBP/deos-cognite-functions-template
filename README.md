# opshub-task1
## Introduction
Some tanks on Aker BPs assets are missing draining rate measurements. Draining rate is valuable data to detect leakages from tanks.
The goal of this project is to transform original time series data of fluid volume percentage to drainage rate from the tanks using Cognite Functions. The new time series is computed with a granularity of 15 minutes. For this reason, the Cognite Function is set to run on a 15 minute schedule. 
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
3. Install packages (conda manages the dependencies)
```
conda install -c conda-forge numpy statsmodels matplotlib python-dotenv msal ipykernel
pip install "cognite-sdk[pandas, numpy]"
```
- The `cognite-sdk` package is used to perform transformations for CDF directly through Python. The package supports integrated functionality with `pandas` for data structuring, and `numpy` for vectorization and performance boosts. These are therefore specified as dependencies inside brackets.
- To deploy Cognite Functions, the main entry point, `handler.py`, is supported by a `requirements.txt` file located in the same folder
- *If your virtual environment includes other packages not used by `handler.py`, we recommend using `pipreqs` to ensure consistency with the `requirements.txt` file*
```
pip install pipreqs
pipreqs src
```
- For advanced management of Python virtual environments, `poetry` is recommended for the installation. See (https://github.com/cognitedata/using-cognite-python-sdk) for more details
4. Deploy and run the Cognite Function
- The jupyter file `src/run_functions.ipynb is devoted to creating and executing the Cognite Function
- Input data to the `handle` function in `handler.py` is provided by the `data_dict` dictionary. If you create your own Cognite Function, make sure to change the key-value pairs to fit your purpose
- Run the code cells consequtively authenticate with CDF and deploy the Cognite Function at a schedule for given input data

## Authentication with Python SDK.
To read/write data from/to CDF, we need to connect with the Cognite application. This section describes the process of authenticating with a Cognite client using app registration and the OIDC protocol. The complete code for authenticating is found in `src/cognite_authentication.py`
- Create a user (or sign into your existing) account at (Cognite Hub)[https://hub.cognite.com/]. This will connect you to an Azure Active Directory tenant that is used to authenticate with CDF, which gives you read access to the time series dataset used in this project. All Aker BP accounts and guest accounts have by default access to the development environment of CDF (Cognite Fusion Dev).
- To authenticate with the Cognite API we generate a client credential, more specifically a `OAuthClientCredentials`. Authentication is done in `src/initialize.py`. Five parameters must be specified:
  1. `TENANT_ID`: ID of the Azure AD tenant where the user is signed in (here: `3b7e4170-8348-4aa4-bfae-06a3e1867469`)
  2. `CLIENT_ID`: ID of the application in Azure AD (here: `779f2b3b-b599-401a-96aa-48bd29132a27`)
  3. `CDF_CLUSTER`: Cluster where your CDF project is installed (here: `api`)
  4. `COGNITE_PROJECT`: Name of CDF project (here: `akerbp`)
  5. `CLIENT SECRET`: A secret token required for deployement. GIVE MORE INFO ...
- With these, we can authenticate by fetching our credentials
```
creds = OAuthClientCredentials(
          token_url=AUTHORITY_URI + "/oauth2/v2.0/token",
          client_id=sCLIENT_ID,
          scopes=SCOPES,
          client_secret=str(os.getenv("CLIENT_SECRET")),
      )
```
- The client is configured as follows (where `GET_TOKEN` is the access token acquired by the client `app`)
```
config = ClientConfig(
    client_name="my-client-name",
    project=COGNITE_PROJECT,
    credentials=Token(GET_TOKEN),
    base_url=f"https://{CDF_CLUSTER}.cognitedata.com"
)
```
- Your Cognite client is instantiated by running `client = CogniteClient(config)`
- For an overview of read/write accesses granted for different resources and projects, see `client.iam.token.inspect()`

## Calculation of drainage rate
This section gives the mathematical and technical details how to calculate drainage rate from a time series of volume percentage. If you are only interested in deployment of Cognite Functions, we recommend jumping to section Update time series at prescribed schedules.

Drainage rate is the amount of a fluid entering/leaving the tank, here given in units of [L/min]. The input signal is sampled with a granularity of one minute. To denoise the signal, we perform `lowess` filtering using the`statsmodels` Python package. It computes locally weighted fits, which can be quite time consuming for many datapoints. Since our initial write to the dataset spans all historic data, there are potentially a lot of computations. From our experiments, filtering a 1-minute granularity signal over three years takes around 30 minutes. It is possible to reduce computations by adjusting the `delta` parameter, which specifies the threshold interval between datapoint for which to substitute weighted regression with linear regression. Setting `delta=5e-4*len(vol_perc)` reduces runtime to about 90 seconds. Apart from the initial write, filtering is only performed on datapoints from the most recent date. This allow us to rely entirely on weighted regression, i.e., `delta=0`. We use 1% of the datapoints to compute the local regression at a particular point in `time`. Lowess filtering is run by calling
```
smooth = lowess(vol_perc, time, is_sorted=True,
                frac=0.01, it=0, delta=delta)
```

After filtering, the drainage rate is calculated as the derivative of the volume percentage. For this, we use `numpy`s `gradient` function, operating on the full vector of datapoints.
```
drainage_rate = np.gradient(smooth, time)
```
To get the daily average leakage rate, we group the data by date, calculate the mean value for each date. To get in units of [L/min] we multiply by tank volume and divide by 100.

## Deployment of Cognite Function and scheduling
This section outlines the procedure for creating a Cognite function for CDF, deployment and scheduling using Cognite's Python SDK. The code snippets are found in `run_functions.ipynb`.

*A client secret is required to deploy the function to CDF. This means that we need to authenticate with a Cognite client using app registration (see section Authentication with Python SDK), **not** through interactive login. This requirement is not yet specified in the documentation from Cognite. The message of improving their documentation of Cognite functions has been conveyed to the CDF team to hopefully resolve any confusions regarding deployment.*

### 1. Create file
To successfully write a Cognite function to CDF, we first need to create a Cognite file scoped to the dataset (with id `dataset_id`) that our function is associated with. The file must point to the path where your Cognite function is located. The function must be named `handle` and placed in a `handler.py` file.
```
folder = os.getcwd().replace("\\", "/")
function_path = "handler.py"

uploaded = client.files.upload(path=f"{folder}/{function_path}", name=function_path, data_set_id=dataset_id)
```
### 2. Deployment
The next step is to create an instance of the `handle` function to be deployed to CDF.
```
func_drainage = client.functions.create(
    name="avg-drainage-rate",
    external_id="avg-drainage-rate",
    file_id=uploaded.id,
)
```
The `file_id` is assigned the id of the newly created file. 
### 3. Set up schedule
Finally, we set up a schedule for our function. Here, we want the function to run every 15 minutes. This is specified using the cron expression `*/15 * * * *`. The function receives necessary input data `data_dict` through the `data` argument. The schedule is instantiated by
```
func_drainage_schedule = client.functions.schedules.create(
    name="avg-drainage-rate-schedule",
    cron_expression="*/15 * * * *", # every 15 min
    function_id=func_drainage.id, # id of function instance
    description="Calculation scheduled every hour",
    data=data_dict
)
```
Firstly, we check if the transformed time series already exists. If not, we create the time series (code snippets from `handler.py`)
```
ts_leak = client.time_series.list(name=ts_output_name).to_pandas()  # transformed time series
if ts_leak.empty:
    client.time_series.create(
        TimeSeries(name=ts_output_name, external_id=ts_output_name, data_set_id=dataset_id)
    )
```
To avoid redundant work, we only query and transform parts of the original time series from the current date. To not put too much pressure on the system, we aggregate the signal with 1 minute averages.
```
end_date = pd.Timestamp.now()
start_date = pd.to_datetime(end_date.date())
ts_orig = client.time_series.data.retrieve(external_id=ts_orig_extid,
                                               aggregates="average",
                                               granularity="1m",
                                               start=start_date,
                                               end=end_date)
```
The units of the calculated daily average drainage rate is thus in [L/min]. The calculations from `start` to `end` are inserted into the transformed time series 
```
client.time_series.data.insert_dataframe(mean_df)
```
where `mean_df` is a dataframe with calculated daily average leakage from current date.

## Testing
The integrity and quality of the data product is tested using several approaches. 
- A framework for unit testing is found in the folder `tests`
- User Acceptance Testing (UaT), including plan and test scenarios, have been performed and are documented in the file `docs/development/SIT-UaT-Test`
- System Integration Testing (SIT) is not applicable for this project, because we are not using any external extractors or APIs for data processing

## Presentation in Grafana
- To facilitate an insightful presentation of the transformed time series, we deploy it to the Grafana appliation. Through an Aker BP tenant, Grafana seamlessly connects to CDF as a source system. Once data is deployed to CDF, an API interface handles the ingestion of data into Grafana.
- The time series is to be part of a pump health dashboard containing all kind of insightful data of the "health" of a particular pump. The panels of this dashboard serve the relevant data in a visually pleasing way. We generate a new panel for this dashboard that visualizes our transformed time series, along with other data visualizations of the pump, and the data is continuously updated along with new data that enters CDF.

## Improvements for access request system
Completing all steps in this demonstration, from retrieving the original time series to writing the new time series back to CDF Prod, unfortunately takes an undeseriably long time and is subject to efficiency improvements. The main bottleneck is the process of granting necessary read and write accesses for CDF. 
- The form for requesting access is more comprehensive than necessary. It is not trivial what to fill out in some sections. Thus, we believe too much time is wasted mailing the CDF Operations team back and fourth for particular guidance. This process has potential for streamlining by transitioning from restriction-based to constraint-based, facilitating a more rapid onboarding process for new developers
- If you submit a form with the same title as another submitted form, it is considered as a duplicate and will be deleted. Hence, if you fill out something wrong and have to resubmit the form, it is crucial to rename the title. We think this issue should be communicated better by the CDF Ops team to avoid users waiting forever for their submission to be processed
- The CDF Operations team is by the time of writing (September 2023) understaffed, where a response to your request form is expected to take multiple days, or up to a week. This is not sustainable for a company like Aker BP with lots of employees developing their work scope
