# Template for deployment of Cognite Functions for time series calculations
## Introduction
This project provides a template for a workflow using Cognite Functions to run transformations of time series. The framework generalizes to arbitrary calculations of single or multiple time series. The idea is to facilitate and lower the threshold for end-users and SMEs with minor knowledge in data science to easily set up their own calculations on desired time series for quick insight, and hopefully acknowledge the potential and efficiency of using our workflow with Cognite Functions. For a concise guide for deployment of Cognite Functions, see [Cognite-Function-Demonstration](https://github.com/Aker-BP-OpsHub/Cognite-Function-Demonstration).

We demonstrate the framework by transforming a time series of fluid volume percentage to a new time series of daily average drainage rate from the tanks. The new time series is frequently and automatically updated by letting the Cognite Function run on a prescribed schedule. The setup also facilitates backfilling for quality assurance of the new signal. 
The new time series will be published as a new dataset in the Cognite Fusion Prod tenant and deployed in Grafana dashboards for quick analysis by the end-user.

We further detail how one goes by acquiring read/write access for CDF datasets, and how to use Cognite Functions from the Python SDK to read, transform and write datasets for CDF. We detail the necessities for the three distinct phases of this process; development, testing and production. The project follows Microsoft's recommended template for Python projects: [https://github.com/microsoft/python-package-template/]. The repository is organized as follows (standard template files in parent folder are omitted).
```markdown
├── docs
|   ├── development
├── src
├── tests
├── authentication-data.env
└── handler_data.env
```
Here, `src` is the main "hub" for creating and deploying Cognite Functions, `tests` provides scripts for unit tests and UaT tests, and in `docs/development` you find detailed documentation of project development and deployment. `authentication-data.env` contains IDs for your tenant and the client to authenticate with, and `handler-data.env` holds variables that are agnostic to the Cognite Functions, for instance the dataset id.

## Getting started
1. Clone the repository using git and move to the cloned directory
```
git clone https://github.com/AkerBP-DataOps/deos-cognite-functions-template.git
cd deos-cognite-functions-template
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
- For advanced management of Python virtual environments, `poetry` is recommended for the installation. See [using-cognite-python-sdk](https://github.com/cognitedata/using-cognite-python-sdk) for more details
4. Specify dependencies in `requirements.txt`
- The main entry point for a Cognite Function `myname` is a `handler.py` containing the particular transformations/calculations of your time series. This is located in the subfolder `cf_myname`, and is supported by a `requirements.txt` file located in the same folder
- *If your virtual environment includes other packages not used by `handler.py`, we recommend using `pipreqs` to ensure consistency with the `requirements.txt` file*
```
pip install pipreqs
pipreqs src/cf_myname
```
- ***NB**: `pipreqs` will specify wrong dependency to Cognite Python SDK package. **Replace the line `cognite==X.X.X` with `cognite-sdk` in `requirements.txt`**. If you have installed other packages, it is a good idea to double-check their specification in `requirements.txt`*

## Authentication with Python SDK.
To read/write data from/to CDF, we need to connect with the Cognite application. This section describes the process of authenticating with a Cognite client using app registration and the OIDC protocol. The complete code for authenticating is found in `src/cognite_authentication.py`
- Create a user (or sign into your existing) account at [Cognite Hub](https://hub.cognite.com/). This will connect you to an Azure Active Directory tenant that is used to authenticate with CDF, which gives you read access to the time series dataset used in this project. All Aker BP accounts and guest accounts have by default access to the development environment of CDF (Cognite Fusion Dev).
- To authenticate with the Cognite API we generate a client credential, more specifically a `OAuthClientCredentials`. Authentication is done in `src/initialize.py`. Five parameters must be specified:
  1. `TENANT_ID`: ID of the Azure AD tenant where the user is signed in (here: `3b7e4170-8348-4aa4-bfae-06a3e1867469`)
  2. `CLIENT_ID`: ID of the application in Azure AD. This will be unique value available to users that have been granted write access to the dataset. It is found in your "Key vaults > *key_vault_name* > Secrets" service at [Microsoft Azure](https://portal.azure.com/#home) (reach out to the CDF team if you don't know the exact *key_vault_name*), where the relevant key has suffix ending "-ID"
  3. `CDF_CLUSTER`: Cluster where your CDF project is installed (here: `api`)
  4. `COGNITE_PROJECT`: Name of CDF project (here: `akerbp`)
  5. `CLIENT SECRET`: A secret token required for deployement. This is found in your "Key vaults > Secrets" service at [Microsoft Azure](https://portal.azure.com/#home) where the relevant key has suffix ending "-SECRET"
- With these, we can authenticate by fetching our credentials
```
creds = OAuthClientCredentials(
          token_url=AUTHORITY_URI + "/oauth2/v2.0/token",
          client_id=CLIENT_ID,
          scopes=SCOPES,
          client_secret=CLIENT_SECRET,
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

## Deployment of Cognite Function and scheduling
This section outlines the procedure for creating a Cognite function for CDF, deployment and scheduling using Cognite's Python SDK. The jupyter file `src/run_functions.ipynb` is devoted for this pupose, and contains the code snippets listed in this section. Run the code cells consequtively to authenticate with CDF, and deploy and schedule Cognite Functions for given input data.
The `src` folder is organized as follows.
```markdown
├── src
|   ├── README.md
|   ├── __init__.py
│   ├── cf_avg_drainage_rate
│   │   ├── zip_handle.zip
│   │   ├── requirements.txt
│   │   ├── handler.py
│   │   ├── transformation.py
│   ├── cf_A
│   │   ├── zip_handle.zip
│   │   ├── requirements.txt
│   │   ├── handler.py
│   │   ├── transformation.py
│   ├── cf_B
│   ├── handler_utils.py
│   ├── cognite_authentication.py
│   ├── initialize.py
│   └── run_functions.ipynb
```
Here we find authentication scripts `cognite_authentication.py` and `initialize.py`, a deployment script `run_functions.ipynb`, and a utility script `handler_utils.py` with functionality common for all Cognite Functions run on a time series. 

The subfolder `cf_*myname*` contains all files specific for your Cognite Function labeled `myname` (where convention is that different words in `myname` are separated by dashes (-). 
1. **`handler.py`**: main entry point containing a `handle(client, data)` function that runs a Cognite Function using a Cognite `client` and relevant input data provided in the dictionary `data`
2. **`transformation.py`**: script containing transformations/calculations for the particular Cognite Function, encapsulated in a `run_transformation` function with the following template
   ```
   def run_transformation(data):
    ts_data = get_input_ts(data)
    ts_data = align_time_series(ts_data, data)
    
    ts_output = calculation(data, *ts_data)
   
    out_list = store_output_ts(ts_output, data)
    return out_list
   ```
   where the only modification required is a programmatic setup of your calculation in the `calculation` function, taking as input a data dictionary `data` containing all parameters for your Cognite Function and a list `ts_data` of time series inputs. ***NB:*** Make sure that the time series in `ts_data` are listed in correct order according to the calculations performed in `calculation`.
4. **`requirements.txt`**: file containing Python package requirements to run the Cognite Function
5. **`zip_handle.zip`**: a Cognite File scoped to the dataset that our function is associated with

The desired Cognite Function `myname` is run by supplying `myname` as value to the `function_name` key in the `data` argument of `handle`, i.e., `data['function_name'] = *myname*`. The input `data` can be modified in the `data_dict` dictionary in `run_functions.ipynb`

*A client secret is required to deploy the function to CDF. This means that we need to authenticate with a Cognite client using app registration (see section Authentication with Python SDK), **not** through interactive login. This requirement is not yet specified in the documentation from Cognite. The request of improving the documentation of Cognite Functions has been sent to the CDF team to hopefully resolve any confusions regarding deployment.*

### 1. Create file
First, we create a Cognite File to link with our Cognite Function. The file must point to a zip file `zip_handle.zip` in the subfolder `cf_*myname*` designated for the Cognite Function with name `myname`. The zip file contains the main entry `handler.py` with a function named `handle` inside it, and other necessary files to run `handler.py` (here: `requirements.txt`, `handler_utils.py` and `transformation.py`)
```
folder = os.getcwd().replace("\\", "/")
folder_cf = folder + "/" + data_dict["function_name"]
zip_name = "zip_handle.zip"

uploaded = client.files.upload(path=f"{folder_cf}/{zip_name}", name=zip_name, data_set_id=dataset_id)
```
The Cognite File is associated with a dataset with id `dataset_id` and uploaded to CDF.
### 2. Deployment
The next step is to create an instance of the `handle` function (located in the subfolder `cf_*myname*`) to be deployed to CDF. 
```
func_myname = client.functions.create(
    name=f"{data_dict['function_name']}",
    external_id=f"{data_dict['function_name']}",
    file_id=uploaded.id,
)
```
The `file_id` is assigned the id of the newly created zip file.
### 3. Set up schedule
Finally, we set up a schedule for our function. Here, we want the function to run every 15 minutes. This is specified using the cron expression `*/15 * * * *`. The function receives necessary input data `data_dict` through the `data` argument. The schedule is instantiated by
```
func_myname_schedule = client.functions.schedules.create(
    name=f"{data_dict['function_name']}",
    cron_expression="*/15 * * * *", # every 15 min
    function_id=func_myname.id, # id of function instance
    description="Calculation scheduled every hour",
    data=data_dict
)
```

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

## Improvements for access request system
Completing all steps in this demonstration, from retrieving the original time series to writing the new time series back to CDF Prod, unfortunately takes an undeseriably long time and is subject to efficiency improvements. The main bottleneck is the process of granting necessary read and write accesses for CDF. 
- The form for requesting access is more comprehensive than necessary. It is not trivial what to fill out in some sections. Thus, we believe too much time is wasted mailing the CDF Operations team back and fourth for particular guidance. This process has potential for streamlining by transitioning from restriction-based to constraint-based, facilitating a more rapid onboarding process for new developers
- If you submit a form with the same title as another submitted form, it is considered as a duplicate and will be deleted. Hence, if you fill out something wrong and have to resubmit the form, it is crucial to rename the title. We think this issue should be communicated better by the CDF Ops team to avoid users waiting forever for their submission to be processed
- The CDF Operations team is by the time of writing (September 2023) understaffed, where a response to your request form is expected to take multiple days, or up to a week. This is not sustainable for a company like Aker BP with lots of employees developing their work scope

## *OPTIONAL READ:* Calculation of drainage rate
This section gives the mathematical and technical details how to calculate drainage rate from a time series of volume percentage - the particular case considered in this project. If you are only interested in deployment of Cognite Functions in general, we recommend jumping to section Update time series at prescribed schedules.

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
