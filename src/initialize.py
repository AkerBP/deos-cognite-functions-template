from cognite.client import CogniteClient
from cognite.client.config import ClientConfig
from cognite.client.credentials import OAuthInteractive, OAuthClientCredentials

def initialize_client():
    TENANT_ID = "3b7e4170-8348-4aa4-bfae-06a3e1867469"
    CDF_CLUSTER = "api"
    CLIENT_NAME = "akerbp"
    CLIENT_ID = "779f2b3b-b599-401a-96aa-48bd29132a27"  #Cognite API User access- app registration
    COGNITE_PROJECT = "akerbp"
    SCOPES = [f"https://{CDF_CLUSTER}.cognitedata.com/.default"]

    AUTHORITY_HOST_URI = "https://login.microsoftonline.com"
    AUTHORITY_URI = AUTHORITY_HOST_URI + "/" + TENANT_ID
    PORT = 53000

    creds = OAuthInteractive(client_id=CLIENT_ID, authority_url=AUTHORITY_URI, scopes=SCOPES)

    client_cnf = ClientConfig(client_name=CLIENT_NAME,
                    base_url=f"https://{CDF_CLUSTER}.cognitedata.com",
                    project=COGNITE_PROJECT, credentials=creds)
    client = CogniteClient(client_cnf)

    status = client.iam.token.inspect() #verify your client token and status
    #print(status)
    if "projects" not in vars(status):
        raise Exception("Token Error!")

    return client, status