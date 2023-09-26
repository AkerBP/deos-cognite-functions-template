import atexit
import os
from typing import Any

from cognite.client._cognite_client import CogniteClient
from cognite.client.config import ClientConfig
from cognite.client.credentials import Token

from msal import PublicClientApplication, SerializableTokenCache


def set_cdf_client_connection(
    client_name: str, project: str, client_id: str, tenant_id: str
) -> CogniteClient:
    TOKEN_FILENAME: str = "auth_cache.bin"

    CDF_CLUSTER: str = "api"
    AUTHORITY_HOST_URI: str = "https://login.microsoftonline.com"
    PORT: int = 53000

    AUTHORITY_URI: str = f"{AUTHORITY_HOST_URI}/{tenant_id}"
    SCOPES: list[str] = [f"https://{CDF_CLUSTER}.cognitedata.com/.default"]
    BASE_URL: str = f"https://{CDF_CLUSTER}.cognitedata.com"

    def _create_cache() -> SerializableTokenCache:
        cache = SerializableTokenCache()
        if os.path.exists(TOKEN_FILENAME):
            cache.deserialize(open(TOKEN_FILENAME, "r").read())
        atexit.register(
            lambda: open(TOKEN_FILENAME, "w").write(cache.serialize())
            if cache.has_state_changed
            else None
        )
        return cache

    def _authenticate_azure(app) -> dict:
        # Firstly, check the cache to see if this end user has signed in before
        accounts = app.get_accounts()
        if accounts:
            creds = app.acquire_token_silent(SCOPES, account=accounts[0])
        else:
            # interactive login - make sure you have http://localhost:port in Redirect URI
            # App Registration as type "Mobile and desktop applications"
            creds = app.acquire_token_interactive(
                scopes=SCOPES,
                port=PORT,
            )

        return creds

    def _get_token() -> Any:
        cred = _authenticate_azure(app)
        if cred is not None:
            # Setting environment variable "CDF_ACCESS_TOKEN" to access ToKen directly
            os.environ["CDF_ACCESS_TOKEN"] = cred["access_token"]
            return cred["access_token"]
        else:
            return None

    app = PublicClientApplication(
        client_id=client_id, authority=AUTHORITY_URI, token_cache=_create_cache()
    )

    config = ClientConfig(
        client_name=client_name,
        project=project,
        credentials=Token(_get_token),
        base_url=BASE_URL,
        timeout=300,
    )
    client = CogniteClient(config)

    return client
