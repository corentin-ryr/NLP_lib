from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(), "bcb09b70-6f46-46b4-8090-0f3944e906f0", "ELCA", "Deduplication"
)

# Supported paths include:
# local: './<path>'
# blob:  'https://<account_name>.blob.core.windows.net/<container_name>/<path>'
# ADLS gen2: 'abfss://<file_system>@<account_name>.dfs.core.windows.net/<path>/'
# Datastore: 'azureml://datastores/<data_store_name>/paths/<path>'

my_path = "data/imdbCleaned"

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FOLDER,
    description="imdb large movie review dataset v1",
    name="IMDBSentiment",
    version='1.0'
)

ml_client.data.create_or_update(my_data)