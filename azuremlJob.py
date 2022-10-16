from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command, Input

ml_client = MLClient(
    DefaultAzureCredential(), "bcb09b70-6f46-46b4-8090-0f3944e906f0", "ELCA", "Deduplication"
)

# specify aml compute name.
compute_target = "LowPriorityCompute"

try:
    ml_client.compute.get(compute_target)
except Exception:
    raise ValueError("Impossible to get compute target.")
    

# define the command
command_job = command(
    code=".",
    command="python IMDBSentimentClassification.py --imdb-path ${{inputs.iris_csv}}",
    environment="torchEnv@latest",
    inputs={
        "imdb_path": Input(
            type="uri_folder",
            path="https://portal.azure.com/#blade/Microsoft_Azure_FileStorage/FileShareMenuBlade/overview/storageAccountId/%2Fsubscriptions%2Fbcb09b70-6f46-46b4-8090-0f3944e906f0%2FresourceGroups%2FELCA%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fdeduplication8129376456/path/code-391ff5ac-6576-460f-ba4d-7e03433c68b6%2FaclImdb%2F",
        )
    },
    compute=compute_target,
    experiment_name="testExperiment"
)

# submit the command
returned_job = ml_client.jobs.create_or_update(command_job)
# get a URL for the status of the job
returned_job.services["Studio"].endpoint