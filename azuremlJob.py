from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command, Input




ml_client = MLClient(
    DefaultAzureCredential(exclude_interactive_browser_credential=False), "bcb09b70-6f46-46b4-8090-0f3944e906f0", "ELCA", "Deduplication"
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
            path="azureml:IMDBCleaned:1",
        )
    },
    compute=compute_target,
    experiment_name="testExperiment"
)

# submit the command
returned_job = ml_client.jobs.create_or_update(command_job)
# get a URL for the status of the job
returned_job.services["Studio"].endpoint