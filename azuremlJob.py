from azureml.core import Experiment, Workspace, ScriptRunConfig, Environment

ws = Workspace.get(name="Deduplication", subscription_id="bcb09b70-6f46-46b4-8090-0f3944e906f0", resource_group="ELCA")

experiment = Experiment(workspace=ws, name="testExperiment")


compute_target = "LowPriorityCompute"

myenv = Environment.get(workspace=ws, name="torchEnv", version=3)


src = ScriptRunConfig(
    source_directory=".", script="IMDBSentimentClassification.py", compute_target=compute_target, environment=myenv
)


run = experiment.submit(config=src)
run.wait_for_completion(show_output=True)