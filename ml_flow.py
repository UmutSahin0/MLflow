import mlflow


def create_experiment(name, artifact_location, tags):
    mlflow.create_experiment(
        name=name,
        artifact_location=artifact_location,
        tags=tags
    )


if __name__ == '__main__':
    #create_experiment('testing_mlflow', 'testing_mlflow_artifacts', {'env': 'dev', 'version': '1.0.0'})

    mlflow.set_experiment(experiment_name='testing_mlflow')

    with mlflow.start_run(run_name='testing') as run:

        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("status: {}".format(run.info.status))
        print("start_time: {}".format(run.info.start_time))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))
