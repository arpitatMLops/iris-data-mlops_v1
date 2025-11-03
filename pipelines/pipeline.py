import os
import boto3
from sagemaker.session import Session
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker import get_execution_role
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

REGION = "eu-north-1"
boto_sess = boto3.Session(region_name=REGION)
sm_sess = Session(boto_sess)
pipe_sess = PipelineSession(boto_sess)
role = get_execution_role()
bucket = "sagemaker-pipelines-iris"

output_prefix = ParameterString(
    name="OutputPrefix",
    default_value=f"s3://{bucket}/pipelines/artifacts/preprocess"
)

preproc = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    sagemaker_session=pipe_sess,
)

step_preprocess = ProcessingStep(
    name="PreprocessIris",
    processor=preproc,
    code=os.path.join(os.path.dirname(__file__), "../src/preprocessing/preprocessing.py"),
    job_arguments=["--output-dir", "/opt/ml/processing/output"],
    outputs=[
        ProcessingOutput(
            output_name="processed",
            source="/opt/ml/processing/output",
            destination=output_prefix
        )
    ],
)

pipeline = Pipeline(
    name="IrisPreprocessOnly",
    parameters=[output_prefix],
    steps=[step_preprocess],
    sagemaker_session=pipe_sess,
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    params = {}
    if args.output_prefix:
        params["OutputPrefix"] = args.output_prefix

    pipeline.upsert(role_arn=role)
    execution = pipeline.start(parameters=params)
    print("Pipeline started. Execution ARN:", execution.arn)
