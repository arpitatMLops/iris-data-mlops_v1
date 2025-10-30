import boto3
import json
import time
from botocore.exceptions import ClientError

CF = boto3.client("cloudformation")
SF = boto3.client("stepfunctions")
S3 = boto3.client("s3")

def s3_url_to_https(s3_url):
    """Convert s3://bucket/key to https://bucket.s3.amazonaws.com/key"""
    if not s3_url.startswith("s3://"):
        return s3_url
    parts = s3_url[len("s3://"):].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return f"https://{bucket}.s3.amazonaws.com/{key}"

def deploy_stack(stack_name, template_s3, parameters, capabilities):
    """Create or update a stack from S3 template (non-blocking)."""
    template_url = s3_url_to_https(template_s3)
    cfn_params = [{"ParameterKey": k, "ParameterValue": str(v)} for k, v in (parameters or {}).items()]

    try:
        CF.describe_stacks(StackName=stack_name)
        print(f"[deploy_stack] Updating stack {stack_name} ... TemplateURL={template_url}")
        resp = CF.update_stack(
            StackName=stack_name,
            TemplateURL=template_url,
            Parameters=cfn_params,
            Capabilities=capabilities or [],
        )
        status = "update_started"
    except ClientError as e:
        msg = str(e)
        if "does not exist" in msg or "Stack with id" in msg and "does not exist" in msg:
            print(f"[deploy_stack] Creating stack {stack_name} ... TemplateURL={template_url}")
            resp = CF.create_stack(
                StackName=stack_name,
                TemplateURL=template_url,
                Parameters=cfn_params,
                Capabilities=capabilities or [],
            )
            status = "create_started"
        elif "No updates are to be performed" in msg:
            print(f"[deploy_stack] No updates required for {stack_name}.")
            return {"status": "no_change", "StackName": stack_name}
        else:
            print(f"[deploy_stack] Unexpected error for {stack_name}: {msg}")
            raise

    stack_id = resp.get("StackId")
    print(f"[deploy_stack] {status}: {stack_id}")
    return {"status": status, "StackId": stack_id}


def wait_for_stack(stack_name, wait_seconds=15, timeout_minutes=30):
    """Poll CloudFormation until the stack reaches CREATE_COMPLETE/UPDATE_COMPLETE or fails."""
    deadline = time.time() + timeout_minutes * 60
    while True:
        try:
            resp = CF.describe_stacks(StackName=stack_name)
            stack = resp["Stacks"][0]
            status = stack["StackStatus"]
            print(f"[wait_for_stack] {stack_name} status={status}")
            if status in ("CREATE_COMPLETE", "UPDATE_COMPLETE"):
                return stack
            # failure terminal states
            if status.endswith("_FAILED") or status.endswith("_ROLLBACK_COMPLETE") or status in ("DELETE_COMPLETE",):
                raise RuntimeError(f"Stack {stack_name} entered failure state: {status}")
        except ClientError as e:
            msg = str(e)
            if "does not exist" in msg:
                print(f"[wait_for_stack] {stack_name} not found yet.")
            else:
                print(f"[wait_for_stack] describe_stacks error: {msg}")
                raise
        if time.time() > deadline:
            raise TimeoutError(f"Timeout waiting for stack {stack_name} after {timeout_minutes} minutes")
        time.sleep(wait_seconds)


def get_stack_outputs(stack_name):
    desc = CF.describe_stacks(StackName=stack_name)["Stacks"][0]
    outputs = {o["OutputKey"]: o["OutputValue"] for o in desc.get("Outputs", [])}
    return outputs


def start_state_machine(state_machine_arn, input_payload=None):
    """Start Step Functions state machine and return executionArn."""
    if not state_machine_arn:
        raise ValueError("state_machine_arn is required to start execution")
    payload = input_payload or {"trigger": "ci-deploy", "timestamp": int(time.time())}
    resp = SF.start_execution(
        stateMachineArn=state_machine_arn,
        name=f"ci-deploy-{int(time.time())}",
        input=json.dumps(payload)
    )
    return resp.get("executionArn")


def lambda_handler(event, _context):
    """
    Deploy infra stack, wait for completion, deploy pipeline stack, wait, then start the StepFunctions state machine.
    Returns: JSON with stack statuses and (if started) executionArn.
    """
    results = []
    try:
        # Validate inputs
        infra_template = event.get("InfraTemplateS3")
        pipeline_template = event.get("PipelineTemplateS3")
        if not infra_template or not pipeline_template:
            return {"status": "error", "message": "InfraTemplateS3 and PipelineTemplateS3 are required in event"}

        infra_params = event.get("InfraParameters", {}) or {}
        infra_stack = event.get("InfraStackName", "iris-mlops-infra")
        capabilities = event.get("Capabilities", [])

        # Start infra stack
        infra_res = deploy_stack(infra_stack, infra_template, infra_params, capabilities)
        results.append(infra_res)

        # Wait for infra to finish (if it started)
        if infra_res.get("status") in ("create_started", "update_started"):
            infra_desc = wait_for_stack(infra_stack, wait_seconds=15, timeout_minutes=30)
        else:
            infra_desc = CF.describe_stacks(StackName=infra_stack)["Stacks"][0]

        infra_outputs = {o["OutputKey"]: o["OutputValue"] for o in infra_desc.get("Outputs", [])}
        print("[lambda_handler] Infra outputs:", infra_outputs)

        # Validate required infra outputs exist
        required_infra_outputs = ["StepFnLogGroupArn", "RoleStepFunctionsArn"]
        missing = [k for k in required_infra_outputs if k not in infra_outputs]
        if missing:
            return {"status": "error", "message": f"Missing infra outputs: {missing}", "outputs": infra_outputs}

        # Prepare pipeline parameters (merge provided ones and infra outputs)
        pipeline_params = (event.get("PipelineParameters", {}) or {}).copy()
        pipeline_params["StepFnLogGroupArn"] = infra_outputs["StepFnLogGroupArn"]
        pipeline_params["RoleStepFunctionsArn"] = infra_outputs["RoleStepFunctionsArn"]

        pipeline_stack_name = event.get("PipelineStackName", "iris-mlops-pipeline")
        pipeline_res = deploy_stack(pipeline_stack_name, pipeline_template, pipeline_params, capabilities)
        results.append(pipeline_res)

        # Wait for pipeline stack
        if pipeline_res.get("status") in ("create_started", "update_started"):
            pipeline_desc = wait_for_stack(pipeline_stack_name, wait_seconds=15, timeout_minutes=30)
        else:
            pipeline_desc = CF.describe_stacks(StackName=pipeline_stack_name)["Stacks"][0]

        pipeline_outputs = {o["OutputKey"]: o["OutputValue"] for o in pipeline_desc.get("Outputs", [])}
        print("[lambda_handler] Pipeline outputs:", pipeline_outputs)

        # Try to find the State Machine ARN in pipeline outputs.
        # Your pipeline template defines 'StateMachineArn' in Outputs — use that key.
        state_machine_arn = pipeline_outputs.get("StateMachineArn") or pipeline_outputs.get("StateMachine") or pipeline_outputs.get("StateMachineARN")

        execution_arn = None
        if state_machine_arn:
            # Start Step Functions execution
            try:
                execution_arn = start_state_machine(state_machine_arn, input_payload=event.get("ExecutionInput"))
                print("[lambda_handler] Started StepFunctions execution:", execution_arn)
            except ClientError as e:
                print("[lambda_handler] Failed to start state machine:", str(e))
                return {"status": "error", "message": f"Failed to start state machine: {str(e)}", "stacks": results, "pipeline_outputs": pipeline_outputs}

        final = {
            "status": "ok",
            "stacks": results,
            "infra_outputs": infra_outputs,
            "pipeline_outputs": pipeline_outputs,
            "executionArn": execution_arn
        }
        return final

    except Exception as e:
        print("[lambda_handler] Exception:", repr(e))
        return {"status": "error", "message": str(e), "stacks": results}
        

# Local CLI testing (keeps your original harness)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infra-template", required=True)
    parser.add_argument("--pipeline-template", required=True)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()

    event = {
        "InfraStackName": "iris-mlops-infra",
        "PipelineStackName": "iris-mlops-pipeline",
        "InfraTemplateS3": args.infra_template,
        "PipelineTemplateS3": args.pipeline_template,
        "InfraParameters": {
            "ProjectName": "iris-mlops",
            "SageMakerRoleArn": "arn:aws:iam::182406535835:role/service-role/AmazonSageMaker-ExecutionRole-20251013T175169",
        },
        "PipelineParameters": {
            "ProjectName": "iris-mlops",
            "ECRImageURI": "182406535835.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-studio-d-x5rfkjrvcd2t:default-20251013T175168",
            "S3BucketName": "iris-mlops-bucket-182406535835",
            "SageMakerRoleArn": "arn:aws:iam::182406535835:role/service-role/AmazonSageMaker-ExecutionRole-20251013T175169",
        },
        "Capabilities": ["CAPABILITY_NAMED_IAM"]
    }

    print(json.dumps(lambda_handler(event, None), indent=2))
