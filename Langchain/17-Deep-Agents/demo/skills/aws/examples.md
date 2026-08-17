# AWS — Examples

## Example 1: Upload a deep agent's output file to S3

```python
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3")  # region/credentials from default chain

def upload_report(local_path: str, bucket: str, key: str) -> None:
    try:
        s3.upload_file(local_path, bucket, key)
    except ClientError as e:
        raise RuntimeError(f"Failed to upload {local_path} to s3://{bucket}/{key}") from e
```

## Example 2: Least-privilege IAM policy for a Lambda reading one S3 prefix

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::my-reports-bucket/reports/*"
    }
  ]
}
```
Scoped to `GetObject` on one prefix — not `s3:*` on `Resource: "*"`.

## Example 3: Deploying an agent behind a Lambda Function URL (minimal shape)

```python
# lambda_handler.py
from deepagents import create_deep_agent

agent = create_deep_agent(model="groq:openai/gpt-oss-20b", tools=[...])

def handler(event, context):
    body = event["body"]  # parse JSON as needed
    result = agent.invoke({"messages": [{"role": "user", "content": body["question"]}]})
    return {"statusCode": 200, "body": result["messages"][-1]["content"]}
```
Good default for exposing a demo agent as an HTTP endpoint without standing
up a container platform. If the user needs auth, rate limiting, or multiple
routes, suggest fronting it with API Gateway instead of hand-rolling that in
the Lambda.

## Example 4: Flagging a costly default

User: "spin up a NAT Gateway so my Lambda in a private subnet can call the
Bedrock API."

Good response: note that a NAT Gateway has an hourly + per-GB cost that
runs continuously, and for a Lambda-only workload a **VPC Gateway/Interface
Endpoint** for the specific AWS service (or simply not placing the Lambda
in a VPC, if no private resource access is needed) avoids that cost —
recommend it as the default, with the NAT Gateway as the fallback if the
user has a concrete reason to need general outbound internet access.
