# AWS — Instructions

## Credentials & secrets

- Never hardcode AWS access keys, secret keys, or tokens in code. Assume
  credentials come from the environment (`AWS_PROFILE`, environment
  variables, or an instance/role) via boto3's default credential chain —
  don't ask the user to paste keys into code.
  a `.env` file may hold non-AWS keys (see this repo's existing
  `load_dotenv()` pattern) but AWS credentials specifically should still go
  through boto3's standard chain, not be read manually and passed around.
- If a task needs a new IAM identity, prefer creating a role (for
  services/compute) over a long-lived IAM user access key.

## IAM

- Always propose least-privilege policies: scope `Resource` to the specific
  ARN(s) involved, not `"*"`, unless the action genuinely has no
  resource-level permission support (check the service's IAM reference) or
  the user explicitly wants a broad policy for a throwaway demo.
- Explain the risk in one sentence when reviewing a policy that uses `"*"`
  for `Resource` or `Action`, rather than silently leaving it or silently
  "fixing" it without flagging the change.

## Service choice for demo/prototype work

Given this repo is a set of LangChain/deepagents demos, default to the
lowest-ceremony option unless the user asks for production-grade
infrastructure:

- Expose an agent as an API → **Lambda + Function URL** (or API Gateway if
  they need auth/throttling/multiple routes) rather than standing up
  ECS/EKS for a demo.
- Need object storage for files the deep agent's `FilesystemBackend`
  produces → **S3**, not EFS/EBS, unless low-latency POSIX access is
  required.
- Need a durable key-value store for a `StoreBackend`-style persistence
  layer → **DynamoDB**, simplest managed option matching the access
  pattern (fetch by namespace/key).
- Using AWS's own model hosting instead of/alongside Groq/OpenAI →
  **Bedrock**, via `langchain_aws.ChatBedrock` (or `init_chat_model` with a
  `bedrock:` prefix in modern LangChain) — mention this as an alternative
  when the user asks about swapping model providers, not as an unprompted
  suggestion.

## boto3 conventions

- Use the `boto3.client("<service>")` or `boto3.resource("<service>")`
  pattern; don't manually build HTTP requests to AWS APIs.
- Always handle `botocore.exceptions.ClientError` around AWS calls that can
  legitimately fail for reasons outside the code's control (missing
  resource, throttling) — not as blanket defensive wrapping around every
  call.
- Set an explicit `region_name` (or document that it's expected to come
  from `AWS_DEFAULT_REGION`) rather than silently relying on an
  unstated default.

## Cost awareness

- Flag when a suggested design has an ongoing cost implication a demo
  probably doesn't want (e.g. an always-on NAT Gateway, a provisioned
  (non-serverless) DynamoDB table, an always-on EC2 instance) and suggest
  the serverless/on-demand equivalent.
