---
name: aws
description: Use this skill whenever the user asks about AWS services, architecture, IAM permissions, deployment (e.g. Lambda, ECS, EC2, S3, Bedrock), boto3 code, CloudFormation/CDK/Terraform for AWS resources, or cost/security considerations for AWS infrastructure.
---

# AWS Skill

Gives the deep agent a consistent approach to AWS questions and code: secure
defaults (least-privilege IAM, no hardcoded credentials), correct boto3
usage, and a bias toward serverless/managed services for demo-scale
workloads unless the user indicates otherwise.

## When to use this skill

- Writing or reviewing `boto3` code (S3, Lambda, DynamoDB, Bedrock, etc.).
- Designing AWS architecture for a feature (e.g. "how should I deploy this
  agent so it's callable via an API").
- Writing IAM policies or reviewing them for over-permissive access.
- Infrastructure-as-code for AWS (CDK, CloudFormation, Terraform `aws`
  provider).
- Questions about AWS Bedrock specifically when the user is comparing it to
  or integrating it alongside LangChain/deepagents model providers.

## How this skill is organized

- `instructions.md` — conventions for credentials, IAM, service choice, and
  IaC.
- `examples.md` — worked examples (boto3 snippets, an IAM policy, a minimal
  deployment shape for an agent backend).

Read `instructions.md` before writing any AWS code or proposing an
architecture, and never invent service names/APIs — verify against
`examples.md` or state that you're unsure of the exact API rather than
guessing.
