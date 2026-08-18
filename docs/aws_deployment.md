# AWS ECS/Fargate deployment runbook

AWS deployment is prepared but has not been executed. The container embeds the trained
`artifacts/model.joblib`; run training before building the production image.

## Prerequisites

- Docker, AWS CLI v2, and an AWS account with permission to use ECR, ECS, IAM,
  CloudWatch Logs, EC2 networking, and (optionally) an Application Load Balancer.
- AWS CLI authentication configured outside this repository.
- A trained `artifacts/model.joblib` produced by `python -m scripts.run_train`.

Set shell variables for your region, account, and repository; never commit their real values.

```bash
AWS_REGION=<region>
AWS_ACCOUNT_ID=<12-digit-account-id>
ECR_REPOSITORY=house-prices-api
IMAGE_TAG=latest
```

## Build and push the image

```bash
docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .
aws ecr create-repository --repository-name ${ECR_REPOSITORY} --region ${AWS_REGION}
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}
```

If the ECR repository already exists, skip its creation.

## Create ECS resources

1. Create an ECS cluster using the Fargate capacity provider.
2. Create the `ecsTaskExecutionRole` with the standard ECS task execution policy.
3. Create a CloudWatch Logs group such as `/ecs/house-prices-api` with a short retention period.
4. Register a Fargate task definition using the pushed image, `awsvpc` networking,
   a small CPU/memory size (for example 1 vCPU/2 GB), container port `8000`, and the
   `awslogs` log driver. Set `PORT=8000` only if overriding the image default.
5. Create an ECS service with one desired task in public subnets. Assign a public IP
   for a minimal demo, or attach an ALB for a stable public endpoint.
6. Allow inbound TCP `8000` only from your own IP for a direct demo. With an ALB,
   allow the ALB security group to reach the task on `8000`, and expose only the ALB.

The API is stateless and needs no database or S3 bucket; the small model artifact is
embedded in the image. Configure the health check path as `/health`.

## Verify

```bash
curl http://<public-endpoint>:8000/health
curl -X POST http://<public-endpoint>:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"OverallQual":7,"GrLivArea":1500,"YearBuilt":2005,"YrSold":2010}'
```

Confirm the health response reports `"status":"ok"`, the prediction is numeric,
and the task logs contain no startup error.

## Cost control and teardown

Scale the ECS service desired count to zero when the demo is idle. To remove the
deployment, delete the ECS service, task definitions no longer needed, cluster, ALB
and target group if created, CloudWatch log group, and ECR images/repository. Check
for public IPv4 and load-balancer charges. Keep IAM roles only if they are reused.
