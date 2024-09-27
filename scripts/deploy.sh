#!/usr/bin/env bash
# Build, push to ECR, and force ECS to pull the new image.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
PROJECT="${PROJECT:-mnmt}"
TAG="${TAG:-$(git rev-parse --short HEAD)}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${PROJECT}-api"

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REPO"

docker build -t "$PROJECT-api:$TAG" .
docker tag "$PROJECT-api:$TAG" "$REPO:$TAG"
docker push "$REPO:$TAG"

aws ecs update-service \
  --cluster "${PROJECT}-cluster" \
  --service "${PROJECT}-api" \
  --force-new-deployment \
  --region "$REGION"
