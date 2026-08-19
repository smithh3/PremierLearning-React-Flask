#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REGION="${AWS_REGION:-eu-west-2}"
TAG="${IMAGE_TAG:-latest}"

cd "$ROOT/infra/terraform"
REPO_URL="$(terraform output -raw ecr_repository_url)"
cd "$ROOT"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${REPO_URL%%/*}"

docker build --platform linux/amd64 -t "${REPO_URL}:${TAG}" .
docker push "${REPO_URL}:${TAG}"
echo "Pushed ${REPO_URL}:${TAG}"
