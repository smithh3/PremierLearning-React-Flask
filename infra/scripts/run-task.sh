#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK="pipeline"
FOLLOW=0
SUGGEST=0
for arg in "$@"; do
  case "$arg" in
    --follow) FOLLOW=1 ;;
    --suggest) SUGGEST=1 ;;
    pipeline|learn|apply) TASK="$arg" ;;
    -h|--help)
      echo "Usage: $0 [pipeline|learn|apply] [--follow] [--suggest]"
      echo "  --suggest  Override this run: train, then print/email transfers without submitting."
      exit 0
      ;;
    *)
      echo "Usage: $0 [pipeline|learn|apply] [--follow] [--suggest]"
      exit 1
      ;;
  esac
done

cd "$ROOT/infra/terraform"
CLUSTER="$(terraform output -raw ecs_cluster_name)"
SG="$(terraform output -raw ecs_security_group_id)"
SUBNETS="$(terraform output -json subnet_ids | python -c "import json,sys; print(','.join(json.load(sys.stdin)))")"
ASSIGN_IP="$(terraform output -raw assign_public_ip)"
REGION="$(terraform output -raw region)"
LOG_GROUP="$(terraform output -raw log_group)"

case "$TASK" in
  pipeline) DEF="$(terraform output -raw pipeline_task_definition)" ;;
  learn) DEF="$(terraform output -raw learn_task_definition)" ;;
  apply) DEF="$(terraform output -raw apply_task_definition)" ;;
esac

if [ "$DEF" = "null" ] || [ -z "$DEF" ]; then
  echo "No task definition for $TASK. Did you set enable_split=true?"
  exit 1
fi

export AWS_PAGER=""
export MSYS_NO_PATHCONV=1

CONTAINER="$TASK"
RUN_ARGS=(
  --cluster "$CLUSTER"
  --task-definition "$DEF"
  --launch-type FARGATE
  --region "$REGION"
  --network-configuration "awsvpcConfiguration={subnets=[${SUBNETS}],securityGroups=[${SG}],assignPublicIp=${ASSIGN_IP}}"
  --query "tasks[0].taskArn"
  --output text
)
if [ "$SUGGEST" -eq 1 ]; then
  echo "Suggest-only override: APPLY_TRANSFERS=false SUGGEST_TRANSFERS=true RUN_LEARNING=1"
  RUN_ARGS+=(--overrides "{\"containerOverrides\":[{\"name\":\"${CONTAINER}\",\"environment\":[{\"name\":\"APPLY_TRANSFERS\",\"value\":\"false\"},{\"name\":\"SUGGEST_TRANSFERS\",\"value\":\"true\"},{\"name\":\"RUN_LEARNING\",\"value\":\"1\"}]}]}")
fi

TASK_ARN="$(aws ecs run-task "${RUN_ARGS[@]}")"

echo "Started $TASK_ARN"

if [ "$FOLLOW" -eq 0 ]; then
  exit 0
fi

echo "Waiting for CloudWatch logs from $LOG_GROUP"
echo "Ctrl+C stops the tail only; the Fargate task keeps running."
aws logs tail "$LOG_GROUP" --follow --since 2m --format short --region "$REGION"
