#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK="pipeline"
FOLLOW=0
SUGGEST=0
APPLY=0
NO_BENCH_FILL=0
POISSON=0
TEAM_ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --follow) FOLLOW=1 ;;
    --suggest) SUGGEST=1 ;;
    --apply) APPLY=1 ;;
    --no-bench-fill) NO_BENCH_FILL=1 ;;
    --poisson) POISSON=1 ;;
    --team-id)
      TEAM_ID="${2:-}"
      if [ -z "$TEAM_ID" ]; then
        echo "--team-id requires a manager id" >&2
        exit 1
      fi
      shift
      ;;
    pipeline|learn|apply) TASK="$1" ;;
    -h|--help)
      echo "Usage: $0 [pipeline|learn|apply] [--follow] [--suggest|--apply] [--poisson] [--team-id ID] [--no-bench-fill]"
      echo "  --suggest       Override this run: train, then print/email transfers without submitting."
      echo "  --apply         Override this run: submit this week's transfers (still emails)."
      echo "  --poisson       Train h1 Poisson + minutes weights, skip the horizon net."
      echo "  --team-id ID    Public squad lookup for this manager id (no FPL login)."
      echo "  --no-bench-fill Skip the second-pass bench refill (rotating XI only)."
      exit 0
      ;;
    *)
      echo "Usage: $0 [pipeline|learn|apply] [--follow] [--suggest|--apply] [--poisson] [--team-id ID] [--no-bench-fill]"
      exit 1
      ;;
  esac
  shift
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
ENV_VARS=()
if [ "$SUGGEST" -eq 1 ]; then
  echo "Suggest-only override: APPLY_TRANSFERS=false SUGGEST_TRANSFERS=true RUN_LEARNING=1"
  ENV_VARS+=(
    '{"name":"APPLY_TRANSFERS","value":"false"}'
    '{"name":"SUGGEST_TRANSFERS","value":"true"}'
    '{"name":"APPLY_ON_DEADLINE_DAY","value":"false"}'
    '{"name":"RUN_LEARNING","value":"1"}'
  )
elif [ "$APPLY" -eq 1 ]; then
  echo "Apply override: APPLY_TRANSFERS=true SUGGEST_TRANSFERS=false RUN_LEARNING=1"
  ENV_VARS+=(
    '{"name":"APPLY_TRANSFERS","value":"true"}'
    '{"name":"SUGGEST_TRANSFERS","value":"false"}'
    '{"name":"APPLY_ON_DEADLINE_DAY","value":"false"}'
    '{"name":"RUN_LEARNING","value":"1"}'
  )
fi
if [ "$POISSON" -eq 1 ]; then
  echo "Poisson override: PREMIERLEARNING_ARGS=--h1-poisson"
  ENV_VARS+=('{"name":"PREMIERLEARNING_ARGS","value":"--h1-poisson"}')
fi
if [ -n "$TEAM_ID" ]; then
  echo "Public squad lookup for manager $TEAM_ID (no FPL login)"
  ENV_VARS+=("{\"name\":\"FPL_MANAGER_ID\",\"value\":\"${TEAM_ID}\"}")
fi
if [ "$NO_BENCH_FILL" -eq 1 ]; then
  echo "Skipping bench fill pass: SKIP_BENCH_FILL=true"
  ENV_VARS+=('{"name":"SKIP_BENCH_FILL","value":"true"}')
fi
RUN_ARGS=(
  --cluster "$CLUSTER"
  --task-definition "$DEF"
  --launch-type FARGATE
  --region "$REGION"
  --network-configuration "awsvpcConfiguration={subnets=[${SUBNETS}],securityGroups=[${SG}],assignPublicIp=${ASSIGN_IP}}"
  --query "tasks[0].taskArn"
  --output text
)
if [ "${#ENV_VARS[@]}" -gt 0 ]; then
  IFS=,
  RUN_ARGS+=(--overrides "{\"containerOverrides\":[{\"name\":\"${CONTAINER}\",\"environment\":[${ENV_VARS[*]}]}]}")
  unset IFS
fi

TASK_ARN="$(aws ecs run-task "${RUN_ARGS[@]}")"

echo "Started $TASK_ARN"

if [ "$FOLLOW" -eq 0 ]; then
  exit 0
fi

echo "Waiting for CloudWatch logs from $LOG_GROUP"
echo "Ctrl+C stops the tail only; the Fargate task keeps running."
aws logs tail "$LOG_GROUP" --follow --since 2m --format short --region "$REGION"
