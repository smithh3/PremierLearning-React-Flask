import json
import os

import boto3


def handler(event, _context):
    ecs = boto3.client("ecs")
    kwargs = {
        "cluster": os.environ["CLUSTER"],
        "taskDefinition": os.environ["TASK_DEFINITION"],
        "launchType": "FARGATE",
        "count": 1,
        "networkConfiguration": {
            "awsvpcConfiguration": {
                "subnets": os.environ["SUBNETS"].split(","),
                "securityGroups": os.environ["SECURITY_GROUPS"].split(","),
                "assignPublicIp": "DISABLED",
            }
        },
    }
    print("run_task", json.dumps({k: v for k, v in kwargs.items() if k != "networkConfiguration"}))
    response = ecs.run_task(**kwargs)
    failures = response.get("failures") or []
    if failures:
        raise RuntimeError("ecs.run_task failures: %s" % failures)
    arns = [task.get("taskArn") for task in response.get("tasks") or []]
    print("started", arns)
    return {"started": arns}
