#!/usr/bin/env bash
# Git Bash on Windows often omits the Terraform installer directory.
if ! command -v terraform >/dev/null 2>&1; then
  localapp="${LOCALAPPDATA:-}"
  if [ -n "$localapp" ]; then
    tf_dir="$localapp/Programs/Terraform"
    if [ -x "$tf_dir/terraform.exe" ] || [ -x "$tf_dir/terraform" ]; then
      PATH="$tf_dir:$PATH"
    fi
  fi
fi
if ! command -v terraform >/dev/null 2>&1; then
  echo "terraform: command not found (looked on PATH and %LOCALAPPDATA%/Programs/Terraform)" >&2
  exit 1
fi
