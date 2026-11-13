# LightGBM v6 live (2026-11-09, Hal asked): build, test on Fargate, then make it production's image.
#   1. incremental external-data crawl (features must cover the upcoming fixtures)
#   2. pauses the second lgbm_v6 manager run (v3_lgbm_v6_seed1) after a save, to free memory for Docker
#   3. starts Docker Desktop, builds the image from the working tree, runs the live tests inside it,
#      pushes ECR tag `lgbm`, then shuts Docker down and resumes the manager run
#   4. runs task definition premierlearning-lgbm-test (suggest only, emails Hal, never rebooks the schedule)
#   5. only if that task exits 0: points ECR `latest` (production, premierlearning-pipeline:8) at the same
#      image. Rollback: tag latest-2026-11-05 holds the previous production image.
# Log: api\run_metrics\fargate_lgbm_v6.log. Do not edit while it runs.
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\hal\Documents\git\premierlearningweb\premier-learning-app'
$api = Join-Path $root 'api'
$log = Join-Path $api 'run_metrics\fargate_lgbm_v6.log'
$region = 'eu-west-2'
$registry = '497087878047.dkr.ecr.eu-west-2.amazonaws.com'
$image = "$registry/premierlearning:lgbm"
$env:AWS_PAGER = ''
$env:PYTHONIOENCODING = 'utf-8'
function Say($text) { Add-Content -Path $log -Value ('{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $text) }
function Native($command) {
    # Through cmd so stderr progress lines go to the log as plain text.
    & cmd.exe /c "$command >> `"$log`" 2>&1"
    return $LASTEXITCODE
}
function Resume-Manager {
    $cmd = 'cmd.exe /c "' + $api + '\fpl_manager\run_manager_v3.cmd" 97 fpl_manager_runs\v3_lgbm_v6_seed1 "--seed 1"'
    $made = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $api }
    Say ('Resumed manager run v3_lgbm_v6_seed1, PID {0}' -f $made.ProcessId)
}

Say '===== Started'

# 1. External data.
Native ('"' + $api + '\premierlearning\external_data\run_crawl.cmd"') | Out-Null
$crawl = Select-String -Path (Join-Path $api 'data\external\crawl.log') -Pattern '^==== crawl exit' | Select-Object -Last 1
Say ('Crawl: {0}' -f $crawl.Line)
if ($crawl.Line -notmatch 'crawl exit 0 ') { Say 'Crawl failed; stopping before the build'; exit 1 }

# 2. Memory for Docker.
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $api 'fpl_manager\stop_after_save.ps1') -Names v3_lgbm_v6_seed1 -MaxWaitMinutes 20
Say ('Paused v3_lgbm_v6_seed1; free MB {0}' -f [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB))

# 3. Build, test inside the image, push.
Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
$ready = $false
for ($i = 0; $i -lt 60 -and -not $ready; $i++) {
    Start-Sleep 10
    & cmd.exe /c 'docker info >nul 2>&1'
    $ready = ($LASTEXITCODE -eq 0)
}
if (-not $ready) { Say 'Docker did not start in 10 minutes'; Resume-Manager; exit 1 }
Say 'Docker ready'
$code = Native ("aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $registry")
if ($code -ne 0) { Say "ECR login failed ($code)"; Resume-Manager; exit 1 }
Set-Location $root
$code = Native ("docker build --platform linux/amd64 -t $image .")
Say "Build exit $code"
$tests = 'tests.test_lgbm_news tests.test_lgbm tests.test_live_blend tests.test_live_data_update tests.test_live_strategy tests.test_chance_to_play'
if ($code -eq 0) {
    $code = Native ("docker run --rm --entrypoint python $image -m unittest $tests")
    Say "Tests inside the image exit $code"
}
if ($code -eq 0) {
    $code = Native ("docker push $image")
    Say "Push exit $code"
}
Stop-Process -Name 'Docker Desktop' -Force -ErrorAction SilentlyContinue
& wsl.exe --shutdown
Say 'Docker shut down'
Resume-Manager
if ($code -ne 0) { Say 'Stopping: build, tests or push failed; nothing was run on Fargate'; exit 1 }

# 4. Suggest-only test on Fargate (emails Hal).
$network = 'awsvpcConfiguration={subnets=[subnet-0bfd69b285666ee0e,subnet-0b15a9d02a14cee62],securityGroups=[sg-06cff2e0d1497d24b],assignPublicIp=DISABLED}'
$task = (& aws ecs run-task --cluster premierlearning --task-definition premierlearning-lgbm-test --launch-type FARGATE --region $region --network-configuration $network --query 'tasks[0].taskArn' --output text).Trim()
Say "Fargate test task $task"
if (-not $task -or $task -eq 'None') { Say 'run-task failed'; exit 1 }
$status = ''
for ($i = 0; $i -lt 240 -and $status -ne 'STOPPED'; $i++) {
    Start-Sleep 60
    $status = (& aws ecs describe-tasks --cluster premierlearning --tasks $task --region $region --query 'tasks[0].lastStatus' --output text).Trim()
}
$exit = (& aws ecs describe-tasks --cluster premierlearning --tasks $task --region $region --query 'tasks[0].containers[0].exitCode' --output text).Trim()
$reason = (& aws ecs describe-tasks --cluster premierlearning --tasks $task --region $region --query 'tasks[0].stoppedReason' --output text).Trim()
Say "Fargate test: status $status, exit $exit, $reason"
if ($exit -ne '0') { Say 'Test failed: latest (production) left unchanged'; exit 1 }

# 5. Production image.
$manifest = & aws ecr batch-get-image --repository-name premierlearning --region $region --image-ids imageTag=lgbm --query 'images[0].imageManifest' --output text
$media = (& aws ecr batch-get-image --repository-name premierlearning --region $region --image-ids imageTag=lgbm --query 'images[0].imageManifestMediaType' --output text).Trim()
$manifestFile = Join-Path $env:TEMP 'lgbm_v6_manifest.json'
[IO.File]::WriteAllText($manifestFile, ($manifest -join "`n"))
$code = Native ("aws ecr put-image --repository-name premierlearning --region $region --image-tag latest --image-manifest file://$manifestFile --image-manifest-media-type $media")
Say "latest -> lgbm image, exit $code"
Say '===== Finished'
