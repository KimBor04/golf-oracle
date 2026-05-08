$ErrorActionPreference = "Stop"

# Go to repository root
$repoRoot = git rev-parse --show-toplevel
Set-Location $repoRoot

$workflowFile = "api_backfill.yml"
$tempDir = Join-Path $repoRoot ".tmp_api_artifacts"
$featuresDir = Join-Path $repoRoot "features"
$apiCacheDir = Join-Path $featuresDir "api_cache"

Write-Host "Repository root: $repoRoot"
Write-Host "Checking GitHub CLI login..."

gh auth status

Write-Host ""
Write-Host "Looking for latest successful API Backfill workflow run..."

$runId = gh run list `
    --workflow $workflowFile `
    --status success `
    --limit 1 `
    --json databaseId `
    --jq ".[0].databaseId"

if ([string]::IsNullOrWhiteSpace($runId)) {
    throw "No successful API Backfill workflow run found."
}

Write-Host "Latest successful run ID: $runId"

if (Test-Path $tempDir) {
    Remove-Item $tempDir -Recurse -Force
}

New-Item -ItemType Directory -Path $tempDir | Out-Null
New-Item -ItemType Directory -Path $featuresDir -Force | Out-Null
New-Item -ItemType Directory -Path $apiCacheDir -Force | Out-Null

Write-Host ""
Write-Host "Downloading artifacts from GitHub Actions..."

gh run download $runId --dir $tempDir

Write-Host ""
Write-Host "Copying API parquet artifacts into features/..."

$parquetPatterns = @(
    "api_schedule_*.parquet",
    "api_target_tournaments_*.parquet",
    "api_results_*.parquet",
    "api_fields_*.parquet",
    "live_features.parquet"
)

foreach ($pattern in $parquetPatterns) {
    $files = Get-ChildItem -Path $tempDir -Recurse -File -Filter $pattern

    foreach ($file in $files) {
        $target = Join-Path $featuresDir $file.Name
        Copy-Item $file.FullName $target -Force
        Write-Host "Copied $($file.Name) -> features/"
    }
}

Write-Host ""
Write-Host "Copying API cache files..."

$cacheDirs = Get-ChildItem -Path $tempDir -Recurse -Directory | Where-Object {
    $_.Name -eq "api_cache"
}

foreach ($cacheDir in $cacheDirs) {
    Copy-Item "$($cacheDir.FullName)\*" $apiCacheDir -Recurse -Force
    Write-Host "Copied api_cache/ files -> features/api_cache/"
}

Write-Host ""
Write-Host "Cleaning temporary download folder..."

Remove-Item $tempDir -Recurse -Force

Write-Host ""
Write-Host "Done. API artifacts are now available locally in features/."
Write-Host ""
Write-Host "Current API feature artifacts:"

Get-ChildItem -Path $featuresDir -File | Where-Object {
    $_.Name -like "api_*.parquet" -or $_.Name -eq "live_features.parquet"
} | Sort-Object Name | Select-Object Name, Length, LastWriteTime