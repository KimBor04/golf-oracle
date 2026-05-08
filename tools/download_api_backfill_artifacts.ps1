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

if (Test-Path $tempDir) {
    Remove-Item $tempDir -Recurse -Force
}

New-Item -ItemType Directory -Path $tempDir | Out-Null
New-Item -ItemType Directory -Path $featuresDir -Force | Out-Null
New-Item -ItemType Directory -Path $apiCacheDir -Force | Out-Null

function Copy-DownloadedArtifacts {
    param (
        [string]$SourceDir
    )

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
        $files = Get-ChildItem -Path $SourceDir -Recurse -File -Filter $pattern

        foreach ($file in $files) {
            $target = Join-Path $featuresDir $file.Name
            Copy-Item $file.FullName $target -Force
            Write-Host "Copied $($file.Name) -> features/"
        }
    }

    Write-Host ""
    Write-Host "Copying API cache files..."

    $cacheDirs = Get-ChildItem -Path $SourceDir -Recurse -Directory | Where-Object {
        $_.Name -eq "api_cache"
    }

    foreach ($cacheDir in $cacheDirs) {
        Copy-Item "$($cacheDir.FullName)\*" $apiCacheDir -Recurse -Force
        Write-Host "Copied api_cache/ files -> features/api_cache/"
    }
}

function Download-LatestArtifactForMode {
    param (
        [string]$Mode
    )

    Write-Host ""
    Write-Host "Looking for latest successful API Backfill workflow artifact for mode: $Mode"

    $runs = gh run list `
        --workflow $workflowFile `
        --status success `
        --limit 20 `
        --json databaseId `
        | ConvertFrom-Json

    foreach ($run in $runs) {
        $runId = $run.databaseId
        $modeTempDir = Join-Path $tempDir $Mode
        $runTempDir = Join-Path $modeTempDir "run_$runId"

        New-Item -ItemType Directory -Path $runTempDir -Force | Out-Null

        Write-Host "Checking run ID: $runId"

        gh run download $runId --dir $runTempDir 2>$null

        $matchingArtifactFiles = Get-ChildItem -Path $runTempDir -Recurse -File | Where-Object {
            $_.FullName -like "*api-backfill-$Mode-*"
        }

        $matchingArtifactDirs = Get-ChildItem -Path $runTempDir -Recurse -Directory | Where-Object {
            $_.Name -like "api-backfill-$Mode-*"
        }

        if ($matchingArtifactFiles.Count -gt 0 -or $matchingArtifactDirs.Count -gt 0) {
            Write-Host "Found latest $Mode artifact in run ID: $runId"
            Copy-DownloadedArtifacts -SourceDir $runTempDir
            return
        }

        Remove-Item $runTempDir -Recurse -Force
    }

    Write-Warning "No successful artifact found for mode: $Mode"
}

Download-LatestArtifactForMode -Mode "fields"
Download-LatestArtifactForMode -Mode "results"

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

Write-Host ""
Write-Host "Current API cache file count:"

$cacheFileCount = (Get-ChildItem -Path $apiCacheDir -File -Recurse | Measure-Object).Count
Write-Host "$cacheFileCount cached API files in features/api_cache/"