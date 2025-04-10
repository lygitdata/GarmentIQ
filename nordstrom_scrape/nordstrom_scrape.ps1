param(
    [string]$WorkingDir,
    [string]$GarmentType,
    [string]$InputJsonFile,
    [string]$OutputJsonFile,
    [string]$ImageDir
)

# Move to working directory
Set-Location $WorkingDir

# Ensure image directory exists
if (!(Test-Path $ImageDir)) {
    New-Item -ItemType Directory -Path $ImageDir | Out-Null
}

# Load input JSON
$jsonPath = Join-Path $WorkingDir $InputJsonFile
$json = Get-Content $jsonPath | ConvertFrom-Json

# Prepare temporary list
$temp = @()
$counter = 0

# Extract image URLs
foreach ($entry in $json.log.entries) {
    $url = $entry.request.url
    if ($url) {
        $key = "${GarmentType}_$counter.jpg"
        $temp += [PSCustomObject]@{ Index = $counter; Key = $key; Url = $url }
        $counter++
    }
}

# Sort and build ordered result
$sorted = $temp | Sort-Object Index
$result = [ordered]@{}
foreach ($item in $sorted) {
    $result[$item.Key] = $item.Url
}

# Save metadata
$outputPath = Join-Path $WorkingDir $OutputJsonFile
$result | ConvertTo-Json -Depth 2 | Out-File $outputPath
Write-Host "Saved metadata to $OutputJsonFile"

# Download images with progress
$totalItems = $result.Count
$currentItem = 0
$successCount = 0
$allDownloaded = $true

foreach ($key in $result.Keys) {
    $currentItem++
    $url = $result[$key]
    $imgPath = Join-Path $ImageDir $key

    # Update progress bar
    $percentComplete = ($currentItem / $totalItems) * 100
    Write-Progress -Activity "Downloading $GarmentType Images" `
                   -Status "Progress: $currentItem/$totalItems ($([math]::Round($percentComplete))%)" `
                   -PercentComplete $percentComplete `
                   -CurrentOperation $key

    try {
        Invoke-WebRequest -Uri $url -OutFile $imgPath -ErrorAction Stop
        $successCount++
    }
    catch {
        Write-Warning "Failed to download $key from $url"
        $allDownloaded = $false
    }
}

# Clear progress bar
Write-Progress -Activity "Downloading $GarmentType Images" -Completed

# Show final download count
Write-Host "Downloaded $successCount/$totalItems images"

# Delete input JSON if all downloads were successful
if ($allDownloaded) {
    Remove-Item $jsonPath -Force
    Write-Host "All downloads completed. Deleted input file: $InputJsonFile"
}
else {
    Write-Warning "Some downloads failed. Input file NOT deleted."
}

Write-Host "Process complete for garment type '$GarmentType'"