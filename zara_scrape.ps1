# Set paths
$csvPath = "zara_measures_data.csv"
$outputDir = "downloaded_images"

# Create output directory if it doesn't exist
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

# Read the CSV
$rows = Import-Csv $csvPath

foreach ($row in $rows) {
    $articleId = $row.ID_ARTICLE
    $pictureField = $row.PICTURES

    # Clean and convert stringified list to actual list
    $urls = $pictureField -replace "`n", "" -replace "\[|\]" , "" -split "," | ForEach-Object {
        $_.Trim().Trim('"')
    }

    $i = 1
    foreach ($url in $urls) {
        if ([string]::IsNullOrWhiteSpace($url)) { continue }

        # Determine filename
        $suffix = if ($urls.Count -gt 1) { "_$i" } else { "" }
        $filename = "${articleId}${suffix}.jpg"
        $filePath = Join-Path $outputDir $filename

        # Download the image
        try {
            Invoke-WebRequest -Uri $url -OutFile $filePath -TimeoutSec 15
            Write-Host "Downloaded: $filePath"
        } catch {
            Write-Warning "Failed to download $url for ID $articleId"
        }

        $i++
    }
}
