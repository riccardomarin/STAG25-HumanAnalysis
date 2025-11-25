# Save as amass_download.ps1
function UrlEncode([string]$str) {
    return [uri]::EscapeDataString($str)
}

Write-Host "`nYou need to register at https://amass.is.tue.mpg.de"

$username = Read-Host "Username (AMASS)"
$password = Read-Host "Password (AMASS)"
$username = UrlEncode $username
$password = UrlEncode $password

$datasetPath = "dataset/motions"
$zipPath = "poses.tar.bz2"

# Create directories
New-Item -ItemType Directory -Force -Path $datasetPath | Out-Null

# Download only if file not already present
if (-Not (Test-Path $zipPath)) {
    Write-Host "Downloading DanceDB archive..."
    Invoke-WebRequest -Uri "https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/DanceDB.tar.bz2" `
        -Method Post `
        -Body "username=$username&password=$password" `
        -OutFile $zipPath `
        -UseBasicParsing
} else {
    Write-Host "Found existing $zipPath, skipping download."
}

# Extract if target file not already there
$targetFile = Join-Path $datasetPath "poses.npz"
if (-Not (Test-Path $targetFile)) {
    if (Get-Command 7z -ErrorAction SilentlyContinue) {
        Write-Host "Extracting with 7-Zip..."
        & 7z x $zipPath -so | 7z x -aoa -si -ttar
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        Write-Host "Extracting with Python..."
        $pycode = @"
import tarfile
with tarfile.open("poses.tar.bz2", "r:bz2") as tar:
    tar.extractall()
"@
        $pycode | python
    }
    else {
        Write-Host "Neither 7-Zip nor Python is available. Please install one to extract .tar.bz2 files."
        exit 1
    }

    # Move the required file
    Move-Item "DanceDB/20120731_StefanosTheodorou/Stefanos_1os_antrikos_karsilamas_C3D_poses.npz" `
              $targetFile -Force

    # Cleanup extracted folder
    Remove-Item "DanceDB" -Recurse -Force
    Write-Host "Extraction complete"
} else {
    Write-Host "Found existing $targetFile, skipping extraction."
}

# Optionally remove archive to save space
# Remove-Item $zipPath -Force

Write-Host "AMASS DanceDB pose ready at $targetFile"