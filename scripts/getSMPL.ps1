# Save as smplify_download.ps1
function UrlEncode([string]$str) {
    return [uri]::EscapeDataString($str)
}

Write-Host "`nYou need to register at https://smplify.is.tue.mpg.de"

$username = Read-Host "Username (SMPLify)"
$password = Read-Host "Password (SMPLify)"
$username = UrlEncode $username
$password = UrlEncode $password

$datasetPath = "dataset/body_models"
$smplPath = "$datasetPath/smpl"
$zipPath = "$datasetPath/smplify.zip"
$extractPath = "$datasetPath/smplify"

# Create directories
New-Item -ItemType Directory -Force -Path $smplPath | Out-Null

# Download
Invoke-WebRequest -Uri "https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip" `
    -Method Post `
    -Body "username=$username&password=$password" `
    -OutFile $zipPath `
    -UseBasicParsing

# Unzip
Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

# Move
Move-Item "$extractPath/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl" `
          "$smplPath/SMPL_NEUTRAL.pkl" -Force

# Cleanup
Remove-Item $extractPath -Recurse -Force
Remove-Item $zipPath -Force

Write-Host "✅ Download and setup complete"