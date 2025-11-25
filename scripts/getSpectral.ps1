
$datasetPath = "dataset/spectral"
New-Item -ItemType Directory -Force -Path $datasetPath | Out-Null

$files = @(
    @{ url = "https://github.com/riccardomarin/EG22_Tutorial_Spectral_Geometry/raw/main/data/minnesota_g.mat";  out = "minnesota_g.mat" },
    @{ url = "https://raw.githubusercontent.com/riccardomarin/EG22_Tutorial_Spectral_Geometry/main/data/tr_reg_090.off"; out = "tr_reg_090.off" },
    @{ url = "https://raw.githubusercontent.com/riccardomarin/EG22_Tutorial_Spectral_Geometry/main/data/tr_reg_043.off"; out = "tr_reg_043.off" },
    @{ url = "https://github.com/riccardomarin/EG22_Tutorial_Spectral_Geometry/raw/main/data/pose.mat"; out = "pose.mat" },
    @{ url = "https://github.com/riccardomarin/EG22_Tutorial_Spectral_Geometry/raw/main/data/style.mat"; out = "style.mat" }
)

foreach ($f in $files) {
    $outfile = Join-Path $datasetPath $f.out
    if (-Not (Test-Path $outfile)) {
        Write-Host "Downloading $($f.out)..."
        Invoke-WebRequest -Uri $f.url -OutFile $outfile -UseBasicParsing
    } else {
        Write-Host "Found existing $($f.out), skipping download."
    }
}

Write-Host "All spectral geometry files are ready in $datasetPath"