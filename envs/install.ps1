# Cài drive2sim trên Windows bằng một lệnh (docs/install.md): Studio + CLI.
#
#   powershell -ExecutionPolicy Bypass -File envs\install.ps1 [-Workdir D:\d2s-work] [-Port 8765] [-Start]
#
# Windows chạy Studio: quản lý cảnh, chạy run trên GPU của Kaggle, Colab, xem kết quả, so sánh.
# Train trên máy cần Linux có GPU NVIDIA: trên Windows có GPU NVIDIA, cài WSL2 (Ubuntu) rồi chạy
# `bash envs/install.sh` bên trong WSL.
#
# Mọi thứ nằm trong $env:D2S_HOME (mặc định %USERPROFILE%\.drive2sim): venv\, bin\drive2sim-studio.cmd.
# Chạy lại an toàn: venv đúng bản được dùng lại.
param(
    [string]$Workdir = $(if ($env:D2S_WORKDIR) { $env:D2S_WORKDIR } else { Join-Path $HOME "drive2sim-work" }),
    [int]$Port = 8765,
    [switch]$Start,
    [switch]$NoPath,     # không thêm bin\ vào PATH của người dùng
    [switch]$SkipTools   # không cài Kaggle CLI, Colab CLI (uv tool)
)
$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }
$Src = Split-Path -Parent $PSScriptRoot
$D2sHome = if ($env:D2S_HOME) { $env:D2S_HOME } else { Join-Path $HOME ".drive2sim" }
$Venv = Join-Path $D2sHome "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
function Step($text) { Write-Host ""; Write-Host ("=== {0} {1}" -f (Get-Date -Format HH:mm:ss), $text) }
# không khai tham số: "-e" của uv pip install sẽ bị PowerShell hiểu là tham số của hàm
function Run {
    $rest = @($args | Select-Object -Skip 1)
    & $args[0] @rest
    if ($LASTEXITCODE -ne 0) { throw "lệnh lỗi ($LASTEXITCODE): $args" }
}
Write-Host "drive2sim: chế độ studio, mã nguồn $Src, venv $Venv, thư mục làm việc $Workdir"

Step "1 uv (quản lý Python)"
$env:Path = (Join-Path $HOME ".local\bin") + ";" + $env:Path
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    # trình cài chính thức của uv (https://docs.astral.sh/uv/getting-started/installation/)
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = (Join-Path $HOME ".local\bin") + ";" + $env:Path
}
Run uv --version

Step "2 venv Python 3.11"
$reuse = $false
if (Test-Path $Py) {
    & $Py -c "import sys; sys.exit(sys.version_info[:2] != (3, 11))"
    $reuse = ($LASTEXITCODE -eq 0)
}
if ($reuse) {
    Write-Host "dùng lại $Venv"
} else {
    if (Test-Path $Venv) { Remove-Item -Recurse -Force $Venv }
    New-Item -ItemType Directory -Force $D2sHome | Out-Null
    Run uv venv --seed --python 3.11 $Venv
}

Step "3 drive2sim + Studio"
# Windows khoá d2s.exe khi Studio đang chạy: cài đè sẽ lỗi "used by another process"
$D2sExe = Join-Path $Venv "Scripts\d2s.exe"
$busy = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $D2sExe }
if ($busy) { throw "Studio đang chạy từ venv này (pid $($busy.Id -join ', ')): tắt nó trước (npx github:thaianh20021/drive2sim stop, hoặc đóng cửa sổ Studio) rồi chạy lại" }
Run uv pip install --python $Py -e "$Src[studio]"

Step "4 Kaggle CLI, Colab CLI"
if ($SkipTools) { Write-Host "bỏ qua (-SkipTools)" } else {
& uv tool install --upgrade kaggle
if ($LASTEXITCODE -ne 0) { Write-Host "chưa cài được Kaggle CLI: cài sau bằng uv tool install kaggle" }
& uv tool install --upgrade google-colab-cli --with keyrings.alt
if ($LASTEXITCODE -ne 0) { Write-Host "chưa cài được Colab CLI: cài sau ở trang Thư viện của Studio" }
}

Step "5 lệnh bật Studio"
New-Item -ItemType Directory -Force $Workdir | Out-Null
$Bin = Join-Path $D2sHome "bin"
New-Item -ItemType Directory -Force $Bin | Out-Null
$Launcher = Join-Path $Bin "drive2sim-studio.cmd"
$D2s = Join-Path $Venv "Scripts\d2s.exe"
@"
@echo off
rem Start drive2sim Studio (written by envs\install.ps1). Extra arguments go to d2s studio.
set PYTHONIOENCODING=utf-8
set PATH=%USERPROFILE%\.local\bin;%PATH%
if "%D2S_WORKDIR%"=="" (set "W=$Workdir") else (set "W=%D2S_WORKDIR%")
if "%D2S_PORT%"=="" (set "P=$Port") else (set "P=%D2S_PORT%")
"$D2s" studio --workdir "%W%" --port %P% %*
"@ | Set-Content -Encoding ASCII $Launcher
# đưa bin\ vào PATH của người dùng (một lần)
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $NoPath -and -not ($userPath -split ";" | Where-Object { $_ -eq $Bin })) {
    [Environment]::SetEnvironmentVariable("Path", "$Bin;$userPath", "User")
}

Write-Host ""
Write-Host "Xong (chế độ studio)."
Write-Host "  Bật Studio:  drive2sim-studio   (mở cửa sổ dòng lệnh mới; hoặc $Launcher)"
Write-Host "  Mở:          http://127.0.0.1:$Port"
Write-Host "  Lệnh CLI:    $D2s --help"
Write-Host "  Chạy run trên Kaggle hoặc Colab (Chạy mới -> Nơi chạy). Train trên máy: WSL2 + bash envs/install.sh."
if ($Start) { & $Launcher }
