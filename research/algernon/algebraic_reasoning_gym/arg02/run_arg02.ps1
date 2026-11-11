# ARG-02 GPU chain: development selection, then the single test run, then the report.
# One vramorama lease at a time; every GPU stage is resumable and exits 75 on a power cap.
#
#   powershell -ExecutionPolicy Bypass -File research\algernon\algebraic_reasoning_gym\arg02\run_arg02.ps1 `
#       -RunDir C:\Users\patri\Documents\Codex\algernon\runs\arg02
#
# Run from the Compactification-Theory root. A stage that already finished is skipped by the
# runner (parts are reused under an identical key), so the script can be started again after a stop.
param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [string]$Python = "C:\Users\patri\.venvs\bluebeam\Scripts\python.exe",
    [string]$EvalMathSrc = "C:\Users\patri\OneDrive\Documents\GitHub\EvalMath\src",
    [string]$Vram = "8G",
    [string]$Vramorama = "$env:LOCALAPPDATA\Programs\vramorama\vramorama.exe",
    [string]$ResultsDir = "research\algernon\algebraic_reasoning_gym\arg02\results"
)
# Exit codes decide failure: under Windows PowerShell 5.1 "Stop" would turn a Python
# warning on stderr into a terminating error.
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = ".;$EvalMathSrc"
$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$pkg = "research.algernon.algebraic_reasoning_gym.arg02"

function Cpu([string[]]$a) {
    & $Python -m "$pkg.analyze" @a
    if ($LASTEXITCODE -ne 0) { throw "analyze $($a -join ' ') failed ($LASTEXITCODE)" }
}

function Gpu([string]$name, [string[]]$a) {
    $done = Join-Path $RunDir "$name.done"
    if (Test-Path $done) { Write-Host "skip $name (done)"; return }
    & $Vramorama run --vram $Vram --owner "arg02-$name" --pause-exit 75 --guard-power 90 --restart 20 `
        -- $Python -m "$pkg.runner" @a --run-dir $RunDir
    if ($LASTEXITCODE -ne 0) { throw "runner $name failed ($LASTEXITCODE)" }
    New-Item -ItemType File -Path $done -Force | Out-Null
}

if (-not (Test-Path (Join-Path $RunDir "export_manifest.json"))) { Cpu @("export", "--run-dir", $RunDir) }
Gpu "dev_shots" @("dev", "--stage", "shots")
Cpu @("select", "--run-dir", $RunDir, "--stage", "shots")
Gpu "dev_blocks" @("dev", "--stage", "blocks")
if (-not (Test-Path (Join-Path $RunDir "selection.json"))) { Cpu @("select", "--run-dir", $RunDir, "--stage", "blocks") }
Gpu "test_main" @("test")
Gpu "test_ff" @("test", "--model", "ff")
Cpu @("report", "--run-dir", $RunDir, "--out", $ResultsDir)
Cpu @("replay", "--tape", "$ResultsDir\arg02_answer_tape.jsonl.gz", "--check", "$ResultsDir\arg02_receipt.json")
New-Item -ItemType File -Path (Join-Path $RunDir "chain.done") -Force | Out-Null
Write-Host "ARG-02 chain finished; receipts in $ResultsDir"
