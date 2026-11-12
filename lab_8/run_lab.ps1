param([string]$Python = 'python', [switch]$Notebook, [switch]$Reuse)

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:CUDA_VISIBLE_DEVICES = '-1'
    $env:PYTHONUTF8 = '1'
    $environments = [ordered]@{
        '.venv' = 'requirements.txt'
        '.venv-k' = 'requirements-k.txt'
        '.venv-mwem' = 'requirements-mwem.txt'
        '.venv-mostly' = 'requirements-mostly.txt'
    }
    foreach ($name in $environments.Keys) {
        $interpreter = Join-Path $name 'Scripts/python.exe'
        if (!(Test-Path -LiteralPath $interpreter)) {
            & $Python -m venv $name
            if ($LASTEXITCODE -ne 0) { throw "Не удалось создать окружение $name" }
        }
        & ".\$interpreter" -m pip --disable-pip-version-check install -r $environments[$name]
        if ($LASTEXITCODE -ne 0) { throw "Не удалось установить зависимости для $name" }
    }
    if ($Notebook) {
        & '.\.venv\Scripts\python.exe' -m jupyterlab lab_8.ipynb
    } elseif ($Reuse) {
        & '.\.venv\Scripts\python.exe' run.py --reuse
    } else {
        & '.\.venv\Scripts\python.exe' run.py
    }
    if ($LASTEXITCODE -ne 0) { throw 'Прогон лабораторной завершился с ошибкой' }
} finally {
    Pop-Location
}
