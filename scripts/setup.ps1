[CmdletBinding()]
param(
    [string]$PythonExe
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvDirectory = Join-Path $projectRoot '.venv'
$venvPython = Join-Path $venvDirectory 'Scripts\python.exe'

function Test-Python312x64 {
    param(
        [string]$Executable,
        [string[]]$Arguments = @()
    )
    $probe = "import platform, struct, sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8 and platform.machine().lower() in ('amd64', 'x86_64') else 1)"
    try {
        & $Executable @Arguments -c $probe 1>$null 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

$previousAutomaticInstall = $env:PYTHON_MANAGER_AUTOMATIC_INSTALL
Push-Location -LiteralPath $projectRoot
try {
    # The modern py/python aliases may otherwise install a missing runtime.
    $env:PYTHON_MANAGER_AUTOMATIC_INSTALL = 'false'
    if (Test-Path -LiteralPath $venvDirectory) {
        if (-not (Test-Path -LiteralPath (Join-Path $venvDirectory 'pyvenv.cfg') -PathType Leaf) -or
            -not (Test-Python312x64 -Executable $venvPython)) {
            throw "Existing .venv is not a usable Python 3.12 x64 environment. Preserve it first with Rename-Item -LiteralPath .venv -NewName .venv-incompatible-backup (choose an unused backup name), then rerun setup."
        }
    }
    else {
        $candidates = @()
        if ($PythonExe) {
            if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
                throw "Python executable not found: $PythonExe"
            }
            $candidates += [pscustomobject]@{
                Executable = (Resolve-Path -LiteralPath $PythonExe).ProviderPath
                Arguments = @()
            }
        }
        else {
            if ($env:LOCALAPPDATA) {
                foreach ($relativePath in @('Programs\Python\Python312\python.exe', 'Python\pythoncore-3.12-64\python.exe')) {
                    $candidatePath = Join-Path $env:LOCALAPPDATA $relativePath
                    if (Test-Path -LiteralPath $candidatePath -PathType Leaf -ErrorAction SilentlyContinue) {
                        $candidates += [pscustomobject]@{ Executable = $candidatePath; Arguments = @() }
                    }
                }
            }
            if ($env:ProgramFiles) {
                $candidatePath = Join-Path $env:ProgramFiles 'Python312\python.exe'
                if (Test-Path -LiteralPath $candidatePath -PathType Leaf -ErrorAction SilentlyContinue) {
                    $candidates += [pscustomobject]@{ Executable = $candidatePath; Arguments = @() }
                }
            }
            $launcher = Get-Command -Name 'py' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($launcher) {
                $candidates += [pscustomobject]@{ Executable = $launcher.Source; Arguments = @('-3.12') }
            }
            foreach ($commandName in @('python3.12', 'python3', 'python')) {
                $command = Get-Command -Name $commandName -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($command) {
                    $candidates += [pscustomobject]@{ Executable = $command.Source; Arguments = @() }
                }
            }
        }

        $selected = $null
        foreach ($candidate in $candidates) {
            if (Test-Python312x64 -Executable $candidate.Executable -Arguments $candidate.Arguments) {
                $selected = $candidate
                break
            }
        }
        if (-not $selected) {
            throw 'Python 3.12 x64 is required. Install it from python.org, then rerun setup; use -PythonExe "C:\path\to\Python312\python.exe" if it is not discovered. Python 3.15 cannot use the pinned PyTorch wheels.'
        }

        $selectedExecutable = $selected.Executable
        $selectedArguments = @($selected.Arguments)
        Write-Host "Creating .venv with $selectedExecutable $selectedArguments"
        & $selectedExecutable @selectedArguments -m venv $venvDirectory
        if ($LASTEXITCODE -ne 0) {
            throw 'Creating the virtual environment failed.'
        }
        if (-not (Test-Python312x64 -Executable $venvPython)) {
            throw 'The created environment is not usable as Python 3.12 x64.'
        }
    }

    & $venvPython -m pip install --require-hashes -r (Join-Path $projectRoot 'requirements.lock')
    if ($LASTEXITCODE -ne 0) {
        throw 'Dependency installation failed. Resolve the pip error above, then rerun setup.'
    }
    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw 'Dependency verification failed. Review the pip check output above.'
    }
    Write-Host 'Setup complete. Run: .\.venv\Scripts\python.exe -m uvicorn persian_ocr.api:app --host 127.0.0.1 --port 8000'
}
finally {
    $env:PYTHON_MANAGER_AUTOMATIC_INSTALL = $previousAutomaticInstall
    Pop-Location
}
