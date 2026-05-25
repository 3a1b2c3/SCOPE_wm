@echo off
setlocal enableextensions enabledelayedexpansion

REM Downloads the SCOPE model weights (~36 GB) into the HuggingFace cache.
REM No --local-dir is used, so files live only in:
REM   %USERPROFILE%\.cache\huggingface\hub\models--zizhaotong--SCOPE
REM Re-run safely: huggingface_hub resumes partial downloads.

set "REPO_ROOT=%~dp0"
set "VENV_PY=%REPO_ROOT%.venv\Scripts\python.exe"

if not exist "!VENV_PY!" (
    echo [download_models] ERROR: venv not found at !VENV_PY!
    echo Run: uv venv --python 3.10 .venv
    exit /b 1
)

echo [download_models] Downloading zizhaotong/SCOPE into HF cache...
REM Use snapshot_download directly (not the CLI shim) — recent huggingface_hub
REM versions dropped the ``huggingface_hub.commands`` submodule, so the older
REM ``-m huggingface_hub.commands.huggingface_cli download`` invocation breaks
REM with ModuleNotFoundError. max_workers=1 also dodges the Windows + py3.10
REM thread-shutdown race on Ctrl-C.
"!VENV_PY!" -c "from huggingface_hub import snapshot_download; p = snapshot_download('zizhaotong/SCOPE', max_workers=1); print(p)"
if errorlevel 1 (
    echo [download_models] download failed with exit code %errorlevel%
    exit /b %errorlevel%
)

echo [download_models] Done.
echo Cache path:
"!VENV_PY!" -c "from huggingface_hub import snapshot_download; print(snapshot_download('zizhaotong/SCOPE', local_files_only=True))"

endlocal
