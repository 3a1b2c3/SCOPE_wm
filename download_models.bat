@echo off
setlocal enableextensions enabledelayedexpansion

REM Downloads SCOPE model weights (~36 GB) directly into ./SCOPE via
REM huggingface_hub.snapshot_download(local_dir=...).  huggingface_hub >=0.23
REM writes files straight into local_dir without going through the
REM blobs/snapshots layout, so Windows symlink limitations no longer cause
REM 2x disk usage.  Re-run safely: snapshot_download resumes partial files.

set "REPO_ROOT=%~dp0"
set "VENV_PY=%REPO_ROOT%.venv\Scripts\python.exe"
set "TARGET_DIR=%REPO_ROOT%SCOPE"
set "DL_SCRIPT=%REPO_ROOT%_download_models.py"

if not exist "!VENV_PY!" (
    echo [download_models] ERROR: venv not found at !VENV_PY!
    echo Run: uv venv --python 3.10 .venv
    exit /b 1
)

if not exist "!DL_SCRIPT!" (
    echo [download_models] ERROR: _download_models.py not found at !DL_SCRIPT!
    exit /b 1
)

echo [download_models] Downloading zizhaotong/SCOPE into !TARGET_DIR! ...
REM Invoke the .py helper instead of `python -c "..."` to avoid cmd
REM delayed-expansion / quoting pitfalls on paths with backslashes.
REM --workers 1 dodges the Windows + py3.10 thread-shutdown race on Ctrl-C.
"!VENV_PY!" "!DL_SCRIPT!" --local-dir "!TARGET_DIR!" --workers 1
set "EXIT_CODE=!ERRORLEVEL!"
if not "!EXIT_CODE!"=="0" (
    echo [download_models] download failed with exit code !EXIT_CODE!
    exit /b !EXIT_CODE!
)

echo [download_models] Done.  Model dir:
echo   !TARGET_DIR!

endlocal
