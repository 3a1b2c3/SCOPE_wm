@echo off
setlocal enableextensions enabledelayedexpansion

REM Run a SCOPE inference example end-to-end.
REM
REM Usage:
REM   run_example.bat                  example_0 (It Takes Two), default settings
REM   run_example.bat 0                same
REM   run_example.bat 1                example_1 (Genshin Impact)
REM   run_example.bat 2                example_2 (Black Myth: Wukong)
REM   run_example.bat <idx> --steps 50 forward extra flags to inference.py
REM
REM Output mp4 lands in: %~dp0outputs\example_<idx>\
REM Requires: model weights downloaded via download_models.bat first.

set "REPO_ROOT=%~dp0"
set "VENV_PY=%REPO_ROOT%.venv\Scripts\python.exe"

if not exist "!VENV_PY!" (
    echo [run_example] ERROR: venv not found at !VENV_PY!
    echo Run: uv venv --python 3.10 .venv  ^&^&  uv pip install -r requirements.txt -e .
    exit /b 1
)

REM First positional arg = example index. Default to 0 if not given OR if it
REM starts with '-' (i.e. caller passed only flags).
set "IDX=%~1"
set "FIRST_CHAR=%IDX:~0,1%"
if "%IDX%"=="" set "IDX=0"
if "%FIRST_CHAR%"=="-" set "IDX=0"
if not "%FIRST_CHAR%"=="-" shift

set "EXAMPLE_DIR=%REPO_ROOT%examples\example_%IDX%"
if not exist "!EXAMPLE_DIR!\image.png" (
    echo [run_example] ERROR: example_%IDX% not found at !EXAMPLE_DIR!
    echo Available:
    dir /b /ad "%REPO_ROOT%examples"
    exit /b 1
)

REM Resolve the SCOPE model dir. Prefer a local checkout, fall back to the HF
REM hub cache (where download_models.bat lands the weights when no --local-dir
REM is passed). Resolving via snapshot_download with local_files_only=True
REM returns the cache path; if nothing is downloaded yet, it errors with
REM something more useful than a missing-file traceback later in inference.py.
set "MODEL_DIR=%REPO_ROOT%SCOPE"
if not exist "!MODEL_DIR!\config.json" if not exist "!MODEL_DIR!\model_index.json" (
    for /f "usebackq delims=" %%P in (`"!VENV_PY!" -c "from huggingface_hub import snapshot_download; print(snapshot_download('zizhaotong/SCOPE', local_files_only=True))" 2^>nul`) do set "MODEL_DIR=%%P"
)

if not exist "!MODEL_DIR!" (
    echo [run_example] ERROR: SCOPE model not found.
    echo Run: download_models.bat
    exit /b 1
)

REM Read prompt from prompt.txt (single line) for the --prompt arg.
set "PROMPT="
for /f "usebackq delims=" %%L in ("!EXAMPLE_DIR!\prompt.txt") do (
    if defined PROMPT (set "PROMPT=!PROMPT! %%L") else (set "PROMPT=%%L")
)

set "OUT_DIR=%REPO_ROOT%outputs\example_%IDX%"
if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"

echo ============================================================
echo SCOPE example_%IDX%
echo ============================================================
echo   model    : !MODEL_DIR!
echo   image    : !EXAMPLE_DIR!\image.png
echo   action   : !EXAMPLE_DIR!\action.parquet
echo   out      : !OUT_DIR!
echo ============================================================

"!VENV_PY!" "%REPO_ROOT%inference.py" --model_dir "!MODEL_DIR!" --input_image "!EXAMPLE_DIR!\image.png" --action_path "!EXAMPLE_DIR!\action.parquet" --prompt "!PROMPT!" --output_dir "!OUT_DIR!" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not %EXIT_CODE%==0 (
    echo [run_example] inference.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)
echo [run_example] Done. Output in !OUT_DIR!
endlocal
