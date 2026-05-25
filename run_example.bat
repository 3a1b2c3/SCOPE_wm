@echo off
setlocal enableextensions enabledelayedexpansion

REM Run SCOPE inference. With no args, runs every examples\example_* dir in
REM sorted order. Pass an index to run just one.
REM
REM Usage:
REM   run_example.bat                  every example_* dir (default = all)
REM   run_example.bat 0                only example_0 (It Takes Two)
REM   run_example.bat 1                only example_1 (Genshin Impact)
REM   run_example.bat 2                only example_2 (Black Myth: Wukong)
REM   run_example.bat 3                only example_3 (download.jpg)
REM   run_example.bat --steps 50       every example, forward flags to inference.py
REM   run_example.bat 2 --no_overlay   single example with extra flags
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

REM First positional arg = example index OR a flag. If it starts with '-' it's
REM a flag and we run all examples; otherwise it's a single-example index.
set "FIRST=%~1"
set "FIRST_CHAR=%FIRST:~0,1%"
set "RUN_ALL=0"
if "%FIRST%"=="" set "RUN_ALL=1"
if "%FIRST_CHAR%"=="-" set "RUN_ALL=1"

if "%RUN_ALL%"=="1" (
    REM Flags-only or no-args: run every example_* dir.
    set "INDICES="
    for /d %%D in ("%REPO_ROOT%examples\example_*") do (
        for /f "tokens=2 delims=_" %%I in ("%%~nxD") do (
            if defined INDICES (set "INDICES=!INDICES! %%I") else (set "INDICES=%%I")
        )
    )
    if not defined INDICES (
        echo [run_example] ERROR: no examples\example_* directories found
        exit /b 1
    )
) else (
    set "INDICES=%FIRST%"
    shift
)

REM Remaining args (after the optional index) forward to inference.py.
set "EXTRA_ARGS=%1"
:collect_extra
shift
if not "%~1"=="" (
    set "EXTRA_ARGS=!EXTRA_ARGS! %~1"
    goto collect_extra
)

REM Resolve the SCOPE model dir once for all examples.
set "MODEL_DIR=%REPO_ROOT%SCOPE"
if not exist "!MODEL_DIR!\config.json" if not exist "!MODEL_DIR!\model_index.json" (
    for /f "usebackq delims=" %%P in (`"!VENV_PY!" -c "from huggingface_hub import snapshot_download; print(snapshot_download('zizhaotong/SCOPE', local_files_only=True))" 2^>nul`) do set "MODEL_DIR=%%P"
)
if not exist "!MODEL_DIR!" (
    echo [run_example] ERROR: SCOPE model not found.
    echo Run: download_models.bat
    exit /b 1
)

set "FAIL_COUNT=0"
for %%I in (!INDICES!) do (
    set "IDX=%%I"
    set "EXAMPLE_DIR=%REPO_ROOT%examples\example_!IDX!"
    if not exist "!EXAMPLE_DIR!\image.png" (
        echo [run_example] SKIP example_!IDX!: image.png not found at !EXAMPLE_DIR!
        set /a FAIL_COUNT+=1
    ) else (
        set "PROMPT="
        for /f "usebackq delims=" %%L in ("!EXAMPLE_DIR!\prompt.txt") do (
            if defined PROMPT (set "PROMPT=!PROMPT! %%L") else (set "PROMPT=%%L")
        )

        set "OUT_DIR=%REPO_ROOT%outputs\example_!IDX!"
        if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"

        echo ============================================================
        echo SCOPE example_!IDX!
        echo ============================================================
        echo   model    : !MODEL_DIR!
        echo   image    : !EXAMPLE_DIR!\image.png
        echo   action   : !EXAMPLE_DIR!\action.parquet
        echo   out      : !OUT_DIR!
        echo ============================================================

        "!VENV_PY!" "%REPO_ROOT%inference.py" --model_dir "!MODEL_DIR!" --input_image "!EXAMPLE_DIR!\image.png" --action_path "!EXAMPLE_DIR!\action.parquet" --prompt "!PROMPT!" --output_dir "!OUT_DIR!" !EXTRA_ARGS!
        if not !ERRORLEVEL!==0 (
            echo [run_example] example_!IDX! exited with !ERRORLEVEL!
            set /a FAIL_COUNT+=1
        )
    )
)

if %FAIL_COUNT% gtr 0 (
    echo [run_example] %FAIL_COUNT% example^(s^) failed or skipped.
    exit /b 1
)
echo [run_example] All examples done.
endlocal
