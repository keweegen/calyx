@echo off
rem Run one shard of the window-gap diagnostic (measurement V1c-E6.2) under MSVC.
rem Rule D1: permitted only because stamp 0 of stage V1c was hashed first, see
rem results/v1c/stamp0_sha256.txt. Resolution refinement is sequential and runs
rem through run_refine.sh, not through shards.
setlocal
set SHARD=%1
set OF=%2
if "%SHARD%"=="" set SHARD=0
if "%OF%"=="" set OF=1
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set VSPATH=%%i
if not defined VSPATH (
  echo Visual Studio not found via vswhere
  exit /b 1
)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set DISTUTILS_USE_SDK=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if not exist "results\diag_window\logs" mkdir "results\diag_window\logs"
.\.venv\Scripts\python.exe v1b_window_diag.py --scan --shard %SHARD% --of %OF% > "results\diag_window\logs\wdiag_shard%SHARD%.log" 2>&1
endlocal
