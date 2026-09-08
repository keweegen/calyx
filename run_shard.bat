@echo off
rem Run one calibration shard of stage V1b under the MSVC environment.
rem
rem Brian 2 with codegen.target='cython' compiles through setuptools, which on
rem Windows often fails to locate Visual Studio on its own. vcvars sets the
rem environment up and DISTUTILS_USE_SDK tells distutils to use it.
rem
rem Usage:  run_shard.bat <shard> <of> [stage]
setlocal
set SHARD=%1
set OF=%2
set STAGE=%3
if "%SHARD%"=="" set SHARD=0
if "%OF%"=="" set OF=1
if "%STAGE%"=="" set STAGE=1

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
if not exist "results\v1b\logs" mkdir "results\v1b\logs"
.\.venv\Scripts\python.exe run_v1b_grid.py --shard %SHARD% --of %OF% --stage %STAGE% > "results\v1b\logs\s%STAGE%_shard%SHARD%.log" 2>&1
endlocal
