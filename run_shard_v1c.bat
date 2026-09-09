@echo off
rem Run one shard of stage V1c under the MSVC environment.
rem
rem Brian 2 with codegen.target='cython' compiles through setuptools, which on
rem Windows often fails to locate Visual Studio on its own. vcvars sets the
rem environment up and DISTUTILS_USE_SDK tells distutils to use it.
rem
rem Comments here are ASCII on purpose: the console runs .bat files under an
rem OEM code page, and UTF-8 Cyrillic in a batch file breaks the vswhere line.
rem
rem Usage:  run_shard_v1c.bat <shard> <of>
setlocal
set SHARD=%1
set OF=%2
if "%SHARD%"=="" set SHARD=0
if "%OF%"=="" set OF=8

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
if not exist "results\v1c\logs" mkdir "results\v1c\logs"
.\.venv\Scripts\python.exe run_v1c_grid.py --shard %SHARD% --of %OF% > "results\v1c\logs\shard%SHARD%.log" 2>&1
endlocal
