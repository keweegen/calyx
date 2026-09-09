@echo off
rem Run the V1c execution pilot under the MSVC environment (cython codegen).
rem Comments are ASCII on purpose: the console runs .bat files under an OEM
rem code page, and UTF-8 Cyrillic in a batch file breaks the vswhere line.
setlocal
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
.\.venv\Scripts\python.exe run_v1c_grid.py --pilot --codegen cython > "results\v1c\logs\pilot.log" 2>&1
