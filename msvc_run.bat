@echo off
rem Run the python project in an MSVC environment.
rem
rem Why: with codegen.target='cython' Brian2 compiles inserts via
rem setuptools, and on Windows that often fails to find an installed Visual Studio
rem and fails with "Unable to find a compatible Visual Studio installation".
rem vcvars sets up the environment, DISTUTILS_USE_SDK tells distutils to use it
rem instead of its own search.
rem
rem Paths are not hardcoded: the installation is located via vswhere, python is taken
rem from .venv next to this file.
rem
rem Usage:  msvc_run.bat bench_backend.py cython full
setlocal
set "HERE=%~dp0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo vswhere not found: Visual Studio does not appear to be installed.
  exit /b 1
)
set "VSPATH="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * ^
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
    -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
  echo C++ toolchain not installed: add the VC.Tools.x86.x64 component.
  exit /b 1
)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set DISTUTILS_USE_SDK=1
set MSSdk=1
set "BRIAN2_CACHE_DIR=%HERE%.cython_cache"
set PYTHONIOENCODING=utf-8
"%HERE%.venv\Scripts\python.exe" %*
