@echo off
rem Запуск python проекта в окружении MSVC.
rem
rem Зачем: при codegen.target='cython' Brian2 компилирует вставки через
rem setuptools, а тот на Windows нередко не находит установленную Visual Studio
rem и падает с "Unable to find a compatible Visual Studio installation".
rem vcvars настраивает окружение, DISTUTILS_USE_SDK велит distutils его взять
rem вместо собственного поиска.
rem
rem Пути не зашиты: установка ищется через vswhere, python берётся из .venv
rem рядом с этим файлом.
rem
rem Использование:  msvc_run.bat bench_backend.py cython full
setlocal
set "HERE=%~dp0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo vswhere не найден: Visual Studio, похоже, не установлена.
  exit /b 1
)
set "VSPATH="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * ^
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
    -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
  echo Тулчейн C++ не установлен: добавьте компонент VC.Tools.x86.x64.
  exit /b 1
)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set DISTUTILS_USE_SDK=1
set MSSdk=1
set "BRIAN2_CACHE_DIR=%HERE%.cython_cache"
set PYTHONIOENCODING=utf-8
"%HERE%.venv\Scripts\python.exe" %*
