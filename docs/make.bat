@ECHO OFF

REM Sphinx build wrapper for Windows, where `make` is normally unavailable.
REM `-W` turns every warning into an error, matching CI exactly.

pushd %~dp0

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=_build
if "%SPHINXOPTS%" == "" (
	set SPHINXOPTS=-W
)

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Install the documentation
	echo.toolchain first:
	echo.
	echo.    pip install -e .
	echo.    pip install -r docs/requirements-docs.txt
	echo.
	exit /b 1
)

if "%1" == "" goto help

REM Remove the build output *and* the autosummary stubs, which are generated
REM and never version-controlled.
if "%1" == "clean" (
	if exist "%BUILDDIR%" rmdir /s /q "%BUILDDIR%"
	if exist "%SOURCEDIR%\python\reference\generated" rmdir /s /q "%SOURCEDIR%\python\reference\generated"
	goto end
)

%SPHINXBUILD% -M %1 "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%

:end
popd
