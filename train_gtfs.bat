@echo off
REM Wrapper script for running CoST-GFormer training on a GTFS dataset.
REM Usage: train_gtfs.bat STATIC_FEED [REALTIME_FEED] [options]

SET PYTHON=python
where python3 >nul 2>&1
IF %ERRORLEVEL%==0 SET PYTHON=python3

%PYTHON% -m cost_gformer.train_gtfs %*
