@echo off
title Agent Team - AIM_Composites_Database_Project_20260519
set "PROJ=%~dp0"
if "%PROJ:~-1%"=="\" set "PROJ=%PROJ:~0,-1%"
python "C:\Users\mathi\OneDrive\Documents\Claude\Projects\Agent-Teams\server.py" --project "%PROJ%" --port 8765 --open
echo.
echo Server stopped. Press any key to close.
pause >nul
