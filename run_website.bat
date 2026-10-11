@echo off
title CrediPulse AI Web Server
echo ========================================================
echo Starting CrediPulse AI Web Server on http://localhost:5000
echo ========================================================
start "" http://localhost:5000
python web_app.py
pause
