@ECHO OFF
:loop

python %cd%\OverUnderML_macau.py

timeout 300

goto loop