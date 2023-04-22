@echo off

mkdir ../spectra_venv
py -m venv ../spectra_venv
move ../EP-425-Final-Project-Code ../spectra_venv
cd EP-425-Final-Project-Code

SET /p var123=Do you want to activate the virtual environment? (Y or N): 
if /I %var123%==Y (
    @echo "VENV activated" 
    ../Scripts/activate.bat
) else  (
    @echo "Virtual Environment NOT aactivated"
    exit
)

Pause
exit