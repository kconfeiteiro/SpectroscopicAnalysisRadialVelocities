@echo off

cd ..
mkdir spectra_venv
py -m venv spectra_venv
@echo "spectra_venv created and initialized"

move EP-425-Final-Project-Code spectra_venv
@echo "Moved repo to virtual environemnt"

cd spectra_venv\EP-425-Final-Project-Code

SET /p var123=Do you want to activate the virtual environment? (Y or N): 
if /I %var123%==Y (
    @echo "VENV activated" 
    ..\Scripts\activate
) else  (
    @echo "Virtual Environment NOT aactivated"
    exit
)

@echo: 
Pause
exit