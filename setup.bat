@echo off

py -m pip install git+https://github.com/[kconfeiteiro]/[EP-425-Final-Project-Code]@[main]
@echo "Repo has been pip installed."
@cls

@echo  "Pip instlling requirements"
@echo:

py -m pip install -r requirements.txt

@echo:
Pause
exit