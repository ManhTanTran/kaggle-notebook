$ErrorActionPreference = "Stop"
py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m ipykernel install --user --name chunkbench --display-name "Python (chunkbench)"
& .\.venv\Scripts\python.exe -m pytest
Write-Host "Select the Python (chunkbench) kernel in VS Code."

