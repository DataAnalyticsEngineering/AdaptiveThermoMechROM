# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

ENV PROJECT_DIR="/workspaces/AdaptiveThermoMechROM"

# Set working directory
WORKDIR ${PROJECT_DIR}

ENTRYPOINT ["/bin/bash"]