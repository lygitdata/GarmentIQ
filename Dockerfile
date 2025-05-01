# Use official Python base image
FROM python:3.11.12

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install JupyterLab and ipywidgets to suppress tqdm warnings
RUN pip install --no-cache-dir jupyterlab==4.4.1 ipywidgets

# Suppress matplotlib.font_manager INFO logs using a hidden startup script
RUN mkdir -p /etc/python_startup && \
    echo "import logging; logging.basicConfig(level=logging.WARNING)" > /etc/python_startup/startup.py

# Run the startup script for all Python sessions
ENV PYTHONSTARTUP=/etc/python_startup/startup.py

# Expose JupyterLab port
EXPOSE 8888

# Set default command to start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
