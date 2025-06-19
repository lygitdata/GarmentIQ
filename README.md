# GarmentIQ MagicBox (Version 1.5)

*Last update: 06/19/2025*

*Navigation*:

1. [How to install GarmentIQ MagicBox?](#how-to-install-garmentiq-magicbox)
2. [How to stop / uninstall GarmentIQ MagicBox?](#how-to-stop--uninstall-garmentiq-magicbox)
3. [How to run code in GarmentIQ MagicBox?](#how-to-run-code-in-garmentiq-magicbox)

---

![image](https://github.com/user-attachments/assets/bd40b7b6-941c-429f-a160-0dd0eae0d295)

GarmentIQ MagicBox is a streamlined, Docker-based JupyterLab environment designed to simplify the use of the GarmentIQ Python API. By packaging the development tools and dependencies into a Docker container, MagicBox ensures a consistent and reproducible environment for all users—eliminating the need for complex local setup or manual dependency management. Whether you're analyzing garment data, prototyping models, or running production-grade scripts, MagicBox offers a plug-and-play solution that gets you up and running quickly and reliably.

## How to install GarmentIQ MagicBox?

Before installing GarmentIQ MagicBox, ensure that Docker is installed and running on your system. For optimal performance, it is recommended to have at least 16 GB of RAM, an NVIDIA GPU with CUDA support and a minimum of 4 GB of GPU memory, and at least 20 GB space available on disk.

### Step 1: Download GarmentIQ MagicBox configuration files

Choosing a preferred directory.

- For Windows, run the following command in Windows command prompt.

  ```bash
  powershell -Command "Invoke-WebRequest -Uri 'https://github.com/lygitdata/GarmentIQ/archive/refs/heads/magicbox.zip' -OutFile 'magicbox.zip'; Expand-Archive -Path 'magicbox.zip' -DestinationPath .; Rename-Item 'GarmentIQ-magicbox' 'garmentiq_magicbox'; Remove-Item 'magicbox.zip'"
  ```

- For Linux / MacOS, run the following command in terminal. Make sure you have already installed `curl` and `unzip`.

  ```bash
  curl -L -o magicbox.zip https://github.com/lygitdata/GarmentIQ/archive/refs/heads/magicbox.zip && unzip magicbox.zip && mv GarmentIQ-magicbox garmentiq_magicbox && rm magicbox.zip
  ```

### Step 2: Switch the directory

Run the following command.

```bash
cd garmentiq_magicbox
```

### Step 3: Build the Docker image

Run the following command to build the Docker image. Make sure you have already installed Docker. This process is time consuming. Make sure your have a stable internet connection.

```bash
docker build --no-cache -t garmentiq_magicbox .
```

### Step 4: Run the Docker container

- For Windows, run one of the following commands in Windows command prompt.

  - (Recommended) Run with GPU in addition to CPU.

    ```bash
    docker run -d --name magicbox_container -p 8888:8888 -p 5000:5000 -p 5001:5001 -p 5002:5002 --gpus all -v "%cd%\working:/app/working" garmentiq_magicbox
    ```

  - Run with CPU only.

    ```bash
    docker run -d --name magicbox_container -p 8888:8888 -p 5000:5000 -p 5001:5001 -p 5002:5002 -v "%cd%\working:/app/working" garmentiq_magicbox
    ```

- For Linux / MacOS, run the following command in terminal.

  - (Recommended) Run with GPU in addition to CPU.

    ```bash
    docker run -d --name magicbox_container -p 8888:8888 -p 5000:5000 -p 5001:5001 -p 5002:5002 --gpus all -v "$(pwd)/working:/app/working" garmentiq_magicbox
    ```

  - Run with CPU only.

    ```bash
    docker run -d --name magicbox_container -p 8888:8888 -p 5000:5000 -p 5001:5001 -p 5002:5002 -v "$(pwd)/working:/app/working" garmentiq_magicbox
    ```

### Step 5: Start GarmentIQ MagicBox

Open your browser, type http://127.0.0.1:8888 to access the Jupyter Lab interface of GarmentIQ MagicBox.

## How to stop / uninstall GarmentIQ MagicBox?

### Stop and Remove the Docker Container

- To stop the Docker container, run the following command.

  ```bash
  docker stop magicbox_container
  ```

- To remove the Docker container, run the following command.
  
  ```bash
  docker rm magicbox_container
  ```

### Uninstall the Docker Image

To remove the Docker image, run the following command.

```bash
docker rmi garmentiq_magicbox
```

## How to Run Code in GarmentIQ MagicBox

To run code in GarmentIQ MagicBox, follow the steps below:

### 1. Prepare Your Local Directory
- Place all your configuration files for GarmentIQ MagicBox in your local directory.
- The contents inside the `working/` folder will be synchronized with this directory, so ensure it reflects the latest state.

### 2. Check Dependencies
- All necessary dependencies are already included in GarmentIQ MagicBox.
- You don't need to manually install them, making it easy to get started right away.
- In case if you need additional dipendencies, modify the `requirements.txt` in the configuration files in your local machine, then rebuild the Docker image and container, or simply using `!pip install`.

### 3. Create a New Notebook or Script
- You can create a new Jupyter notebook or a Python script to write your code.
- Ensure your notebook or script is placed inside the `working/` folder to enable synchronization.

### 4. Explore Example Notebooks
- If you're new to GarmentIQ MagicBox or want to quickly see how things work, check out the example notebooks under the `working/examples/` folder.
- These examples will help you understand how to use the system.

### 5. Run Your Code
- Once your notebook or script is ready, simply run it.
- Any outputs or changes made within the `working/` directory will be reflected in your local directory.
