# GarmentIQ MagicBox (Version 1.0)

*Last update of this page: 05/01/2025*

GarmentIQ MagicBox is a streamlined, Docker-based JupyterLab environment designed to simplify the use of the GarmentIQ Python API. By packaging the development tools and dependencies into a Docker container, MagicBox ensures a consistent and reproducible environment for all users—eliminating the need for complex local setup or manual dependency management. Whether you're analyzing garment data, prototyping models, or running production-grade scripts, MagicBox offers a plug-and-play solution that gets you up and running quickly and reliably.

## How to install GarmentIQ MagicBox?

Before installing GarmentIQ MagicBox, ensure that Docker is installed on your system. For optimal performance, it is recommended to have at least 16 GB of RAM, an NVIDIA GPU with CUDA support and a minimum of 4 GB of GPU memory, and at least 20 GB space available on disk.

### Step 1: Download GarmentIQ MagicBox configuration files

Choosing a preferred directory.

- For Windows, run the following command in Windows command prompt.

  ```cmd
  powershell -Command "Invoke-WebRequest -Uri 'https://github.com/lygitdata/GarmentIQ/archive/refs/heads/magicbox.zip' -OutFile 'magicbox.zip'; Expand-Archive -Path 'magicbox.zip' -DestinationPath .; Rename-Item 'GarmentIQ-magicbox' 'garmentiq_magicbox'; Remove-Item 'magicbox.zip'"
  ```

- For Linux / MacOS, run the following command in terminal. Make sure you have already installed `curl` and `unzip`.

  ```cmd
  curl -L -o magicbox.zip https://github.com/lygitdata/GarmentIQ/archive/refs/heads/magicbox.zip && unzip magicbox.zip && mv GarmentIQ-magicbox garmentiq_magicbox && rm magicbox.zip
  ```

### Step 2: Switch the directory

Run the following command.

```cmd
cd garmentiq_magicbox
```

### Step 3: Build the Docker image

Run the following command to build the Docker image. Make sure you have already installed Docker. This process is time consuming. Make sure your have a stable internet connection.

```cmd
docker build --no-cache -t garmentiq_magicbox .
```

### Step 4: Run the Docker container

- For Windows, run one of the following commands in Windows command prompt.

  - (Recommended) Run with GPU in addition to CPU.

    ```cmd
    docker run -d --name magicbox_container -p 8888:8888 -p 5000:5000 -p 5001:5001 -p 5002:5002 --gpus all -v "%cd%\working:/app/working" garmentiq_magicbox
    ```

  - Run with CPU only.

    ```cmd
    docker run -d --name magicbox_container -p 8888:8888 -p 5000:5000 -p 5001:5001 -p 5002:5002 -v "%cd%\working:/app/working" garmentiq_magicbox
    ```

- For Linux / MacOS, run the following command in terminal.

  - (Recommended) Run with GPU in addition to CPU.

    ```cmd
    docker run -d --name magicbox_container -p 8888:8888 -p 5000:5000 -p 5001:5001 -p 5002:5002 --gpus all -v "$(pwd)/working:/app/working" garmentiq_magicbox
    ```

  - Run with CPU only.

    ```cmd
    docker run -d --name magicbox_container -p 8888:8888 -p 5000:5000 -p 5001:5001 -p 5002:5002 -v "$(pwd)/working:/app/working" garmentiq_magicbox
    ```

### Step 5: Start GarmentIQ MagicBox

Open your browser, type http://127.0.0.1:8888 to access the Jupyter Lab interface of GarmentIQ MagicBox.

## How to stop / uninstall GarmentIQ MagicBox?

### Stop and Remove the Docker Container

- To stop the Docker container, run the following command.

  ```cmd
  docker stop magicbox_container
  ```

- To remove the Docker container, run the following command.
  
  ```cmd
  docker rm magicbox_container
  ```

### Uninstall the Docker Image

To remove the Docker image, run the following command.

```cmd
docker rmi garmentiq_magicbox
```
