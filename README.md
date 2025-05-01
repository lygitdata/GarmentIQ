# GarmentIQ MagicBox (Version 1.0)

*Last update of this page: 05/01/2025*

GarmentIQ MagicBox is a streamlined, Docker-based JupyterLab environment designed to simplify the use of the GarmentIQ Python API. By packaging the development tools and dependencies into a Docker container, MagicBox ensures a consistent and reproducible environment for all users—eliminating the need for complex local setup or manual dependency management. Whether you're analyzing garment data, prototyping models, or running production-grade scripts, MagicBox offers a plug-and-play solution that gets you up and running quickly and reliably.

## How to install GarmentIQ MagicBox?

Before installing GarmentIQ MagicBox, ensure that Docker is installed on your system. For optimal performance, it is recommended to have at least 16 GB of RAM, an NVIDIA GPU with CUDA support and a minimum of 4 GB of GPU memory, and at least 20 GB space available on disk.

### Step 1: Build the Docker image

Choosing a preferred directory with 

- For Windows, run the following command in Windows command prompt.

  ```cmd
  powershell -Command "Invoke-WebRequest -Uri 'https://github.com/lygitdata/GarmentIQ/archive/refs/heads/magicbox.zip' -OutFile 'magicbox.zip'; Expand-Archive -Path 'magicbox.zip' -DestinationPath .; Rename-Item 'GarmentIQ-magicbox' 'garmentiq_magicbox'; Remove-Item 'magicbox.zip'"
  ```

- For Linux / MacOS, run the following command in terminal. Make sure you have already installed `curl` and `unzip`.

  ```cmd
  curl -L -o magicbox.zip https://github.com/lygitdata/GarmentIQ/archive/refs/heads/magicbox.zip && unzip magicbox.zip && mv GarmentIQ-magicbox garmentiq_magicbox && rm magicbox.zip
  ```
