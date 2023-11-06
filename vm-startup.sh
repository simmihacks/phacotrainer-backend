#!/bin/bash

# Ensure the needed kernel headers for the NVIDIA drivers are installed
sudo apt install linux-headers-$(uname -r)
# Ensure that NVIDIA driver is loaded
sudo modprobe nvidia
# Ensure the NVIDIA driver is installed w/ nvidia-smi, should see driver details and GPU when run


# Pull the latest container image. See README for how to update the image
echo "Pulling latest container"
docker pull gcr.io/phacotrainer/model-process:latest
echo "Latest container pulled"

# Run the container w/ GPUs
echo "Executing container"
# -it for interative + pseudo-TTY
docker run --rm --gpus all gcr.io/phacotrainer/model-process
echo "Container execution complete"

# Shutdown the VM
echo "Shutting down"
sudo shutdown -h now