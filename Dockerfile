# Use a RAPIDS base image with CUDA and cuML pre-installed.
# Choose a tag that matches your desired CUDA version and Python version.
# Example: 23.08 is a recent stable release, cuda11.8, python3.10
FROM rapidsai/base:25.04-cuda12.8-py3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install *additional* Python dependencies
# (cuML, cuDF, etc. are already in the base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # Remove cuML/cuDF from requirements.txt for pip install as they are in base image
    # and pip installing them might cause conflicts.
    # Instead, ensure the base image provides them.
    # We will only install other non-RAPIDS dependencies here.
    && pip uninstall -y cuml-cuda11 cudf-cuda11 || true # Remove if they were accidentally listed and cause conflicts

# Copy the trainer directory into the container
COPY trainer /app/trainer

# Set the entry point for the training script
ENTRYPOINT ["python", "-m", "trainer.task_gpu"]
