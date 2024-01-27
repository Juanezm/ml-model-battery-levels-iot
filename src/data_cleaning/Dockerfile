# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add dependencies
RUN apt-get update && \
    apt-get install -y libcairo2-dev wkhtmltopdf gcc && \
    rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Additional dependencies
RUN pip install --no-cache-dir pycairo reportlab imgkit

# Copy the directories into the container
COPY data_cleaning /app/data_cleaning
COPY utils /app/utils

# Set PYTHONPATH to include the current working directory
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Run your program when the container launches
CMD ["python", "data_cleaning/all_data_cleaning.py"]
