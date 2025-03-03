# Use an official Python 3.9 image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory's content into the container's /app directory
#COPY main.py requirements.txt /app/
#COPY src/ /app/src/
COPY . /app

# Install any required packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8001 to the outside world
EXPOSE 8501

# Run the Python script
CMD ["streamlit","run", "app.py"]

# Build the Docker image
# docker build -t python-app .

# run the docker container
# docker run -d -p 8001:8001 python-app
# docker run -p 8001:8001 python-app

# run and delete after running
# docker run --rm -p 8001:8001 python-app

# docker ps -a | grep python-app
# docker stop <container_id>
# docker rm <container_id>
