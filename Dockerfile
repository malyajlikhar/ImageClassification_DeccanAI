# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports that your FastAPI and Streamlit apps run on
EXPOSE 8000
EXPOSE 8001

# Start FastAPI on port 8000 and Streamlit on port 8001
CMD ["sh", "-c", "nohup uvicorn app:app --host 0.0.0.0 --port 8001 & streamlit run frontend.py --server.port 8000 --server.address 0.0.0.0"]