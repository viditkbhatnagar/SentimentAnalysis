# Start from a lightweight Python image that supports apt-get
FROM python:3.10-slim

# Install ffmpeg at the system level
RUN apt-get update && apt-get install -y ffmpeg

# Set a working directory
WORKDIR /app

# Copy your requirement file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire codebase into the container
COPY . .

# Expose the default Streamlit port (8501)
EXPOSE 8501

# Command to run your Streamlit app
CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0
