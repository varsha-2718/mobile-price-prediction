# Use official Python 3.11 slim image (since you used 3.11 earlier)
FROM python:3.11-slim

# Set environment variables for python behavior and pip cache disabling
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for building packages (if any)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create required directories matching your project structure
RUN mkdir -p logs artifacts data/output

# Copy all project files into the container
COPY . .

# Set PYTHONPATH to /app so imports in your scripts work fine
ENV PYTHONPATH=/app

# Expose port for Streamlit
EXPOSE 8501

# Set Streamlit environment variables so it binds correctly inside Docker
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run your Streamlit app in the correct folder path
CMD ["streamlit", "run", "Scripts/app.py"]
