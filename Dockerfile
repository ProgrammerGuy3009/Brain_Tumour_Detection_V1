FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for opencv)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy requirements and install python dependencies
COPY requirements.txt .
# Add matplotlib for visualization
RUN pip install --no-cache-dir -r requirements.txt matplotlib

# Copy your application files
COPY . .

# Expose the default port (HuggingFace spaces runs on port 7860)
EXPOSE 7860

# Run the application
CMD ["flask", "--app", "app-debug.py", "run", "--host=0.0.0.0", "--port=7860"]
