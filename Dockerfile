# Use slim Python base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files into container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Download punkt for NLTK (used in sentence tokenization)
RUN python -m nltk.downloader punkt

# Run main.py when container starts
CMD ["python", "main.py"]
