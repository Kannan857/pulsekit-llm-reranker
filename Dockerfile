# Use a vLLM base image for compatibility
FROM vllm/vllm-openai:latest

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy the rest of the application code
COPY ./app /app/app
COPY ./scripts /app/scripts

# Expose the port the app runs on
EXPOSE 8000

# Add this line to reset any default entrypoint from the base image
ENTRYPOINT []

# Command to run the application
# This will start the FastAPI server using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]