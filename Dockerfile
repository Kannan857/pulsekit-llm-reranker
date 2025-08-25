# Use a vLLM base image for compatibility
FROM vllm/vllm-openai:v0.4.2

# Define a build-time argument for the Hugging Face token
ARG HF_TOKEN

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Use the token to log in to Hugging Face CLI
# This authenticates the image for model downloads
RUN huggingface-cli login --token $HF_TOKEN

# Copy the rest of the application code
COPY ./app /app/app
COPY ./scripts /app/scripts

# Expose the port the app runs on
EXPOSE 8000

# Reset the entrypoint and set the correct command to run the application
ENTRYPOINT []
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]