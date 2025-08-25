# Use the latest vLLM base image for best compatibility
FROM vllm/vllm-openai:latest

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./app /app/app
COPY ./scripts /app/scripts

# Expose the port the app runs on
EXPOSE 8000

# Reset the entrypoint and set the correct command to run the application
ENTRYPOINT []
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]