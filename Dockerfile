# Use the official Python image
FROM python:3.12-slim

WORKDIR /app

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the app
COPY . .

# Expose Streamlit port
EXPOSE 8501


# Run Streamlit app
CMD ["streamlit", "run", "app.py"]
