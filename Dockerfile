# Use tiangolo/uwsgi-nginx-flask
FROM tiangolo/uwsgi-nginx-flask:python3.10

# Make sure to not install recommends and to clean the 
# install to minimize the size of the container as much as possible.
RUN apt-get update && \
    apt-get install --no-install-recommends -y ffmpeg &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory within the container
WORKDIR /app

# Copy necessary files to the container
COPY ./app /app
# Create a virtual environment in the container
RUN python3 -m venv .venv

# Activate the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

    # Install Python dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt
    # Get the models from Hugging Face to bake into the container
RUN python3 download_models.py

# Make port 8888 available to the world outside this container
EXPOSE 8888

ENTRYPOINT [ "python3" ]

# Run main.py when the container launches
CMD [ "main.py" ]