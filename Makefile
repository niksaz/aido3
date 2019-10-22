AG_VERSION = 2019-04-17

# Makefile to structure work on the AI-DO baselines.

# This runs in a docker container. Dependencies are taken care of.
# ---------------- DOCKER -----------------------------------
docker_extract_data:
				docker build -t extract_container -f Dockerfile_extract_data .; \

# Extract data from docker container and copies it to the learning repository
docker_copy_for_learning:
				docker create -it --name dummy_for_copying extract_container:latest bash; \
				docker cp dummy_for_copying:/workspace/data data; \
				docker rm -fv dummy_for_copying;
