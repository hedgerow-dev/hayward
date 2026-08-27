# Hayward: reproducible container image for the flagship CLI.
#
# Pinned by tag so local and CI builds agree. A released image should ALSO pin
# the base by digest, so a mutable tag cannot move the base out from under a
# tagged release:
#
#   FROM python:3.12-slim@sha256:<digest>
#
# Fill the digest in at release time from:
#   docker buildx imagetools inspect python:3.12-slim
FROM python:3.12-slim

# Quiet, deterministic Python and pip: no .pyc files written into the layers,
# unbuffered stdout, no wheel cache, no version-check chatter.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Create the unprivileged runtime user first: it rarely changes, so this layer
# stays cached across source edits.
RUN useradd --create-home --uid 10001 hayward

# The build context is trimmed by .dockerignore, so only the package sources,
# pyproject.toml and the metadata files hatchling needs reach the image.
WORKDIR /src
COPY . .

# hatchling reads the version from hayward/__init__.py (the single source of
# truth). defusedxml is the one runtime dependency and installs from the index.
RUN pip install --no-cache-dir .

# Drop privileges. The scanner never writes to its own install and reads only
# the paths you mount in at run time.
USER hayward
WORKDIR /home/hayward

# `docker run <image> scan /work/model.pt` mirrors the local CLI exactly.
ENTRYPOINT ["hayward"]
CMD ["--help"]
