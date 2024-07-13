FROM nvcr.io/nvidia/pytorch:23.06-py3

RUN apt-get update -qq && apt-get install -y libsodium-dev libbsd-dev python3-dev git bc

# Setup Users
RUN useradd --create-home --user-group stellatrain

# Copy Files
WORKDIR /home/stellatrain
COPY --chown=stellatrain:stellatrain . explore-dp

# Configure
WORKDIR /home/stellatrain/explore-dp/backend/libzmq
RUN ./autogen.sh

WORKDIR /home/stellatrain/explore-dp/backend
RUN cmake -DPYTHON_EXECUTABLE=/usr/bin/python \
      -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch \
      -DPYTHON_LIB_PATH=/usr/local/lib/python3.10 \
      -DPYTHON_INCLUDE_PATH=/usr/include/python3.10 \
      -DPYTHON_VERSION=3.10 \
      -B build

# Build
RUN cmake --build build --config RelWithDebInfo --target all -j

# Install
WORKDIR /home/stellatrain/explore-dp/backend/scikit-optimize
RUN pip install --editable .

USER stellatrain
WORKDIR /home/stellatrain/explore-dp