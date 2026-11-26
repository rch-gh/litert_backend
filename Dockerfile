ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS litert_backend

ENV DEBIAN_FRONTEND=noninteractive

# CMake Version options (4.0 or newer required)
ARG CMAKE_VERSION=4.0.2

RUN apt-get update && \
    apt-get install -yqq --no-install-recommends \
    git \
    wget \
    ca-certificates \
    curl \
    gpg \
    autoconf \
    libtool \
    python3-dev \
    python3-pip \
    python3-numpy \
    build-essential \
    gcc-13 \
    g++-13 \
    ccache \
    libssl-dev \
    zlib1g-dev \
    zip \
    unzip \
    xxd \
    rapidjson-dev \
    libusb-1.0-0 && \
    ARCH=$(uname -m) && \
    wget -qO /tmp/cmake.tar.gz \
      "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${ARCH}.tar.gz" && \
    tar -xzf /tmp/cmake.tar.gz -C /usr/local --strip-components=1 && \
    rm -f /tmp/cmake.tar.gz && \
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
      tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
      gpg --dearmor -o /etc/apt/trusted.gpg.d/coral-edgetpu.gpg && \
    apt-get update && \
    apt-get install -yqq --no-install-recommends libedgetpu1-std && \
    rm -rf /var/lib/apt/lists/*

ENV CC=gcc-13
ENV CXX=g++-13

# Triton Server 2.75.0 corresponds to the r26.11 release branch
ARG TRITON_REPO_TAG=r26.11

# CMake build arguments defaults
ARG CMAKE_BUILD_TYPE=Release
ARG LITERT_TAG=v2.4.0
ARG LITERT_ENABLE_EDGETPU=ON
ARG LITERT_ENABLE_RUY=ON

WORKDIR /opt/litert_backend
COPY . .
RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
    -DCMAKE_C_COMPILER=gcc-13 \
    -DCMAKE_CXX_COMPILER=g++-13 \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DTRITON_BACKEND_REPO_TAG=${TRITON_REPO_TAG} \
    -DTRITON_CORE_REPO_TAG=${TRITON_REPO_TAG} \
    -DTRITON_COMMON_REPO_TAG=${TRITON_REPO_TAG} \
    -DTRITON_ENABLE_GPU=OFF \
    -DLITERT_TAG=${LITERT_TAG} \
    -DLITERT_ENABLE_EDGETPU=${LITERT_ENABLE_EDGETPU} \
    -DLITERT_ENABLE_RUY=${LITERT_ENABLE_RUY} && \
    cmake --build build -j $(nproc) --target install
