# LiteRT Backend

The Triton Inference Server backend for [LiteRT](https://github.com/google-ai-edge/LiteRT) (formerly TensorFlow Lite) using the TFLite Interpreter API, with support for USB Coral Edge TPU acceleration.
You can learn more about Triton backends in the [backend repo](https://github.com/triton-inference-server/backend). Ask questions or report problems on the [issues page](https://github.com/triton-inference-server/server/issues).
This backend is designed to run `.tflite` serialized models using the LiteRT v2.4 runtime and is compatible with Triton Inference Server 2.75 (`r26.11`).

## Build the LiteRT Backend

### Prerequisites
- CMake version **4.0** or newer
- GCC version **13** or newer (C++20 support required)
- `rapidjson-dev`, `git`
- For USB Coral Edge TPU runtime support: `libedgetpu1-std` (or `libedgetpu1-max`)

Install build dependencies on Ubuntu 24.04:
```bash
sudo apt-get update && sudo apt-get install -y rapidjson-dev gcc-13 g++-13 git
```

To install the Coral Edge TPU runtime (`libedgetpu.so.1`) — note that Google's legacy `packages.cloud.google.com/apt` Coral repository is deprecated — use the community-maintained [`feranick/libedgetpu`](https://github.com/feranick/libedgetpu) releases (`.deb` packages for `amd64` / `arm64`):

1. Download the `libedgetpu1-std` (or `libedgetpu1-max`) `.deb` package for your architecture from [github.com/feranick/libedgetpu/releases](https://github.com/feranick/libedgetpu/releases) and install it:
```bash
curl -fsSL -O https://github.com/feranick/libedgetpu/releases/download/<TAG>/libedgetpu1-std_<VERSION>_<ARCH>.deb
sudo dpkg -i libedgetpu1-std_<VERSION>_<ARCH>.deb
sudo ldconfig
```

2. Grant your user USB access permissions (`plugdev` group) and replug the USB Coral Edge TPU device:
```bash
sudo usermod -aG plugdev $USER
```

### Build Independently with CMake
```bash
cmake -S . -B build \
  -DCMAKE_C_COMPILER=gcc-13 \
  -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install
cmake --build build -j $(nproc) --target install
```

The following required Triton repositories will be pulled and used in the build. By default the `r26.11` branch (Triton Server 2.75.0) is used for each repo, and the listed CMake arguments can be used to override:

* `triton-inference-server/backend`: `-DTRITON_BACKEND_REPO_TAG=[tag]`
* `triton-inference-server/core`: `-DTRITON_CORE_REPO_TAG=[tag]`
* `triton-inference-server/common`: `-DTRITON_COMMON_REPO_TAG=[tag]`

You can configure LiteRT and delegates using the following CMake arguments:

* LiteRT tag: `-DLITERT_TAG=[tag]` (default: `v2.4.0`)
* Coral Edge TPU external delegate: `-DLITERT_ENABLE_EDGETPU=[ON|OFF]` (default: `ON`)
* RUY library: `-DLITERT_ENABLE_RUY=[ON|OFF]` (default: `ON`)

### Build with Docker
```bash
docker build -t triton-litert-backend .
```

## Model Repository Structure
The layout for your model repository remains the same as for other standard Triton backends. Your default model filename should be set to `model.tflite`. An example model repository layout for `ssd_mobilenetv1_coco` is shown below:
```
litert-backend-model-repo
├── ssd_mobilenetv1_coco
│   ├── 1
│   │   └── model.tflite
│   └── config.pbtxt
```

## Runtime Optimizations
By default, the backend uses the standard LiteRT Interpreter without any accelerators.
To accelerate quantized models on a [Google Coral Edge TPU](https://coral.ai/docs/accelerator/get-started/), enable it via model `parameters`.

### Default Optimization Options
Optimization parameters for the default LiteRT interpreter can be passed using the `parameters` section of the model configuration.

By default the LiteRT interpreter will use the maximum number of threads available to the system.
To set the number of threads available to the LiteRT interpreter, add the following section to your model configuration:
```protobuf
parameters: {
  key: "litert_num_threads"
  value: {
    string_value: "<num_threads>"
  }
}
```

### USB Coral Edge TPU Delegate Optimization Options
To accelerate Edge TPU-compiled INT8 `.tflite` models on a USB Coral Edge TPU accelerator, configure the `edgetpu_*` model `parameters`:
```protobuf
parameters: {
  key: "edgetpu_enable"
  value: { string_value: "true" }
}
parameters: {
  key: "edgetpu_device"
  value: { string_value: "usb" }
}
parameters: {
  key: "edgetpu_lib_path"
  value: { string_value: "libedgetpu.so.1" }
}
parameters: {
  key: "edgetpu_performance"
  value: { string_value: "Max" }
}
```

Available Edge TPU parameters:
* `edgetpu_enable`: Set to `"true"` to enable the Coral Edge TPU delegate (default: `"false"`).
* `edgetpu_device`: Optional Edge TPU device specifier (default: `"usb"`). Examples: `"usb"`, `"usb:0"`, `"usb:1"`, `"pci"`, `"pci:0"`.
* `edgetpu_lib_path`: Optional path or shared library name for the Edge TPU delegate (default: `"libedgetpu.so.1"`).
* `edgetpu_performance`: Optional clock performance mode (`"Max"`, `"High"`, `"Medium"`, or `"Low"`).

When running inside Docker with a USB Coral Edge TPU, pass the USB bus device to the container:
```bash
docker run --rm -it --device /dev/bus/usb -v <full path to your model repo on host>:/models -p 8000:8000 -p 8001:8001 -p 8002:8002 tritonserver:latest
```
