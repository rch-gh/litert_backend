cmake_minimum_required(VERSION 4.0)

project(tritonlitertbackend LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
  message(
    FATAL_ERROR
      "GCC version 13 or newer is required (found ${CMAKE_CXX_COMPILER_VERSION})")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Triton Options
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_ENABLE_GPU "Enable GPU support in Triton backend utilities" OFF)

set(TRITON_BACKEND_REPO_TAG
    "r26.11"
    CACHE STRING "Tag for triton-inference-server/backend repo (Triton 2.75.0)")
set(TRITON_CORE_REPO_TAG
    "r26.11"
    CACHE STRING "Tag for triton-inference-server/core repo (Triton 2.75.0)")
set(TRITON_COMMON_REPO_TAG
    "r26.11"
    CACHE STRING "Tag for triton-inference-server/common repo (Triton 2.75.0)")

# LiteRT Options
set(LITERT_TAG
    "v2.4.0"
    CACHE STRING "Version of LiteRT to build")
option(LITERT_ENABLE_EDGETPU "Build backend with USB Coral Edge TPU delegate support" ON)
option(LITERT_ENABLE_RUY "Enable RUY matrix multiplication library in LiteRT" ON)

set(TFLITE_ENABLE_XNNPACK ON CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_EXTERNAL_DELEGATE ${LITERT_ENABLE_EDGETPU} CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_RUY ${LITERT_ENABLE_RUY} CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funsafe-math-optimizations")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -funsafe-math-optimizations")

#
# Dependencies
#
# FetchContent's composability isn't very good. We must include the transitive
# closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
  EXCLUDE_FROM_ALL)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
  EXCLUDE_FROM_ALL)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
  EXCLUDE_FROM_ALL)
FetchContent_Declare(
  litert
  GIT_REPOSITORY https://github.com/google-ai-edge/LiteRT.git
  GIT_TAG ${LITERT_TAG}
  GIT_SHALLOW ON
  GIT_SUBMODULES ""
  SOURCE_SUBDIR tflite
  EXCLUDE_FROM_ALL)

if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

FetchContent_MakeAvailable(repo-common repo-core repo-backend litert)

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_litert.ldscript
               libtriton_litert.ldscript COPYONLY)

add_library(triton-litert-backend SHARED
            src/litert.cc src/litert_utils.cc src/litert_utils.h)

add_library(TritonLiteRTBackend::triton-litert-backend ALIAS
            triton-litert-backend)

target_include_directories(
  triton-litert-backend
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
          ${litert_SOURCE_DIR}
          ${litert_BINARY_DIR}/flatbuffers/include)

target_compile_features(triton-litert-backend PRIVATE cxx_std_20)
target_compile_options(
  triton-litert-backend
  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-type-limits
    -Wno-comment
    -Werror>)

set_target_properties(
  triton-litert-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_litert
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_litert.ldscript
    LINK_FLAGS
      "-Wl,--no-as-needed,--version-script libtriton_litert.ldscript")

target_link_libraries(
  triton-litert-backend
  PRIVATE triton-core-serverapi
          triton-core-backendapi
          triton-core-serverstub
          triton-backend-utils
          tensorflow-lite
          ${CMAKE_DL_LIBS})

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonLiteRTBackend)

install(
  TARGETS triton-litert-backend
  EXPORT triton-litert-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/litert
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/litert)

# Install LiteRT license
install(
  FILES ${litert_SOURCE_DIR}/LICENSE
  RENAME litert.LICENSE
  DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/litert)

install(
  EXPORT triton-litert-backend-targets
  FILE TritonLiteRTBackendTargets.cmake
  NAMESPACE TritonLiteRTBackend::
  DESTINATION ${INSTALL_CONFIGDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonLiteRTBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonLiteRTBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TritonLiteRTBackendConfig.cmake
        DESTINATION ${INSTALL_CONFIGDIR})

#
# Export from build tree
#
export(
  EXPORT triton-litert-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonLiteRTBackendTargets.cmake
  NAMESPACE TritonLiteRTBackend::)

export(PACKAGE TritonLiteRTBackend)
