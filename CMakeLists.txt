# Define your standard CMake header (for example):
cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)

set(PDCMAKE_DIR
    Resources/pd.cmake
    CACHE PATH "Path to pd.cmake")
include(${PDCMAKE_DIR}/pd.cmake)

# Declare the name of the project:
project(vamp~)

# ╭──────────────────────────────────────╮
# │             VAMP RT HOST             │
# ╰──────────────────────────────────────╯
include_directories(
  vamp_tilde PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/Libraries/rt-vamp-plugin-sdk/hostsdk/include/")

set(rt-vamp-host-src
    "${CMAKE_CURRENT_SOURCE_DIR}/Libraries/rt-vamp-plugin-sdk/hostsdk/src/")

add_library(
  rt-vamp-plugin-sdk
  $<IF:$<PLATFORM_ID:Windows>,
  "${rt-vamp-host-src}/DynamicLibrary_Windows.cpp",
  "${rt-vamp-host-src}/DynamicLibrary_Unix.cpp">
  "${rt-vamp-host-src}/hostsdk.cpp"
  "${rt-vamp-host-src}/PluginHostAdapter.cpp"
  "${rt-vamp-host-src}/PluginKey.cpp"
  "${rt-vamp-host-src}/PluginLibrary.cpp")

set_target_properties(rt-vamp-plugin-sdk PROPERTIES POSITION_INDEPENDENT_CODE
                                                    ON)

# ╭──────────────────────────────────────╮
# │              Pd Object               │
# ╰──────────────────────────────────────╯
pd_add_external(vamp~ Sources/pd-vamp.cpp TARGET vamp_tilde)
target_link_libraries(vamp_tilde PRIVATE rt-vamp-plugin-sdk fftw3f)
