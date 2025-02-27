cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_STANDARD 20)

cmake_policy(SET CMP0135 NEW)
set(PDCMAKE_FILE ${CMAKE_BINARY_DIR}/pd.cmake)
if(NOT EXISTS ${PDCMAKE_FILE})
    message(STATUS "Downloading pd.cmake")
    file(
        DOWNLOAD https://raw.githubusercontent.com/pure-data/pd.cmake/refs/tags/v0.1.0/pd.cmake ${PDCMAKE_FILE}
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS)
endif()
include(${PDCMAKE_FILE})

# Declare the name of the project:
project(vamp~)

# ╭──────────────────────────────────────╮
# │                FFTW3                 │
# ╰──────────────────────────────────────╯
cmake_policy(SET CMP0135 NEW)

option(BUILD_SHARED_LIBS OFF)
option(BUILD_TESTS OFF)
set(FFTW_SINGLE ON)

set(FFTW3_FILE ${CMAKE_BINARY_DIR}/fftw-3.3.10.tar.gz)
if(NOT EXISTS ${FFTW3_FILE})
  message(STATUS "Downloading FFTW3")
  file(DOWNLOAD https://www.fftw.org/fftw-3.3.10.tar.gz ${FFTW3_FILE})
endif()

file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/fftw-3.3.10.tar.gz DESTINATION
     ${CMAKE_BINARY_DIR}/Libraries/)

add_subdirectory(${CMAKE_BINARY_DIR}/Libraries/fftw-3.3.10 EXCLUDE_FROM_ALL)
set_target_properties(fftw3 PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                       ${CMAKE_SOURCE_DIR})
set_target_properties(fftw3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
include_directories(${CMAKE_BINARY_DIR}/Libraries/fftw-3.3.10/api/)

# ╭──────────────────────────────────────╮
# │             VAMP RT HOST             │
# ╰──────────────────────────────────────╯
include_directories(
  vamp_tilde PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/Libraries/rt-vamp-plugin-sdk/hostsdk/include/")

include_directories(
  vamp_tilde PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/Libraries/rt-vamp-plugin-sdk/3rdparty/")

set(rt-vamp-host-src
    "${CMAKE_CURRENT_SOURCE_DIR}/Libraries/rt-vamp-plugin-sdk/hostsdk/src/")

add_library(
  rt-vamp-plugin-sdk
  $<$<PLATFORM_ID:Windows>:${rt-vamp-host-src}/DynamicLibrary_Windows.cpp>
  $<$<NOT:$<PLATFORM_ID:Windows>>:${rt-vamp-host-src}/DynamicLibrary_Unix.cpp>
  ${rt-vamp-host-src}/hostsdk.cpp
  ${rt-vamp-host-src}/PluginHostAdapter.cpp
  ${rt-vamp-host-src}/PluginKey.cpp
  ${rt-vamp-host-src}/PluginLibrary.cpp)

set_target_properties(rt-vamp-plugin-sdk PROPERTIES POSITION_INDEPENDENT_CODE
                                                    ON)

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
endif()

# ╭──────────────────────────────────────╮
# │              Pd Object               │
# ╰──────────────────────────────────────╯
pd_add_external(vamp~ Sources/pd-vamp.cpp)
pd_add_datafile(vamp~ Resources/vamp~-help.pd)
target_link_libraries(vamp_tilde PRIVATE rt-vamp-plugin-sdk fftw3)
