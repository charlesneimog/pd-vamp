# PureData Vamp Plugin Host

## Overview
This PureData object serves as a host for Vamp plugins, enabling real-time audio analysis and feature extraction within PureData. It leverages the Real-time Vamp plugin SDK for C++20, which is designed for performance-critical applications.

## Key Features
* Real-time Processing: Hosts Vamp plugins efficiently, reducing latency and ensuring smooth real-time audio analysis.
* Performance Optimization by Real-time Vamp plugin SDK:
  * Memory Management: Eliminates memory allocations during processing, enhancing performance and stability.
  * Simplified API: Provides a streamlined and restricted plugin API for ease of use.
  * Compile-time Error Checking: Utilizes constexpr evaluation to catch errors at compile time rather than runtime.

## Requirements

Before using this PureData object, ensure you have installed the necessary Vamp plugins. You can download them from the Vamp Plugins website.

## Installation
Download and install the required Vamp plugins from the Vamp Plugins [website](https://www.vamp-plugins.org/download.html).
Place the PureData object in your PureData project.

## License
This project is licensed under the GPL3 License.
