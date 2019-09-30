> _Replace all italics text with content describing your project's software. Be sure to test your software installation on a clean-state system. When ready coordinate with @stav405 for delivering to the sponsor._

Table of Contents
=================

*   [Project Overview](#project-overview)
    *   [Detailed Summary](#detailed-summary)
*   [Installation Guide](#installation-guide)
    *   [Environment Requirements](#environment-requirements)
    *   [Dependencies](#dependencies)
    *   [Distribution Files](#distrubution-files)
    *   [Installation Instructions](#installation-instructions)
    *   [Test Cases](#test-cases)
*   [User Guide](#user-guide)

Project Overview
================

**Project Name:** Abstract RunTime System (ARTS)

**Principle Investigator:** Joshua Suetterlein (Joshua.Suetterlein@pnnl.gov)

**General Area or Topic of Investigation:** Asynchronous Many Task runtime (AMT)

**Release Number:** exaArtsDev

Detailed Summary
----------------

The ARTS runtime system is an AMT that explores macro-dataflow execution for data analytics.  This runtime provides users
with a distributed global adress space, a distributed memory model, and efficent synchronization constructs to write
efficent applications on a massively parallel system.

Installation Guide
==================

The following sections detail the compilation, packaging, and installation of the software. Also included are test data and scripts to verify the installation was successful.

Environment Requirements
------------------------

**Programming Language:** C, C++, and CUDA

**Operating System & Version:** Ubuntu 16.04.3, CentOS 7, 

**Required Disk Space:** 160MB

**Required Memory:** At least 1GB

**Nodes / Cores Used:** _If applicable_

Dependencies
------------

| Name | Version | Download Location | Country of Origin | Optional/Required | Special Instructions |
| ---- | ------- | ----------------- | ----------------- | ----------------- | -------------------- |
| cmake | 3.8 | https://github.com/Kitware/CMake | USA | Required | Must use 3.8 or above for CUDA language support | 
| CUDA | 9.2.148 | https://developer.nvidia.com/cuda-92-download-archive | USA | Required | Tested with CUDA 9.2. Please check OS CUDA combination |
| CUBLAS | 9.0 | https://developer.nvidia.com/cuda-90-download-archive | USA | Optional | Typically ships with CUDA or CUDA Toolkit. |
| Thrust | 9.0 | https://developer.nvidia.com/cuda-90-download-archive | USA | Optional | Typically ships with CUDA or CUDA Toolkit. |
| HWLoc | 1.11 | https://www.open-mpi.org/software/hwloc/v1.11/ | USA | Optional | New versions not yet supported | 
Distribution Files
------------------

core/ - Main directory containing the runtime source files
example/ - Directory containing both CPU and GPU examples
graph/ - Directory containing graph data structures/methods for applications.  Not part of the core runtime development.
sampleConfigs/ - Directory with arts.conf files required to run examples.
test/ - Directory containing tests to debug issues.  For development purposes.

Key Files:
arts.h - Include file required by arts programs.
arts.cfg - Arts configuration file.  This file is required in the same directory as a running ARTS program.
libarts.so - Runtime library generated after building.  Required for linking programs.


Installation Instructions
-------------------------

Before attempting to build ARTS, please take a look at the requirements in dependencies.  While cmake will attempt to find the libraries in your path, you can help cmake by providing the path of a library using a flag -D<LIB_NAME>_ROOT=<PATH_TO_LIB_DIR> (e.g. -DHWLOC_ROOT=/usr/lib64/ or -DCUDA_ROOT=/usr/lib64/cuda9.2.148).

For CPU build only:
```
git clone <url-to-ARTS-repo>  # or untar the ARTS source code.
cd arts
mkdir build && cd build
cmake ..
make -j
```

For GPU builds:
```
git clone <url-to-ARTS-repo>  # or untar the ARTS source code.
cd arts
mkdir build && cd build
cmake .. -DCUDA_ROOT=$CUDAROOT
make -j
```

Test Cases
----------

_Include test data in the distribution that can be used to verify the installation was successful. Document detailed steps of how to execute these tests and the ways to identify success or failure. Example input & output data files is preferred over manual data entry. Do not include hard-coded paths within test scripts to allow flexibility in the installation. If possible, include test cases that can be performed on systems smaller than the target system, allowing the sponsor to demonstrate an installation on a different machine._

User Guide
==========

_This section is largely up to the project to determine its contents. Include information on how to run & configure the system, common usage, special configurations, etc. This section should demonstrate to the sponsor how to generally use the software to perform the desired analysis. Consider including troubleshooting guides for known, common problems._