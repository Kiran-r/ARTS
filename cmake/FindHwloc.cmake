# Try to find the HWLOC runtime system
# Input variables:
#   HWLOC_ROOT   - The HWLOC install directory
# Output variables:
#   HWLOC_FOUND          - System has HWLOC
#   HWLOC_INC        - The HWLOC include directories
#   HWLOC_LIB            - The HWLOC libraries
#   HWLOC_VERSION_STRING - HWLOC version 

include(FindPackageHandleStandardArgs)

if (NOT DEFINED HWLOC_FOUND)

  # Set default search paths
  if (HWLOC_ROOT)
    set(HWLOC_INCLUDE_DIR ${HWLOC_ROOT}/include CACHE PATH "The include directory for HWLOC")
    set(HWLOC_LIBRARY_DIR ${HWLOC_ROOT}/lib CACHE PATH "The library directory for HWLOC")
    set(HWLOC_BINARY_DIR ${HWLOC_ROOT}/bin CACHE PATH "The bin directory for HWLOC")
  elseif(DEFINED ENV{HWLOC_INSTALL_DIR})
    set(HWLOC_INCLUDE_DIR $ENV{HWLOC_INSTALL_DIR}/include CACHE PATH "The include directory for HWLOC")
    set(HWLOC_LIBRARY_DIR $ENV{HWLOC_INSTALL_DIR}/lib CACHE PATH "The library directory for HWLOC")
    set(HWLOC_BINARY_DIR $ENV{HWLOC_INSTALL_DIR}/bin CACHE PATH "The bin directory for HWLOC")
  endif()

  find_path(HWLOC_INC 
    NAMES hwloc.h 
    HINTS ${HWLOC_INCLUDE_DIR})

  if(HWLOC_INC)
    message("Found ${HWLOC_INC}")
  else()
    message("Can't find HWLOC_INC Hint ${HWLOC_INCLUDE_DIR}")
  endif()

  # Search for the HWLOC library
  find_library(HWLOC_LIB 
    NAMES hwloc 
    HINTS ${HWLOC_LIBRARY_DIR})

  if(HWLOC_INC AND HWLOC_LIB)
    message("Found ${HWLOC_LIB}")
  else()
    message("Can't find HWLOC_LIB Hint ${HWLOC_LIBRARY_DIR}")
  endif()

  find_program(HWLOC_INFO_EXECUTABLE 
    NAMES hwloc-info
    HINTS ${HWLOC_BINARY_DIR})
  
  if(HWLOC_LIB AND HWLOC_LIB AND HWLOC_INFO_EXECUTABLE)
    execute_process(
      COMMAND ${HWLOC_INFO_EXECUTABLE} "--version" 
      OUTPUT_VARIABLE HWLOC_VERSION_LINE 
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    string(REGEX MATCH "([0-9]+.[0-9]+.[0-9]+)$" 
      HWLOC_VERSION_STRING "${HWLOC_VERSION_LINE}")
    unset(HWLOC_VERSION_LINE)

    if(${HWLOC_VERSION_STRING} VERSION_LESS "2.0.0")
        message("HWLOC version ${HWLOC_VERSION_LINE}")
    else()
        message("HWLOC version ${HWLOC_VERSION_LINE} -DHWLOC_V2 flag set")
        add_definitions(-DHWLOC_V2)
    endif()
  else()
    message("Can't find hwloc-info Hint ${HWLOC_BINARY_DIR}")
  endif()

  find_package_handle_standard_args(HWLOC
    FOUND_VAR HWLOC_FOUND
    REQUIRED_VARS HWLOC_INC HWLOC_LIB 
    HANDLE_COMPONENTS)

  mark_as_advanced(HWLOC_INC HWLOC_LIB HWLOC_INCLUDE_DIR HWLOC_LIBRARY_DIR HWLOC_BINARY_DIR)

endif()