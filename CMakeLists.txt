cmake_minimum_required (VERSION 3.18)
project (libphoenix LANGUAGES CUDA C CXX)

set (CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CUDA_STANDARD 20)
set (CMAKE_CUDA_ARCHITECTURES 90)

find_package (Threads REQUIRED)
find_package (CUDA 12.4 REQUIRED)
find_package (CUDAToolkit REQUIRED)
find_library (CUFILE_LIBRARIES cufile ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBURING REQUIRED liburing)

set (NVIDIA "" CACHE PATH "Path to Nvidia driver source")
set (KERNEL "/lib/modules/${CMAKE_SYSTEM_VERSION}/build" CACHE PATH "Path to kernel source, or module directory")

set (nvidia_archs "90" CACHE STRING "NVIDIA compute architectures")
set (no_module false CACHE BOOL "Do not build kernel module")
set (no_cuda false CACHE BOOL "Do not build library with CUDA support")
set (BUILD_SHARED_LIBS true CACHE BOOL "Build shared library, set to false for static")

if (no_smartio)
    set (no_smartio_samples true)
    set (no_smartio_benchmarks true)
endif ()

# Set global compiler flags
set (GPU_ARCHS "" CACHE INTERNAL "" FORCE)
foreach (arch ${nvidia_archs})
    string (CONCAT GPU_ARCHS ${GPU_ARCHS} " -gencode arch=compute_${arch},code=sm_${arch}")
endforeach ()

set (convert_warnings implicit-function-declaration incompatible-pointer-types int-conversion return-type unused-function unused-variable)
set (flags "")
foreach (flag ${convert_warnings})
    string (APPEND flags " -Werror=${flag}")
endforeach()

# C_FLAGS
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra ${flags}" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} -g -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)
# CXX_FLAGS
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++20" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG " CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -std=c++20" CACHE INTERNAL "" FORCE)

if (no_cc)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CCUDA_NVCC_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
endif ()

include_directories ("${PROJECT_SOURCE_DIR}/libphoenix/include"
                    "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/")

# Try to find Nvidia driver
if (CUDA_FOUND AND NOT no_cuda)

    # Try to find Nvidia headers (will choose the newest version)
    file (GLOB_RECURSE driver_paths "/usr/src/nvidia-*/nv-p2p.h")
    string (REPLACE "nv-p2p.h" "" driver_paths "${driver_paths}")
    if (driver_paths)
        list (REVERSE driver_paths)
    endif ()

    find_path (driver_include NAMES "nv-p2p.h" PATHS "${NVIDIA}" ${driver_paths})
    set (driver_dir "${driver_include}")

    # Find Nvidia driver symbols
    if (NOT EXISTS "${driver_dir}/Module.symvers")
        get_filename_component (driver_dir "${driver_dir}" DIRECTORY)
    endif ()

    if ("${driver_dir}" STREQUAL "")
        message(STATUS "No NVIDIA driver symbols found")
    else ()
        message(STATUS "Using NVIDIA driver found in ${driver_dir}")
    endif ()
endif ()

# # Module source files
set (module_root "${PROJECT_SOURCE_DIR}/module")
set (module_output "${CMAKE_BINARY_DIR}/module")
# for Makefile.in, do not delete
set (module_ccflags "-I${PROJECT_SOURCE_DIR}/libnvm/include -I${driver_include}")
configure_file ("${module_root}/Makefile.in" "${module_output}/Makefile" @ONLY)
message (STATUS "Configuring module build in ${module_output}")

add_custom_target(
    modules
    ALL
    COMMAND make
    WORKING_DIRECTORY ${module_output}
    COMMENT "Building kernel module"
)

add_custom_target(
    clean_modules
    COMMAND make clean
    WORKING_DIRECTORY ${module_output}
    COMMENT "Cleaning kernel module"
)

add_custom_target(
    insmod
    COMMAND sudo make insmod
    WORKING_DIRECTORY ${module_output}
    COMMENT "Inserting kernel module"
)

add_custom_target(
    rmmod
    COMMAND sudo make rmmod
    WORKING_DIRECTORY ${module_output}
    COMMENT "Removing kernel module"
)


# Library source files
set(libphoenix_root "${PROJECT_SOURCE_DIR}/libphoenix")
file(GLOB libphoenix_source "${libphoenix_root}/*.cc")
file(GLOB libphoenix_include "${libphoenix_root}/include/*.h")

add_library(libphoenix ${libphoenix_source} ${libphoenix_include})
set_target_properties (libphoenix PROPERTIES OUTPUT_NAME "phoenix")
target_include_directories(libphoenix PRIVATE "${libphoenix_root}" CUDA::cudart)
# Set install locations
install (FILES ${libphoenix_include} DESTINATION "include")
install (TARGETS libphoenix DESTINATION "lib")


# Make target depend on pthreads
macro (set_multithread target)
    target_link_libraries (${target} Threads::Threads)
endmacro ()

# Benchmarks
set (benchmarks_root "${PROJECT_SOURCE_DIR}/benchmarks")
set (benchmarks_utils "${benchmarks_root}/utils/")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Set preferred ordering for sample targets
# Make benchmark target
# We assume all benchmarks are CUDA programs
message(STATUS "liburing include dir: ${LIBURING_INCLUDE_DIRS}")
message(STATUS "liburing library dir: ${LIBURING_LIBRARIES}")
macro (make_benchmark target binary_name files)

    add_executable (${target}) # Ugly bugly
    target_sources (${target} PRIVATE ${files})

    add_dependencies (${target} libphoenix)
    target_include_directories(${target} PRIVATE ${benchmarks_utils} ${libphoenix_root}/include ${LIBURING_INCLUDE_DIRS})
    target_link_libraries (${target} libphoenix ${CUFILE_LIBRARIES}  CUDA::cudart cuda ${LIBURING_LIBRARIES})
    set_target_properties (${target} PROPERTIES OUTPUT_NAME "${binary_name}")

    list (APPEND benchmark_targets "${target}")
    set (benchmark_targets "${benchmark_targets}" PARENT_SCOPE)

endmacro ()

add_subdirectory ("${PROJECT_SOURCE_DIR}/benchmarks/breakdown")
add_subdirectory ("${PROJECT_SOURCE_DIR}/benchmarks/end-to-end")
add_subdirectory ("${PROJECT_SOURCE_DIR}/benchmarks/kvcache")
add_subdirectory ("${PROJECT_SOURCE_DIR}/benchmarks/micro-benchmarks")
add_subdirectory ("${PROJECT_SOURCE_DIR}/benchmarks/safetensor")

add_subdirectory ("${PROJECT_SOURCE_DIR}/example/")

