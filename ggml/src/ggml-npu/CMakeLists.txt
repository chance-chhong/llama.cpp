if (GGML_STATIC)
    set(BLA_STATIC ON)
endif()

set(BLA_VENDOR OpenBLAS)
find_package(BLAS)

if (BLAS_FOUND)
    message(STATUS "BLAS found, Libraries: ${BLAS_LIBRARIES}")

    ggml_add_backend_library(ggml-npu
                             ggml-npu.cpp
                            )

    if ("${BLAS_INCLUDE_DIRS}" STREQUAL "")
        # BLAS_INCLUDE_DIRS is missing in FindBLAS.cmake.
        # see https://gitlab.kitware.com/cmake/cmake/-/issues/20268
        find_package(PkgConfig REQUIRED)
        # As of openblas v0.3.22, the 64-bit is named openblas64.pc
        pkg_check_modules(DepBLAS openblas64)
        if (NOT DepBLAS_FOUND)
            pkg_check_modules(DepBLAS openblas)
        endif()
        if (DepBLAS_FOUND)
            set(BLAS_INCLUDE_DIRS ${DepBLAS_INCLUDE_DIRS})
        else()
            message(WARNING "BLAS_INCLUDE_DIRS neither been provided nor been automatically"
            " detected by pkgconfig, trying to find cblas.h from possible paths...")
            find_path(BLAS_INCLUDE_DIRS
                NAMES cblas.h
                HINTS
                    /usr/include
                    /usr/local/include
                    /usr/include/openblas
                    /opt/homebrew/opt/openblas/include
                    /usr/local/opt/openblas/include
                    /usr/include/x86_64-linux-gnu/openblas/include
            )
        endif()
    endif()

    message(STATUS "BLAS found, Includes: ${BLAS_INCLUDE_DIRS}")

    target_compile_options(ggml-npu PRIVATE ${BLAS_LINKER_FLAGS})


    target_link_libraries     (ggml-npu PRIVATE ${BLAS_LIBRARIES})
    target_include_directories(ggml-npu PRIVATE ${BLAS_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "BLAS not found, please refer to "
                        "https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors"
                        " to set correct GGML_BLAS_VENDOR")
endif()
