cmake_minimum_required(VERSION 3.20)

project(BLAS_at_home)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)


#################
### Build library
#################

add_library(
  bah
  STATIC
  src/bah_level1.cpp
)
target_include_directories(bah PUBLIC src)

#################
### Build test
#################

add_executable(
  bah_test
  test/test_main.cpp
)

target_link_libraries(bah_test PRIVATE bah)
target_include_directories(bah_test PRIVATE src)

# Link with a vendor BLAS to compare
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
  message(STATUS "BLAS found.")

  find_path(CBLAS_INCLUDE_DIR cblas.h)
  if(CBLAS_INCLUDE_DIR)
    # CBLAS header file found
    message(STATUS "CBLAS header file found at: ${CBLAS_INCLUDE_DIR}")
    target_include_directories(bah_test PRIVATE ${CBLAS_INCLUDE_DIR})
  else()
    # CBLAS header file not found, handle the error
    message(FATAL_ERROR "CBLAS header file (cblas.h) not found.")
  endif()

  target_include_directories(bah_test PRIVATE ${BLAS_INCLUDE_DIRS})
  target_compile_options(bah_test PRIVATE ${BLAS_LINKER_FLAGS})
  target_link_libraries(bah_test PRIVATE ${BLAS_LIBRARIES})
endif()
