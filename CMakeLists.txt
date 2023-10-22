cmake_minimum_required(VERSION 3.20)

project(BLAS_at_home)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


#################
### Build library
#################

add_library(
  bah
  STATIC
  src/bah_level1.cpp
)
target_include_directories(bah PUBLIC src)
target_compile_definitions(bah PRIVATE BAH_NO_C_API)

#################
### Build test
#################

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


add_executable(
  bah_experiment
  test/test_main.cpp
)

# Link with a vendor BLAS first to compare
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
  message(STATUS "BLAS found.")

  find_path(CBLAS_INCLUDE_DIR
            cblas.h
            /opt/homebrew/opt/
            )
  if(CBLAS_INCLUDE_DIR)
    # CBLAS header file found
    message(STATUS "CBLAS header file found at: ${CBLAS_INCLUDE_DIR}")
    target_include_directories(bah_experiment PRIVATE ${CBLAS_INCLUDE_DIR})
  else()
    # CBLAS header file not found, handle the error
    message(FATAL_ERROR "CBLAS header file (cblas.h) not found.")
  endif()

  target_compile_options(bah_experiment PRIVATE ${BLAS_LINKER_FLAGS})
  target_link_libraries(bah_experiment PRIVATE ${BLAS_LIBRARIES})
endif()

# Link bah last
target_link_libraries(bah_experiment PRIVATE bah)
target_include_directories(bah_experiment PRIVATE src)


### GoogleTest
add_executable(
  bah_test
  test/bah_level1_test.cpp
)
target_link_libraries(
  bah_test
  PRIVATE
  GTest::gtest_main
  GTest::gmock_main
)
include(GoogleTest)
gtest_discover_tests(bah_test)


# link vendor BLAS
target_compile_options(bah_test PRIVATE ${BLAS_LINKER_FLAGS})
target_link_libraries(bah_test PRIVATE ${BLAS_LIBRARIES})

# link BLAS at home
target_link_libraries(bah_test PRIVATE bah)
target_include_directories(bah_test PRIVATE src)
find_path(CBLAS_INCLUDE_DIR
          cblas.h
          /opt/homebrew/opt/
          )
target_include_directories(bah_test PRIVATE ${CBLAS_INCLUDE_DIR})
