cmake_minimum_required(VERSION 3.1.3)
project(lab_11_segmentation)

set(CMAKE_CXX_STANDARD 11)

# Add libraries.
find_package(OpenCV 4 REQUIRED)

add_executable(${PROJECT_NAME}
  multivariate_normal_model.h
  multivariate_normal_model.cpp
  main.cpp
  )

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  )

target_compile_options(${PROJECT_NAME} PRIVATE
  -Wall
  -Wextra
  -Wno-unknown-pragmas
  -Wno-unused-local-typedefs
  -Wno-unused-parameter
  -Wpedantic
  )
