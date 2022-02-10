cmake_minimum_required(VERSION 3.5.1)

project(lab_segmentation)

# Compiler flags
if (MSVC)
  set(compiler_options /W4)
  add_definitions(/D_USE_MATH_DEFINES)
else ()
  set(compiler_options -Wall -Wextra -Wpedantic)
endif (MSVC)

# Add libraries.
find_package(OpenCV 4 REQUIRED)

add_executable(${PROJECT_NAME}
  common-lab-utils.h
  common-lab-utils.cpp
  lab-segmentation.h
  lab-segmentation.cpp
  multivariate-normal-model.h
  multivariate-normal-model.cpp
  main.cpp
  )

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  )

target_compile_options(${PROJECT_NAME} PRIVATE ${compiler_options})
set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CXX_STANDARD 17
  )
