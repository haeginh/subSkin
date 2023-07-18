cmake_minimum_required(VERSION 3.16)
project(MPPD)
set(CMAKE_BUILD_TYPE Release)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
option(LIBIGL_GLFW             "Build target igl::glfw"            ON)
#option(LIBIGL_IMGUI            "Build target igl::imgui"           ON)
option(LIBIGL_COPYLEFT_TETGEN  "Build target igl_copyleft::tetgen" ON)
include(libigl)

# Enable OpenMP
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add your project files
file(GLOB SRC_FILES *.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw igl_copyleft::tetgen)


