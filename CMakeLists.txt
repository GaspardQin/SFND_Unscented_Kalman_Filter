cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++11  -O3)
set(CMAKE_CXX_COMPILER g++)
set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS} ")

project(playback)

find_package(PCL 1.2 REQUIRED)



include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

add_executable (ukf_highway src/main.cpp src/ukf.cpp src/tools.cpp src/render/render.cpp)

# if enable visualization of NIS
#find_package(PythonLibs 2.7)
#target_link_libraries (ukf_highway ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
#target_include_directories(ukf_highway PRIVATE ${PYTHON_INCLUDE_DIRS})

# if disable visualization of NIS
target_link_libraries (ukf_highway ${PCL_LIBRARIES})




