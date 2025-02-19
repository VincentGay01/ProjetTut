# CMakeList.txt : projet CMake pour projettut, incluez la source et définissez
# la logique spécifique au projet ici.
#
cmake_minimum_required (VERSION 3.8)

# Activez Rechargement à chaud pour les compilateurs MSVC si cela est pris en charge.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# TODO: Ajoutez des tests et installez des cibles si nécessaire.


cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
include(C:/vcpkg/scripts/buildsystems/vcpkg.cmake)
project(cloud_viewer)
set(Boost_DEBUG ON)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC ON)

find_package(PCL 1.14 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(CGAL REQUIRED)
add_executable (cloud_viewer cloud_viewer.cpp)

target_link_libraries (cloud_viewer ${PCL_LIBRARIES} CGAL )


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET cloud_viewer PROPERTY CXX_STANDARD 20)
endif()
