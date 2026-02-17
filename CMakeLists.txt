cmake_minimum_required(VERSION 3.20)
project(Swarm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_executable(Swarm ${SOURCES})
target_include_directories(Swarm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(Swarm PRIVATE glfw glad::glad glm::glm)
