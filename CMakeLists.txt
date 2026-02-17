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
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui")

set(IMGUI_SOURCES
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/imgui_demo.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
)

add_executable(Swarm ${SOURCES} ${IMGUI_SOURCES})
target_include_directories(Swarm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(Swarm PRIVATE
    "${IMGUI_DIR}"
    "${IMGUI_DIR}/backends"
)

target_link_libraries(Swarm PRIVATE glfw glad::glad glm::glm)
