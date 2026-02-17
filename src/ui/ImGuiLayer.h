
#pragma once
#include <GLFW/glfw3.h>

class ImGuiLayer
{
public:
    // glslVersion should match your OpenGL context, e.g. "#version 330"
    static void Init(GLFWwindow *window, const char *glslVersion);
    static void Shutdown();

    static void BeginFrame();
    static void EndFrame();
};
