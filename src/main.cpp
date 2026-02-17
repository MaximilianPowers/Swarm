#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <random>
#include <vector>

#include "graphics/Shader.h"
#include "simulation/Boid.h"

static void framebuffer_size_callback(GLFWwindow*, int w, int h) {
    glViewport(0, 0, w, h);
}

static float clampLen(glm::vec2& v, float maxLen) {
    float len2 = glm::dot(v, v);
    if (len2 > maxLen * maxLen) {
        float inv = maxLen / std::sqrt(len2);
        v *= inv;
        return maxLen;
    }
    return std::sqrt(len2);
}

int main() {
    // ---- GLFW / OpenGL init ----
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    const int W = 1280;
    const int H = 720;

    GLFWwindow* window = glfwCreateWindow(W, H, "Swarm", nullptr, nullptr);
    if (!window) {
        std::cerr << "Window create failed\n";
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSwapInterval(1); // vsync

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD init failed\n";
        return 1;
    }

    std::cout << "OpenGL: " << glGetString(GL_VERSION) << "\n";

    // ---- Shaders ----
    Shader shader("assets/shaders/boid.vert", "assets/shaders/boid.frag");

    // ---- Geometry: a small triangle facing +X in local space ----
    // Local coords chosen so the "nose" points right.
    const glm::vec2 tri[3] = {
        {  1.0f,  0.0f},
        { -0.6f,  0.4f},
        { -0.6f, -0.4f}
    };

    GLuint vao = 0, vbo = 0, instanceVBO = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Vertex buffer (triangle)
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)0);

    // Instance buffer: vec2 pos + vec2 vel per boid
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STREAM_DRAW);

    // iPos at location 1
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Boid), (void*)offsetof(Boid, pos));
    glVertexAttribDivisor(1, 1);

    // iVel at location 2
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Boid), (void*)offsetof(Boid, vel));
    glVertexAttribDivisor(2, 1);

    glBindVertexArray(0);

    // ---- Boids initial state ----
    const int N = 1000;
    std::vector<Boid> boids;
    boids.reserve(N);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> rx(0.0f, float(W));
    std::uniform_real_distribution<float> ry(0.0f, float(H));
    std::uniform_real_distribution<float> rv(-1.0f, 1.0f);

    for (int i = 0; i < N; ++i) {
        Boid b;
        b.pos = { rx(rng), ry(rng) };
        b.vel = glm::normalize(glm::vec2(rv(rng), rv(rng)) + glm::vec2(0.001f)) * 120.0f; // px/s
        boids.push_back(b);
    }

    // ---- Camera: orthographic pixels -> NDC ----
    glm::mat4 vp = glm::ortho(0.0f, float(W), 0.0f, float(H), -1.0f, 1.0f);

    // ---- Timing ----
    double last = glfwGetTime();

    // ---- Render settings ----
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const float maxSpeed = 160.0f; // px/s (used even in drift)
    const float boidSize = 8.0f;   // px

    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = float(now - last);
        last = now;

        // dt safety clamp (math/stability):
        // Avoid huge steps if you drag the window / breakpoint.
        if (dt > 0.05f) dt = 0.05f;

        glfwPollEvents();

        // ---- Update (temporary: simple drift + wrap) ----
        for (auto& b : boids) {
            // Integrate: x(t+dt) = x + v*dt
            b.pos += b.vel * dt;

            // Wrap boundaries
            if (b.pos.x < 0) b.pos.x += W;
            if (b.pos.x >= W) b.pos.x -= W;
            if (b.pos.y < 0) b.pos.y += H;
            if (b.pos.y >= H) b.pos.y -= H;

            // Keep speed bounded (important later too)
            glm::vec2 v = b.vel;
            clampLen(v, maxSpeed);
            b.vel = v;
        }

        // ---- Upload instance data ----
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER, boids.size() * sizeof(Boid), boids.data(), GL_STREAM_DRAW);

        // ---- Draw ----
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        shader.use();
        shader.setMat4("uVP", &vp[0][0]);
        shader.setFloat("uScale", boidSize);
        shader.setVec3("uColor", 0.85f, 0.9f, 1.0f);

        glBindVertexArray(vao);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 3, (GLsizei)boids.size());
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    glDeleteBuffers(1, &instanceVBO);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
