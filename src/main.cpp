#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <random>
#include <vector>
#include <cmath>

#include "graphics/Shader.h"
#include "simulation/Boid.h"

static void framebuffer_size_callback(GLFWwindow*, int w, int h) {
    glViewport(0, 0, w, h);
}

static glm::vec2 safeNormalize(const glm::vec2 &v)
{
    float s2 = glm::dot(v, v);
    if (s2 < 1e-8f)
        return {1.0f, 0.0f};
    return v / std::sqrt(s2);
}

static glm::vec2 limitVec(const glm::vec2 &v, float maxLen)
{
    float s2 = glm::dot(v, v);
    if (s2 > maxLen * maxLen)
        return v * (maxLen / std::sqrt(s2));
    return v;
}

// "Steering" force: desired velocity minus current velocity, force-capped
static glm::vec2 steerTowards(const glm::vec2 &desiredVel, const glm::vec2 &currentVel, float maxForce)
{
    return limitVec(desiredVel - currentVel, maxForce);
}

// With wrap-around, use the shortest displacement on a torus ("minimum image")
static glm::vec2 torusDelta(glm::vec2 d, float W, float H)
{
    if (d.x > W * 0.5f)
        d.x -= W;
    if (d.x < -W * 0.5f)
        d.x += W;
    if (d.y > H * 0.5f)
        d.y -= H;
    if (d.y < -H * 0.5f)
        d.y += H;
    return d;
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
    const int N = 100;
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

    // ---- Boids parameters (units: px, s) ----
    float rSep = 18.0f; // separation radius (px)
    float rAli = 45.0f; // alignment radius (px)
    float rCoh = 55.0f; // cohesion radius (px)

    float wSep = 1.6f; // separation weight
    float wAli = 1.0f; // alignment weight
    float wCoh = 0.8f; // cohesion weight

    float maxSpeed = 180.0f; // px/s
    float maxForce = 220.0f; // px/s^2 (acceleration cap; very important for stability)

    const float boidSize = 8.0f; // px

    // We'll store next velocities here (avoid read/write conflicts inside the loop)
    std::vector<glm::vec2> newVel(boids.size());

    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = float(now - last);
        last = now;

        // dt safety clamp (math/stability):
        // Avoid huge steps if you drag the window / breakpoint.
        if (dt > 0.05f) dt = 0.05f;

        glfwPollEvents();

        // ---- Update (temporary: simple drift + wrap) ----
        if (newVel.size() != boids.size())
            newVel.resize(boids.size());

        for (size_t i = 0; i < boids.size(); ++i)
        {
            const Boid &bi = boids[i];

            glm::vec2 sep(0.0f);
            glm::vec2 aliSum(0.0f);
            glm::vec2 cohSum(0.0f);

            int sepCount = 0;
            int aliCount = 0;
            int cohCount = 0;

            for (size_t j = 0; j < boids.size(); ++j)
            {
                if (j == i)
                    continue;
                const Boid &bj = boids[j];

                glm::vec2 d = bj.pos - bi.pos;
                d = torusDelta(d, (float)W, (float)H);

                float dist2 = glm::dot(d, d);
                if (dist2 < 1e-8f)
                    continue;

                float dist = std::sqrt(dist2);

                if (dist < rSep)
                {
                    // Separation: push away. We use 1/(dist^2 + eps) to avoid singularities.
                    // Direction should be away from neighbor: -(d).
                    sep += (-d) / (dist2 + 25.0f);
                    sepCount++;
                }
                if (dist < rAli)
                {
                    aliSum += bj.vel;
                    aliCount++;
                }
                if (dist < rCoh)
                {
                    // Cohesion: accumulate neighbor positions in torus-consistent coordinates.
                    cohSum += (bi.pos + d); // bj.pos "unwrapped" relative to bi
                    cohCount++;
                }
            }

            glm::vec2 acc(0.0f);

            // Separation steering
            if (sepCount > 0)
            {
                glm::vec2 desired = safeNormalize(sep) * maxSpeed;
                acc += wSep * steerTowards(desired, bi.vel, maxForce);
            }

            // Alignment steering
            if (aliCount > 0)
            {
                glm::vec2 avgVel = aliSum / (float)aliCount;
                glm::vec2 desired = safeNormalize(avgVel) * maxSpeed;
                acc += wAli * steerTowards(desired, bi.vel, maxForce);
            }

            // Cohesion steering
            if (cohCount > 0)
            {
                glm::vec2 center = cohSum / (float)cohCount;
                glm::vec2 toCenter = center - bi.pos;
                toCenter = torusDelta(toCenter, (float)W, (float)H);

                glm::vec2 desired = safeNormalize(toCenter) * maxSpeed;
                acc += wCoh * steerTowards(desired, bi.vel, maxForce);
            }

            // Final acceleration cap (prevents blow-ups)
            acc = limitVec(acc, maxForce);

            // Semi-implicit Euler:
            // v_{t+dt} = v_t + a*dt
            // x_{t+dt} = x_t + v_{t+dt}*dt
            glm::vec2 vNext = bi.vel + acc * dt;
            vNext = limitVec(vNext, maxSpeed);
            newVel[i] = vNext;
        }

        // Apply vNext and integrate positions
        for (size_t i = 0; i < boids.size(); ++i)
        {
            boids[i].vel = newVel[i];
            boids[i].pos += boids[i].vel * dt;

            // Wrap boundaries
            if (boids[i].pos.x < 0)
                boids[i].pos.x += W;
            if (boids[i].pos.x >= W)
                boids[i].pos.x -= W;
            if (boids[i].pos.y < 0)
                boids[i].pos.y += H;
            if (boids[i].pos.y >= H)
                boids[i].pos.y -= H;
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
