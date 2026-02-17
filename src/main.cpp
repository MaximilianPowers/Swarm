#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <numeric>

#include <imgui.h>
#include "ui/ImGuiLayer.h"
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

static int wrapIndex(int i, int n)
{
    i %= n;
    if (i < 0)
        i += n;
    return i;
}

static int cellIndexFromPos(float x, float y, float cellSize, int cols, int rows)
{
    int cx = (int)std::floor(x / cellSize);
    int cy = (int)std::floor(y / cellSize);
    cx = wrapIndex(cx, cols);
    cy = wrapIndex(cy, rows);
    return cy * cols + cx;
}


struct BoidStats
{
    float avgNeighborsAli = 0.0f; // within rAli
    float avgNeighborsSep = 0.0f; // within rSep
    float meanSpeed = 0.0f;
    float stdSpeed = 0.0f;
    float alignmentPhi = 0.0f; // order parameter in [0,1]
    float meanCellOcc = 0.0f;  // mean occupancy among non-empty cells
};

struct Controls
{
    bool paused = false;
    bool stepOnce = false;

    float rSepStep = 2.0f;
    float rAliStep = 2.0f;
    float rCohStep = 2.0f;

    float wStep = 0.1f;
    float speedStep = 10.0f;
    float forceStep = 20.0f;

    double lastPrint = 0.0;
};

static bool keyPressed(GLFWwindow *window, int key)
{
    return glfwGetKey(window, key) == GLFW_PRESS;
}

static bool keyJustPressed(GLFWwindow *window, int key, bool &prevDown)
{
    bool down = keyPressed(window, key);
    bool jp = down && !prevDown;
    prevDown = down;
    return jp;
}

static void printParams(double now,
                        float rSep, float rAli, float rCoh,
                        float wSep, float wAli, float wCoh,
                        float maxSpeed, float maxForce)
{
    // Rate-limit console spam
    // Prints at most ~4 times per second.
    static double last = 0.0;
    if (now - last < 0.25)
        return;
    last = now;

    std::cout << std::fixed << std::setprecision(2)
              << "[params] rSep=" << rSep << " rAli=" << rAli << " rCoh=" << rCoh
              << " | wSep=" << wSep << " wAli=" << wAli << " wCoh=" << wCoh
              << " | maxSpeed=" << maxSpeed << " maxForce=" << maxForce
              << "\n";
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

    // Fixed UI panel width (pixels) on the right
    const int UI_W = 420;

    // Simulation/render area is everything left of the UI panel
    const int SIM_W = W - UI_W;

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
    ImGuiLayer::Init(window, "#version 330");

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
    const int N = 1'000;
    std::vector<Boid> boids;
    boids.reserve(N);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> rx(0.0f, float(SIM_W));
    std::uniform_real_distribution<float> ry(0.0f, float(H));
    std::uniform_real_distribution<float> rv(-1.0f, 1.0f);

    for (int i = 0; i < N; ++i) {
        Boid b;
        b.pos = { rx(rng), ry(rng) };
        b.vel = glm::normalize(glm::vec2(rv(rng), rv(rng)) + glm::vec2(0.001f)) * 120.0f; // px/s
        boids.push_back(b);
    }

    // ---- Camera: orthographic pixels -> NDC ----
    glm::mat4 vp = glm::ortho(0.0f, float(SIM_W), 0.0f, float(H), -1.0f, 1.0f);

    // ---- Timing ----
    double last = glfwGetTime();

    // ---- Render settings ----
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ---- Default parameters ----
    const float DEFAULT_rSep = 18.0f;
    const float DEFAULT_rAli = 45.0f;
    const float DEFAULT_rCoh = 55.0f;

    const float DEFAULT_wSep = 1.6f;
    const float DEFAULT_wAli = 1.0f;
    const float DEFAULT_wCoh = 0.8f;

    const float DEFAULT_maxSpeed = 180.0f;
    const float DEFAULT_maxForce = 220.0f;

    // ---- Active parameters (modifiable via UI) ----
    float rSep = DEFAULT_rSep;
    float rAli = DEFAULT_rAli;
    float rCoh = DEFAULT_rCoh;

    float wSep = DEFAULT_wSep;
    float wAli = DEFAULT_wAli;
    float wCoh = DEFAULT_wCoh;

    float maxSpeed = DEFAULT_maxSpeed;
    float maxForce = DEFAULT_maxForce;

    const float boidSize = 8.0f; // px

    // We'll store next velocities here (avoid read/write conflicts inside the loop)
    std::vector<glm::vec2> newVel(boids.size());

    Controls controls;

    // Track key edge transitions (just-pressed logic)
    bool prevP = false, prevO = false, prevR = false;
    bool prev1 = false, prev2 = false, prev3 = false;
    bool prevQ = false, prevW = false, prevE = false;
    bool prevA = false, prevS = false, prevD = false;
    bool prevZ = false, prevX = false, prevC = false;
    bool prevUp = false, prevDown = false, prevLeft = false, prevRight = false;

    // ---- Uniform grid (spatial hash) ----
    // Choose cellSize >= max interaction radius so we only check neighboring 3x3 cells.
    float cellSize = std::max(rSep, std::max(rAli, rCoh));

    // Grid dimensions
    int gridCols = std::max(1, (int)std::ceil(SIM_W / cellSize));
    int gridRows = std::max(1, (int)std::ceil(H / cellSize));

    // Linked-list-in-arrays grid (no per-frame allocations):
    // head[cell] is the first boid index in that cell, next[i] is the next boid in same cell.
    std::vector<int> head(gridCols * gridRows, -1);
    std::vector<int> next((int)boids.size(), -1);

    BoidStats stats;

    // Full-history storage (since simulation start)
    std::vector<float> h_avgNeiAli;
    std::vector<float> h_meanSpeed;
    std::vector<float> h_alignment;
    std::vector<float> h_closeSep;
    std::vector<float> h_cellOcc;

    // Optional: reserve to avoid frequent reallocs
    h_avgNeiAli.reserve(10000);
    h_meanSpeed.reserve(10000);
    h_alignment.reserve(10000);
    h_closeSep.reserve(10000);
    h_cellOcc.reserve(10000);

    auto resetSimulationAndPlots = [&]()
    {
        // Reset boids
        for (auto &b : boids)
        {
            b.pos = {rx(rng), ry(rng)};
            glm::vec2 v(rv(rng), rv(rng));
            v = safeNormalize(v);
            b.vel = v * (0.7f * maxSpeed);
        }

        // Reset stats (optional, but keeps dashboard clean)
        stats = BoidStats{};

        // Reset plots (full-history vectors)
        h_avgNeiAli.clear();
        h_meanSpeed.clear();
        h_alignment.clear();
        h_closeSep.clear();
        h_cellOcc.clear();
    };

    // contiguous scratch buffers for plotting
    std::vector<float> plotBuf1, plotBuf2, plotBuf3, plotBuf4, plotBuf5;
    
    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = float(now - last);
        last = now;

        // dt safety clamp (math/stability):
        // Avoid huge steps if you drag the window / breakpoint.
        if (dt > 0.05f) dt = 0.05f;

        glfwPollEvents();
        ImGuiLayer::BeginFrame();

        // ---- ImGui UI ----
        // ---- Fixed Right Panel: Controls + Dashboard ----
        ImGuiIO &io = ImGui::GetIO();

        // Pin panel exactly to right-side reserved area
        ImGui::SetNextWindowPos(ImVec2((float)SIM_W, 0.0f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2((float)UI_W, io.DisplaySize.y), ImGuiCond_Always);

        ImGuiWindowFlags panelFlags =
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoCollapse;

        ImGui::Begin("Swarm Panel", nullptr, panelFlags);

        if (ImGui::BeginTabBar("RightPanelTabs"))
        {

            // ---- Controls tab ----
            if (ImGui::BeginTabItem("Controls"))
            {

                ImGui::Text("Boids: %d", (int)boids.size());
                ImGui::Text("FPS: %.1f", io.Framerate);

                ImGui::Separator();
                ImGui::Text("Radii (px)");
                ImGui::SliderFloat("Separation radius", &rSep, 2.0f, 120.0f);
                ImGui::SliderFloat("Alignment radius", &rAli, 2.0f, 200.0f);
                ImGui::SliderFloat("Cohesion radius", &rCoh, 2.0f, 240.0f);

                ImGui::Separator();
                ImGui::Text("Weights");
                ImGui::SliderFloat("Separation weight", &wSep, 0.0f, 5.0f);
                ImGui::SliderFloat("Alignment weight", &wAli, 0.0f, 5.0f);
                ImGui::SliderFloat("Cohesion weight", &wCoh, 0.0f, 5.0f);

                ImGui::Separator();
                ImGui::Text("Caps");
                ImGui::SliderFloat("Max speed (px/s)", &maxSpeed, 10.0f, 600.0f);
                ImGui::SliderFloat("Max force (px/s^2)", &maxForce, 10.0f, 1200.0f);

                ImGui::Separator();
                ImGui::Checkbox("Paused", &controls.paused);

                if (ImGui::Button("Step once"))
                    controls.stepOnce = true;

                ImGui::SameLine();

                if (ImGui::Button("Defaults"))
                {
                    rSep = DEFAULT_rSep;
                    rAli = DEFAULT_rAli;
                    rCoh = DEFAULT_rCoh;
                    wSep = DEFAULT_wSep;
                    wAli = DEFAULT_wAli;
                    wCoh = DEFAULT_wCoh;
                    maxSpeed = DEFAULT_maxSpeed;
                    maxForce = DEFAULT_maxForce;
                }

                ImGui::SameLine();

                if (ImGui::Button("Reset"))
                {
                    resetSimulationAndPlots();
                }

                ImGui::EndTabItem();
            }

            // ---- Dashboard tab ----
            if (ImGui::BeginTabItem("Dashboard"))
            {

                ImGui::Text("FPS: %.1f", io.Framerate);
                ImGui::Text("Boids: %d", (int)boids.size());
                ImGui::Separator();

                ImGui::Text("Avg neighbors (rAli): %.2f", stats.avgNeighborsAli);
                ImGui::Text("Avg close contacts (rSep): %.2f", stats.avgNeighborsSep);
                ImGui::Text("Mean speed: %.2f px/s", stats.meanSpeed);
                ImGui::Text("Speed std: %.2f", stats.stdSpeed);
                ImGui::Text("Alignment phi: %.3f", stats.alignmentPhi);
                ImGui::Text("Mean cell occupancy: %.2f", stats.meanCellOcc);

                ImGui::Separator();
                ImGui::Text("Trends (since reset)");

                if (!h_avgNeiAli.empty())
                    ImGui::PlotLines("Neighbors (Ali)", h_avgNeiAli.data(), (int)h_avgNeiAli.size(), 0, nullptr, FLT_MIN, FLT_MAX, ImVec2(0, 70));
                else
                    ImGui::Text("Neighbors (Ali): (no data)");

                if (!h_closeSep.empty())
                    ImGui::PlotLines("Close (Sep)", h_closeSep.data(), (int)h_closeSep.size(), 0, nullptr, FLT_MIN, FLT_MAX, ImVec2(0, 70));
                else
                    ImGui::Text("Close (Sep): (no data)");

                if (!h_meanSpeed.empty())
                    ImGui::PlotLines("Mean speed", h_meanSpeed.data(), (int)h_meanSpeed.size(), 0, nullptr, FLT_MIN, FLT_MAX, ImVec2(0, 70));
                else
                    ImGui::Text("Mean speed: (no data)");

                if (!h_alignment.empty())
                    ImGui::PlotLines("Alignment phi", h_alignment.data(), (int)h_alignment.size(), 0, nullptr, 0.0f, 1.0f, ImVec2(0, 70));
                else
                    ImGui::Text("Alignment phi: (no data)");

                if (!h_cellOcc.empty())
                    ImGui::PlotLines("Cell occupancy", h_cellOcc.data(), (int)h_cellOcc.size(), 0, nullptr, FLT_MIN, FLT_MAX, ImVec2(0, 70));
                else
                    ImGui::Text("Cell occupancy: (no data)");

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        ImGui::End();

        // ---- Controls ----
        // Toggle pause
        if (keyJustPressed(window, GLFW_KEY_P, prevP))
        {
            controls.paused = !controls.paused;
            std::cout << (controls.paused ? "[control] paused\n" : "[control] resumed\n");
        }

        // Step one frame when paused
        if (keyJustPressed(window, GLFW_KEY_O, prevO))
        {
            controls.stepOnce = true;
            std::cout << "[control] step once\n";
        }

        // Reset boids (re-randomize)
        if (keyJustPressed(window, GLFW_KEY_R, prevR))
        {
            resetSimulationAndPlots();
            std::cout << "[control] reset boids + plots\n";
        }

        // Radii (1/2/3 decrease, Q/W/E increase)
        if (keyJustPressed(window, GLFW_KEY_1, prev1))
            rSep = std::max(2.0f, rSep - controls.rSepStep);
        if (keyJustPressed(window, GLFW_KEY_Q, prevQ))
            rSep += controls.rSepStep;

        if (keyJustPressed(window, GLFW_KEY_2, prev2))
            rAli = std::max(2.0f, rAli - controls.rAliStep);
        if (keyJustPressed(window, GLFW_KEY_W, prevW))
            rAli += controls.rAliStep;

        if (keyJustPressed(window, GLFW_KEY_3, prev3))
            rCoh = std::max(2.0f, rCoh - controls.rCohStep);
        if (keyJustPressed(window, GLFW_KEY_E, prevE))
            rCoh += controls.rCohStep;

        // Weights (A/S/D decrease, Z/X/C increase)
        if (keyJustPressed(window, GLFW_KEY_A, prevA))
            wSep = std::max(0.0f, wSep - controls.wStep);
        if (keyJustPressed(window, GLFW_KEY_Z, prevZ))
            wSep += controls.wStep;

        if (keyJustPressed(window, GLFW_KEY_S, prevS))
            wAli = std::max(0.0f, wAli - controls.wStep);
        if (keyJustPressed(window, GLFW_KEY_X, prevX))
            wAli += controls.wStep;

        if (keyJustPressed(window, GLFW_KEY_D, prevD))
            wCoh = std::max(0.0f, wCoh - controls.wStep);
        if (keyJustPressed(window, GLFW_KEY_C, prevC))
            wCoh += controls.wStep;

        // Speed/force (arrow keys)
        if (keyJustPressed(window, GLFW_KEY_UP, prevUp))
            maxSpeed += controls.speedStep;
        if (keyJustPressed(window, GLFW_KEY_DOWN, prevDown))
            maxSpeed = std::max(10.0f, maxSpeed - controls.speedStep);

        if (keyJustPressed(window, GLFW_KEY_RIGHT, prevRight))
            maxForce += controls.forceStep;
        if (keyJustPressed(window, GLFW_KEY_LEFT, prevLeft))
            maxForce = std::max(10.0f, maxForce - controls.forceStep);

        // Print params if anything was changed this frame (simple heuristic):
        // We just print every frame you press a key; rate-limited in printParams.
        if (keyPressed(window, GLFW_KEY_1) || keyPressed(window, GLFW_KEY_2) || keyPressed(window, GLFW_KEY_3) ||
            keyPressed(window, GLFW_KEY_Q) || keyPressed(window, GLFW_KEY_W) || keyPressed(window, GLFW_KEY_E) ||
            keyPressed(window, GLFW_KEY_A) || keyPressed(window, GLFW_KEY_S) || keyPressed(window, GLFW_KEY_D) ||
            keyPressed(window, GLFW_KEY_Z) || keyPressed(window, GLFW_KEY_X) || keyPressed(window, GLFW_KEY_C) ||
            keyPressed(window, GLFW_KEY_UP) || keyPressed(window, GLFW_KEY_DOWN) ||
            keyPressed(window, GLFW_KEY_LEFT) || keyPressed(window, GLFW_KEY_RIGHT))
        {
            printParams(now, rSep, rAli, rCoh, wSep, wAli, wCoh, maxSpeed, maxForce);
        }

        bool doSim = !controls.paused || controls.stepOnce;

        if (doSim) {    // Keep newVel sized correctly
            controls.stepOnce = false; // reset stepOnce (if we were stepping)

            if (newVel.size() != boids.size())
                newVel.resize(boids.size());

            // If you later change radii interactively, recompute these each frame.
            // For now they’re constant, but this keeps it correct if you tweak values:
            cellSize = std::max(rSep, std::max(rAli, rCoh));
            gridCols = std::max(1, (int)std::ceil(W / cellSize));
            gridRows = std::max(1, (int)std::ceil(H / cellSize));

            // Resize arrays only if needed (rare)
            if ((int)head.size() != gridCols * gridRows)
                head.assign(gridCols * gridRows, -1);
            else
                std::fill(head.begin(), head.end(), -1);

            if ((int)next.size() != (int)boids.size())
                next.assign((int)boids.size(), -1);
            else
                std::fill(next.begin(), next.end(), -1);

            // ---- Stats accumulators (reset each sim step) ----
            double sumSpeed = 0.0;
            double sumSpeed2 = 0.0;
            double sumNeiAli = 0.0;
            double sumNeiSep = 0.0;

            // alignment order parameter: sum of normalized velocity vectors
            glm::vec2 sumDir(0.0f, 0.0f);

            // cell occupancy stats (computed after grid build)
            int nonEmptyCells = 0;
            int totalInNonEmpty = 0;

            // 1) Build grid: insert each boid into a cell
            for (int i = 0; i < (int)boids.size(); ++i)
            {
                const auto &b = boids[i];
                int cell = cellIndexFromPos(b.pos.x, b.pos.y, cellSize, gridCols, gridRows);

                // Insert i at head of linked list for this cell
                next[i] = head[cell];
                head[cell] = i;
            }

            // ---- Cell occupancy (density proxy) ----
            for (int c = 0; c < (int)head.size(); ++c)
            {
                if (head[c] == -1)
                    continue;
                nonEmptyCells++;
                int count = 0;
                for (int j = head[c]; j != -1; j = next[j])
                    count++;
                totalInNonEmpty += count;
            }
            stats.meanCellOcc = (nonEmptyCells > 0) ? (float)totalInNonEmpty / (float)nonEmptyCells : 0.0f;

            // 2) Compute new velocities using only nearby cells (3x3 neighborhood)
            for (int i = 0; i < (int)boids.size(); ++i)
            {
                const Boid &bi = boids[i];

                glm::vec2 sep(0.0f);
                glm::vec2 aliSum(0.0f);
                glm::vec2 cohSum(0.0f);

                int sepCount = 0;
                int aliCount = 0;
                int cohCount = 0;

                int cx = wrapIndex((int)std::floor(bi.pos.x / cellSize), gridCols);
                int cy = wrapIndex((int)std::floor(bi.pos.y / cellSize), gridRows);

                // Scan adjacent cells
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        int nx = wrapIndex(cx + dx, gridCols);
                        int ny = wrapIndex(cy + dy, gridRows);
                        int nCell = ny * gridCols + nx;

                        for (int j = head[nCell]; j != -1; j = next[j])
                        {
                            if (j == i)
                                continue;
                            const Boid &bj = boids[j];

                            glm::vec2 d = bj.pos - bi.pos;
                            d = torusDelta(d, (float)SIM_W, (float)H);

                            float dist2 = glm::dot(d, d);
                            if (dist2 < 1e-8f)
                                continue;
                            float dist = std::sqrt(dist2);

                            if (dist < rSep)
                            {
                                sep += (-d) / (dist2 + 25.0f); // soften singularity
                                sepCount++;
                            }
                            if (dist < rAli)
                            {
                                aliSum += bj.vel;
                                aliCount++;
                            }
                            if (dist < rCoh)
                            {
                                cohSum += (bi.pos + d); // torus-consistent neighbor position
                                cohCount++;
                            }
                        }
                    }
                }

                glm::vec2 acc(0.0f);

                if (sepCount > 0)
                {
                    glm::vec2 desired = safeNormalize(sep) * maxSpeed;
                    acc += wSep * steerTowards(desired, bi.vel, maxForce);
                }
                if (aliCount > 0)
                {
                    glm::vec2 avgVel = aliSum / (float)aliCount;
                    glm::vec2 desired = safeNormalize(avgVel) * maxSpeed;
                    acc += wAli * steerTowards(desired, bi.vel, maxForce);
                }
                if (cohCount > 0)
                {
                    glm::vec2 center = cohSum / (float)cohCount;
                    glm::vec2 toCenter = center - bi.pos;
                    toCenter = torusDelta(toCenter, (float)SIM_W, (float)H);

                    glm::vec2 desired = safeNormalize(toCenter) * maxSpeed;
                    acc += wCoh * steerTowards(desired, bi.vel, maxForce);
                }

                // Cap acceleration for stability
                acc = limitVec(acc, maxForce);

                // Semi-implicit Euler
                glm::vec2 vNext = bi.vel + acc * dt;
                vNext = limitVec(vNext, maxSpeed);
                newVel[i] = vNext;

                // ---- Stats: speed & alignment ----
                float speed = std::sqrt(glm::dot(vNext, vNext));
                sumSpeed += speed;
                sumSpeed2 += (double)speed * (double)speed;

                // order parameter uses normalized directions
                glm::vec2 dir = safeNormalize(vNext);
                sumDir += dir;

                // neighbor counts (already computed)
                sumNeiAli += aliCount;
                sumNeiSep += sepCount;
            }

            // ---- Finalize stats (per sim step) ----
            int Nf = (int)boids.size();
            if (Nf > 0)
            {
                stats.meanSpeed = (float)(sumSpeed / Nf);

                // std = sqrt(E[v^2] - (E[v])^2) (numerically stable enough here)
                double mean2 = sumSpeed2 / Nf;
                double mean = sumSpeed / Nf;
                double var = mean2 - mean * mean;
                if (var < 0.0)
                    var = 0.0;
                stats.stdSpeed = (float)std::sqrt(var);

                stats.avgNeighborsAli = (float)(sumNeiAli / Nf);
                stats.avgNeighborsSep = (float)(sumNeiSep / Nf);

                // alignment phi in [0,1]
                stats.alignmentPhi = (float)(std::sqrt(glm::dot(sumDir, sumDir)) / (float)Nf);
            }

            h_avgNeiAli.push_back(stats.avgNeighborsAli);
            h_meanSpeed.push_back(stats.meanSpeed);
            h_alignment.push_back(stats.alignmentPhi);
            h_closeSep.push_back(stats.avgNeighborsSep);
            h_cellOcc.push_back(stats.meanCellOcc);

            // 3) Apply and integrate positions
            for (int i = 0; i < (int)boids.size(); ++i)
            {
                boids[i].vel = newVel[i];
                boids[i].pos += boids[i].vel * dt;

                // Wrap boundaries
                if (boids[i].pos.x < 0)
                    boids[i].pos.x += SIM_W;
                if (boids[i].pos.x >= SIM_W)
                    boids[i].pos.x -= SIM_W;
                if (boids[i].pos.y < 0)
                    boids[i].pos.y += H;
                if (boids[i].pos.y >= H)
                    boids[i].pos.y -= H;
            }
        }

        // ---- Upload instance data ----
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER, boids.size() * sizeof(Boid), boids.data(), GL_STREAM_DRAW);

        // ---- Draw ----

        // Clear whole window first (including UI area background)
        glViewport(0, 0, W, H);
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Render boids ONLY in simulation viewport (left side)
        glViewport(0, 0, SIM_W, H);

        shader.use();
        shader.setMat4("uVP", &vp[0][0]);
        shader.setFloat("uScale", boidSize);
        shader.setVec3("uColor", 0.85f, 0.9f, 1.0f);

        glBindVertexArray(vao);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 3, (GLsizei)boids.size());
        glBindVertexArray(0);

        // Restore full viewport for ImGui
        glViewport(0, 0, W, H);

        ImGuiLayer::EndFrame();
        glfwSwapBuffers(window);
    }

    glDeleteBuffers(1, &instanceVBO);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    ImGuiLayer::Shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
