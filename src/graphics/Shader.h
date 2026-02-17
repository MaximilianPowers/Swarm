#pragma once
#include <string>
#include <glad/glad.h>

class Shader {
public:
    Shader(const std::string& vertexPath, const std::string& fragmentPath);
    ~Shader();

    void use() const;

    void setMat4(const char* name, const float* value) const;
    void setVec3(const char* name, float x, float y, float z) const;
    void setFloat(const char* name, float v) const;

private:
    GLuint program_ = 0;

    static std::string readFile(const std::string& path);
    static GLuint compile(GLenum type, const std::string& src);
};
