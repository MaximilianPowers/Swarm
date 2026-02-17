#include "Shader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

std::string Shader::readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Failed to open shader: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

GLuint Shader::compile(GLenum type, const std::string& src) {
    GLuint id = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(id, 1, &c, nullptr);
    glCompileShader(id);

    GLint ok = 0;
    glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[2048];
        glGetShaderInfoLog(id, sizeof(buf), nullptr, buf);
        std::string msg = "Shader compile error: ";
        msg += buf;
        glDeleteShader(id);
        throw std::runtime_error(msg);
    }
    return id;
}

Shader::Shader(const std::string& vertexPath, const std::string& fragmentPath) {
    auto vsSrc = readFile(vertexPath);
    auto fsSrc = readFile(fragmentPath);

    GLuint vs = compile(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fsSrc);

    program_ = glCreateProgram();
    glAttachShader(program_, vs);
    glAttachShader(program_, fs);
    glLinkProgram(program_);

    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = 0;
    glGetProgramiv(program_, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[2048];
        glGetProgramInfoLog(program_, sizeof(buf), nullptr, buf);
        std::string msg = "Program link error: ";
        msg += buf;
        glDeleteProgram(program_);
        program_ = 0;
        throw std::runtime_error(msg);
    }
}

Shader::~Shader() {
    if (program_) glDeleteProgram(program_);
}

void Shader::use() const { glUseProgram(program_); }

void Shader::setMat4(const char* name, const float* value) const {
    GLint loc = glGetUniformLocation(program_, name);
    glUniformMatrix4fv(loc, 1, GL_FALSE, value);
}

void Shader::setVec3(const char* name, float x, float y, float z) const {
    GLint loc = glGetUniformLocation(program_, name);
    glUniform3f(loc, x, y, z);
}

void Shader::setFloat(const char* name, float v) const {
    GLint loc = glGetUniformLocation(program_, name);
    glUniform1f(loc, v);
}
