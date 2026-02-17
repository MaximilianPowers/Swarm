#version 330 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 iPos;
layout (location = 2) in vec2 iVel;

uniform mat4 uVP;
uniform float uScale;

void main() {
    vec2 v = iVel;
    float sp2 = dot(v, v);
    if (sp2 < 1e-8) v = vec2(1.0, 0.0); // avoid atan(0,0)
    else v = normalize(v);

    float ang = atan(v.y, v.x);
    float c = cos(ang);
    float s = sin(ang);

    // Column-major: columns are (c, s) and (-s, c)
    mat2 R = mat2(c,  s,
                 -s,  c);

    vec2 world = iPos + R * (aPos * uScale);
    gl_Position = uVP * vec4(world, 0.0, 1.0);
}
