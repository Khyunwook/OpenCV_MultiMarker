
#pragma once

enum {MYCAR = 1};
void normalize(float *v);
void idle();
void rotate_view(float *view, float angle, float x, float y, float z);
void updateKeys();
void camLook();
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void init(void);

