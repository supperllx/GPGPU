// 
// lightedCube.c 
//
// rendering a single, lighted cube.
//
// To compile: 
// gcc lightedCube.c -lGL -lGLU -lglut -o lightedCube
//
#include <stdio.h>
#include <stdlib.h>
#include <GL/gl.h>
#include <GL/glut.h>

// cube geometry, the hard way (with lots of repetition)
float front[][3]={{0.0,0.0,1.0},{1.0,0.0,1.0},{1.0,1.0,1.0},{0.0,1.0,1.0}};
float back[][3]={{0.0,0.0,0.0},{0.0,1.0,0.0},{1.0,1.0,0.0},{1.0,0.0,0.0}};
float left[][3]={{0.0,0.0,0.0},{0.0,0.0,1.0},{0.0,1.0,1.0},{0.0,1.0,0.0}};
float right[][3]={{1.0,0.0,0.0},{1.0,1.0,0.0},{1.0,1.0,1.0},{1.0,0.0,1.0}};
float top[][3]={{0.0,1.0,0.0},{0.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,0.0}};
float bottom[][3]={{0.0,0.0,0.0},{0.0,0.0,1.0},{1.0,0.0,1.0},{1.0,0.0,0.0}};

// eye position, directions
float eye[] = {2.0,2.0,2.0};
float view[] = {0.0,0.0,0.0};
float up[] = {0.0,1.0,0.0};

void setup_the_viewvolume() {
  // Specify size and shape of view volume.
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0,1.0,0.1,20.0);

  // Specify position for view volume.
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(eye[0],eye[1],eye[2],view[0],view[1],view[2],up[0],up[1],up[2]);
}

void draw_stuff() {
  int i;
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

  // Draw the cube.
  glBegin(GL_QUADS); 
    glNormal3f(0.0,0.0,1.0);
    for(i=0;i<4;i++) glVertex3fv(front[i]);
    glNormal3f(0.0,0.0,-1.0);
    for(i=0;i<4;i++) glVertex3fv(back[i]);
    glNormal3f(-1.0,0.0,0.0);
    for(i=0;i<4;i++) glVertex3fv(left[i]);
    glNormal3f(1.0,0.0,0.0);
    for(i=0;i<4;i++) glVertex3fv(right[i]);
    glNormal3f(0.0,1.0,0.0);
    for(i=0;i<4;i++) glVertex3fv(top[i]);
    glNormal3f(0.0,-1.0,0.0);
    for(i=0;i<4;i++) glVertex3fv(bottom[i]);
  glEnd();

  glFlush();
}

void do_lights() {
  // white light
  float light_ambient[] = { 0.0, 0.0, 0.0, 0.0 };
  float light_diffuse[] = { 2.0, 2.0, 2.0, 0.0 };
  float light_specular[] = { 2.25, 2.25, 2.25, 0.0 };
  float light_position[] = { 1.5, 2.0, 2.0, 1.0 };
  float light_direction[] = { -1.5, -2.0, -2.0, 1.0};

  // This allows an ambient light to be applied to the entire scene.
  // We don't want any, and so we set it to RGBA = (0,0,0,0).
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT,light_ambient);

  // Make specular correct for spots. Otherwise, specular values are
  // calculated as if the view direction were the negative z axis.
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,1);

  glLightfv(GL_LIGHT0,GL_AMBIENT,light_ambient);
  glLightfv(GL_LIGHT0,GL_DIFFUSE,light_diffuse);
  glLightfv(GL_LIGHT0,GL_SPECULAR,light_specular);
  glLightf(GL_LIGHT0,GL_SPOT_EXPONENT,5.0);
  glLightf(GL_LIGHT0,GL_SPOT_CUTOFF,180.0);
  glLightf(GL_LIGHT0,GL_CONSTANT_ATTENUATION,1.0);
  glLightf(GL_LIGHT0,GL_LINEAR_ATTENUATION,0.2);
  glLightf(GL_LIGHT0,GL_QUADRATIC_ATTENUATION,0.01);
  glLightfv(GL_LIGHT0,GL_POSITION,light_position);
  glLightfv(GL_LIGHT0,GL_SPOT_DIRECTION,light_direction);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
}

void do_material() {
  float mat_ambient[] = {0.0,0.0,0.0,1.0};
  float mat_diffuse[] = {0.9,0.9,0.1,1.0};
  float mat_specular[] = {1.0,1.0,1.0,1.0};
  //float mat_red[] = {1.0,0.0,0.0,0.0};
  float mat_shininess[] = {2.0};

  glMaterialfv(GL_FRONT,GL_AMBIENT,mat_ambient);
  glMaterialfv(GL_FRONT,GL_DIFFUSE,mat_diffuse);
  glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular);
  glMaterialfv(GL_FRONT,GL_SHININESS,mat_shininess);
  //glMaterialfv(GL_FRONT,GL_EMISSION,mat_red);
}

void cleanup() {
  // Release resources here.
  exit(0);
}

void getout(unsigned char key, int x, int y) {
  switch(key) {
    case 'q':               
      cleanup();
    default:
      break;
  }
}


int main(int argc, char **argv) {
  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_RGBA|GLUT_DEPTH|GLUT_MULTISAMPLE);
  glutInitWindowSize(512,512);
  glutInitWindowPosition(100,50);
  glutCreateWindow("lighted_cube");
  setup_the_viewvolume();
  // Comment these out and put in solid yellow to see
  // lighting effects.
  //glColor3f(1.0,1.0,0.0);
  do_lights();
  do_material();
  glClearColor(0.35,0.35,0.35,0.0);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MULTISAMPLE_ARB);
  glutDisplayFunc(draw_stuff);
  glutKeyboardFunc(getout);
  glutMainLoop();
  return 0;
}
