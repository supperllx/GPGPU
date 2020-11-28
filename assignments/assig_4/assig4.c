//
//
// OpenGL - elementary texture mapping
//
//
#include <stdio.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <string.h>

static float rotate = 0.0;

struct point {
	float x;
	float y;
	float z;
	};

void setup_the_view()
{
struct point eye;
struct point view;
struct point up;

glEnable(GL_DEPTH_TEST);

// Specify the size and shape of the view volume.
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
gluPerspective(45.0,1.0,0.1,20.0);
// Specify position of the view volume.
glMatrixMode(GL_MODELVIEW);
glLoadIdentity();

eye.x = 12.0; eye.y = 12.0; eye.z = 12.0;
view.x = 0.0; view.y = 0.0; view.z = 0.0;
up.x = 0.0; up.y = 1.0; up.z = 0.0;

gluLookAt(eye.x,eye.y,eye.z,view.x,view.y,view.z,up.x,up.y,up.z);
}

void draw_stuff()
{

// Clear the depth buffer (to +oo) and the background (to black).
glClearColor(0.0,0.0,0.0,0.0);
glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

GLUquadric *quad = gluNewQuadric();
gluQuadricTexture(quad, 1);
glBindTexture(GL_TEXTURE_2D,1);
glEnable(GL_TEXTURE_2D);
gluSphere(quad, 5.0, 20, 20);

glEnd();
glRotatef(rotate, 0.0, 1.0, 0.0);
glFlush();
}

void do_lights()
{
// Use one blue light.
float light0_ambient[] = { 0.0, 0.0, 1.0, 0.0 };
float light0_diffuse[] = { 1.0, 1.0, 1.0, 0.0 };
float light0_specular[] = { 1.0, 1.0, 1.0, 0.0 };
float light0_position[] = { 15.0, 15.0, 15.0, 0 };
float light0_direction[] = { 0.0, 0.0, 0.0, 1.0};

// Turn off scene default ambient.
glLightModelfv(GL_LIGHT_MODEL_AMBIENT,light0_ambient);

// Make specular correct for spots.
glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,1);

glLightfv(GL_LIGHT0,GL_AMBIENT,light0_ambient);
glLightfv(GL_LIGHT0,GL_DIFFUSE,light0_diffuse);
glLightfv(GL_LIGHT0,GL_SPECULAR,light0_specular);
glLightf(GL_LIGHT0,GL_SPOT_EXPONENT,1.0);
glLightf(GL_LIGHT0,GL_SPOT_CUTOFF,180.0);
glLightf(GL_LIGHT0,GL_CONSTANT_ATTENUATION,0.5);
glLightf(GL_LIGHT0,GL_LINEAR_ATTENUATION,0.1);
glLightf(GL_LIGHT0,GL_QUADRATIC_ATTENUATION,0.01);
glLightfv(GL_LIGHT0,GL_POSITION,light0_position);
glLightfv(GL_LIGHT0,GL_SPOT_DIRECTION,light0_direction);

glEnable(GL_LIGHTING);
glEnable(GL_LIGHT0);
}

void do_material()
{
float mat_ambient[] = {0.4, 0.4, 0.4, 1.0};
float mat_diffuse[] = {0.0, 0.0, 0.0, 1.0};
float mat_specular[] = {0.5, 0.5 ,0.5, 1.0};
float mat_shininess[] = {10.0};

glMaterialfv(GL_FRONT,GL_AMBIENT,mat_ambient);
glMaterialfv(GL_FRONT,GL_DIFFUSE,mat_diffuse);
glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular);
glMaterialfv(GL_FRONT,GL_SHININESS,mat_shininess);
}

void load_texture()
{
FILE *fopen(), *fptr;
char buf[512];
int im_size, im_width, im_height, max_color;
unsigned char *texture_bytes;
char *parse;

// Load a ppm file and hand it off to the graphics card.
fptr=fopen("scuff.ppm","r");
fgets(buf,512,fptr);
do{
	fgets(buf,512,fptr);
	} while(buf[0]=='#');
parse = strtok(buf," ");
im_width = atoi(parse);

parse = strtok(NULL,"\n");
im_height = atoi(parse);

fgets(buf,512,fptr);
parse = strtok(buf," ");
max_color = atoi(parse);

im_size = im_width*im_height;
texture_bytes = (unsigned char *)calloc(3,im_size);
fread(texture_bytes,1,3*im_size,fptr);
fclose(fptr);

glBindTexture(GL_TEXTURE_2D,1);
glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,im_width,im_height,0,GL_RGB,
	GL_UNSIGNED_BYTE,texture_bytes);
glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_MODULATE);
// We don't need this copy; the graphics card has its own now.
cfree(texture_bytes);
}

void cleanup()
{
// Release resources here.
exit(0);
}

void getout(unsigned char key, int x, int y)
{
switch(key) {
        case 'q':
                cleanup();
        default:
                break;
    }
}

void update(int value){
	rotate += 2.0f;
	if(rotate>360.f)	rotate-=360;
	glutPostRedisplay();
	glutTimerFunc(25, update, 0);
}

int main(int argc, char **argv)
{
glutInit(&argc,argv);
glutInitDisplayMode(GLUT_RGBA|GLUT_DEPTH);
glutInitWindowSize(512,512);
glutInitWindowPosition(100,50);
glutCreateWindow("my_cool_cube");
glutDisplayFunc(draw_stuff);
glutKeyboardFunc(getout);
load_texture();
setup_the_view();
do_lights();
do_material();
glutTimerFunc(25, update, 0);
glutMainLoop();
return 0;
}
