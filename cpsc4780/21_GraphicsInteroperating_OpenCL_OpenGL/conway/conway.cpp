// Conway's Game of Life 
//
// This is an example use of OpenCL image objects, where the images are shared 
// as textures with OpenGL.
//
// rules of the game:
// 1. Every cell is labeled "alive" or "dead". Each updates itself,
//    synchronously, on each iteration.
// 2. Dead cells with exactly 3 live neighbors (including diagonals)
//    come to life.
// 3. Live cells with fewer than 2 live neighbors die of loneliness.
// 4. Live cells with more than 3 live neighbors die of overcrowding.
//
// Command line argument is the probability that the cell is initially alive.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glx.h>
#include <unistd.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_platform.h>
#include <CL/opencl.h>
#include "./RGU.h"


#define WIDTH 768
#define HEIGHT 768
#define DATA_SIZE (WIDTH*HEIGHT*4*sizeof(GLubyte)) 

// OpenCL vars
cl_context mycontext;
cl_command_queue mycq;
cl_kernel mykernel;
cl_program myprogram;
cl_mem oclimo[2];

size_t gwsize[2] = {WIDTH,HEIGHT}; 
size_t lwsize[2] = {8,8}; 

unsigned char *host_image;
unsigned char *red_image;
int tid[] = {1,2};

int do_kernel()
{
static int bindex = 1;
cl_event wlist[1];

// Flip this (current buffer index) on each call.
bindex = 1-bindex;
clSetKernelArg(mykernel,0,sizeof(cl_mem),(void *)&oclimo[bindex]);
clSetKernelArg(mykernel,1,sizeof(cl_mem),(void *)&oclimo[1-bindex]);
clEnqueueNDRangeKernel(mycq,mykernel,2,NULL,gwsize,lwsize,0,0,&wlist[0]);
clWaitForEvents(1,wlist);
return(1-bindex);
}

void mydisplayfunc()
{
int err, widx;

glFinish();
err=clEnqueueAcquireGLObjects(mycq,2,&oclimo[0],0,0,0);

widx = do_kernel();

err=clEnqueueReleaseGLObjects(mycq,2,&oclimo[0],0,0,0);
clFinish(mycq);

glClear(GL_COLOR_BUFFER_BIT);
glColor3f(1.0,0.0,0.0);
glEnable(GL_TEXTURE_2D);
glBindTexture(GL_TEXTURE_2D,tid[widx]);
glBegin(GL_QUADS);
// Just draw a square with this texture on it.
glTexCoord2f(0.0,1.0);
glVertex2f(-1.0,-1.0);
glTexCoord2f(1.0,1.0);
glVertex2f(1.0,-1.0);
glTexCoord2f(1.0,0.0);
glVertex2f(1.0,1.0);
glTexCoord2f(0.0,0.0);
glVertex2f(-1.0,1.0);
glEnd();
glDisable(GL_TEXTURE_2D);
glFlush();
glutSwapBuffers();
usleep(50000);
glutPostRedisplay();
}

void InitGL(int argc, char** argv)
{
glutInit(&argc,argv);
glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE);
glutInitWindowSize(WIDTH,HEIGHT);
glutInitWindowPosition(100,50);
glutCreateWindow("Life");
glDisable(GL_DEPTH_TEST);
glClearColor(0.1,0.2,0.35,1.0);
glewInit();
glBindTexture(GL_TEXTURE_2D,1);
glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,WIDTH,HEIGHT,0,GL_RGBA,GL_UNSIGNED_BYTE,
	host_image);
free(host_image);
glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_REPLACE);
glBindTexture(GL_TEXTURE_2D,2);
glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,WIDTH,HEIGHT,0,GL_RGBA,GL_UNSIGNED_BYTE,
	red_image);
free(red_image);
glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_REPLACE);
//glBindTexture(GL_TEXTURE_2D,0);
glFlush();
return;
}

double genrand()
{
return(((double)(random())+1.0)/((double)(RAND_MAX)+2.0));
}

void init_host_image(float frac_on)
{
int i;

host_image = (unsigned char *)calloc(1,DATA_SIZE);
red_image = (unsigned char *)calloc(1,DATA_SIZE);
srandom(123456789);

/*  Create an initial texture.  White is "alive"; cyan is "dead". */
for(i=0;i<WIDTH*HEIGHT;i++){
        if(genrand()<frac_on){
                /* white */
                host_image[4*i+0] = 255;
                host_image[4*i+1] = 255;
                host_image[4*i+2] = 255;
                host_image[4*i+3] = 0;
                }
        else{
                /* dark cyan */
                host_image[4*i+0] = 0;
                host_image[4*i+1] = 128;
                host_image[4*i+2] = 128;
                host_image[4*i+3] = 0;
                }
	/* red is test */
	red_image[4*i+0] = 255;
       	red_image[4*i+1] = 0;
       	red_image[4*i+2] = 0;
       	red_image[4*i+3] = 0;
        }
}

void cleanup()
{
clReleaseKernel(mykernel);
clReleaseProgram(myprogram);
clReleaseCommandQueue(mycq);
clReleaseMemObject(oclimo[0]);
clReleaseMemObject(oclimo[1]);
clReleaseContext(mycontext);
exit(0);
}

void getout(unsigned char key, int x, int y)
{
switch(key) {
        case 'q':               // escape quits
                cleanup();
        default:
                break;
    }
}

void initCL()
{
cl_platform_id myplatform;
cl_device_id *mydevice;
cl_int err;
char* oclsource; 
size_t program_length;
unsigned int width, height, gpudevcount;

err = RGUGetPlatformID(&myplatform);
err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,0,NULL,&gpudevcount);
mydevice = new cl_device_id[gpudevcount];
err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,gpudevcount,mydevice,NULL);

cl_context_properties props[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)myplatform,
        0};
mycontext = clCreateContext(props,1,&mydevice[0],NULL,NULL,&err);
mycq = clCreateCommandQueue(mycontext,mydevice[0],0,&err);

oclsource = RGULoadProgSource("conway.cl", "", &program_length);
myprogram = clCreateProgramWithSource(mycontext,1,(const char **)&oclsource,
	&program_length, &err);
clBuildProgram(myprogram, 0, NULL, NULL, NULL, NULL);
mykernel = clCreateKernel(myprogram, "flip", &err);
fprintf(stderr,"err is %d\n",err);

/*oclimo[0] = clCreateFromGLTexture2D(mycontext,CL_MEM_READ_WRITE,GL_TEXTURE_2D,
	0,1,&err);
oclimo[1] = clCreateFromGLTexture2D(mycontext,CL_MEM_READ_WRITE,GL_TEXTURE_2D,
	0,2,&err);*/
oclimo[0] = clCreateFromGLTexture(mycontext,CL_MEM_READ_WRITE,GL_TEXTURE_2D,
	0,1,&err);
oclimo[1] = clCreateFromGLTexture(mycontext,CL_MEM_READ_WRITE,GL_TEXTURE_2D,
	0,2,&err);
}

int main(int argc,char **argv)
{
// Load ppm image of initial population with padding inserted during load.
init_host_image(atof(argv[1]));
InitGL(argc, argv); 
initCL();
glutDisplayFunc(mydisplayfunc);
glutKeyboardFunc(getout);
glutMainLoop();
}
