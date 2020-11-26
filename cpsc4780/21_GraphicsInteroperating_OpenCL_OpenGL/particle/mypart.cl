// mypart.cl

#define FREQUENCY 4.0f

__kernel void wave(__global float4* pos, float time)
{
unsigned int x = get_global_id(0);
unsigned int y = get_global_id(1);
float u, v, h;

u = 2.0*(x/(float)(WIDTH))-1.0; 
v = 2.0*(y/(float)(DEPTH))-1.0;
h = 0.5*sin(u*FREQUENCY + time)*cos(v*(0.9f)*FREQUENCY + time);
pos[y*WIDTH+x] = (float4)(u, h, v, 1.0f);
}
