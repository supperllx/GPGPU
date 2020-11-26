const sampler_t mysampler = 
      CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE|CLK_FILTER_NEAREST;

__kernel void flip(__read_only image2d_t iimage, __write_only image2d_t oimage)
{
unsigned int x = get_global_id(0);
unsigned int y = get_global_id(1);
float4 cyan = (float4)(0.0,0.5,0.8,0);
float4 white = (float4)(1.0,1.0,1.0,0);
float4 mycolor, myvalue;

mycolor = read_imagef(iimage,mysampler,(int2)(x,y));
myvalue = read_imagef(iimage,mysampler,(int2)(x,y-1));
myvalue += read_imagef(iimage,mysampler,(int2)(x,y+1));
myvalue += read_imagef(iimage,mysampler,(int2)(x-1,y));
myvalue += read_imagef(iimage,mysampler,(int2)(x+1,y));
myvalue += read_imagef(iimage,mysampler,(int2)(x-1,y-1));
myvalue += read_imagef(iimage,mysampler,(int2)(x+1,y-1));
myvalue += read_imagef(iimage,mysampler,(int2)(x-1,y+1));
myvalue += read_imagef(iimage,mysampler,(int2)(x+1,y+1));

if(myvalue.x < 2.0 || myvalue.x > 3.0) mycolor=cyan;
if(myvalue.x == 3.0) mycolor=white;

write_imagef(oimage,(int2)(x,y),mycolor);
}

