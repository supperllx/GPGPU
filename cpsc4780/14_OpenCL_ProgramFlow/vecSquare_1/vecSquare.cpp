#include <stdio.h>
#include <CL/cl.h>

#define DATA_SIZE 10

const char *kernelSource = 
"__kernel void hello(__global float *input, __global float *output)\n"\
"{\n"\
"  size_t id = get_global_id(0);\n"\
"  output[id] = input[id] * input[id];\n"\
"}\n"\
"\n";

int main(void) {
  cl_platform_id platform_id;
  cl_uint num_of_platforms = 0;
  cl_uint num_of_devices = 0;
  cl_device_id device_id;
  cl_context_properties properties[3];
  cl_int err;
  cl_context context;
  cl_command_queue command_queue;
  cl_program program;
  cl_kernel kernel;
  cl_mem input, output;
  size_t global;

  float inputData[DATA_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  float results[DATA_SIZE] = {0};

  int i;

  // Retrives a list of platforms available
  if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS) {
    printf("Unable to get platform_id\n");
    return 1;
  }

  // Get a supported GPU device
  if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, 
     &num_of_devices) != CL_SUCCESS) {
     printf("Unable to get device_id\n");
     return 1;
  }

  // Context properties list (must be terminated with 0)
  properties[0] = CL_CONTEXT_PLATFORM;
  properties[1] = (cl_context_properties) platform_id;
  properties[2] = 0;

  // Create a context with the GPU device
  context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);

  // Create a command queue using the context and device
  command_queue = clCreateCommandQueue(context, device_id, 0, &err);

  // Create a program from the kernel source code
  program = clCreateProgramWithSource(context, 1, (const char **) 
            &kernelSource, NULL, &err);

  // Compile the program
  if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
     printf("Error building program\n");

     char buffer[4096];
     size_t length;
     clGetProgramBuildInfo(
        program,              // valid program object
        device_id,            // valid device_id that executable was built
        CL_PROGRAM_BUILD_LOG, // indicate to retrieve build log
        sizeof(buffer),       // size of the buffer to write log to
        buffer,               // the actual buffer to write log to
        &length               // the actual size in bytes of data copied to buffer
     );
     printf("%s\n",buffer);
     exit(1);
  }

  // Specify which kernel from the program to execute
  kernel = clCreateKernel(program, "hello", &err);

  // Create buffers for the input and output
  input = clCreateBuffer(context, CL_MEM_READ_ONLY, 
          sizeof(float) * DATA_SIZE, NULL, NULL);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
          sizeof(float) * DATA_SIZE, NULL, NULL);

  // Load data into the input buffer
  clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0,
                       sizeof(float) * DATA_SIZE, inputData, 0, NULL, NULL);

  // Set the argument list for the kernel command
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
  global = DATA_SIZE;

  // Enqueue the kernel command for execution
  clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global,
                         NULL, 0, NULL, NULL);
  clFinish(command_queue);

  // Copy the results from out of the output buffer
  clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0,
                      sizeof(float) * DATA_SIZE, results, 0, NULL, NULL);

  // Print the results
  printf("output: ");
  for (i = 0; i < DATA_SIZE; i++) {
    printf("%f ", results[i]);
  }
  printf("\n");

  // Cleanup (release OpenCL resources)
  clReleaseContext(context);
  clReleaseCommandQueue(command_queue);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseMemObject(input);
  clReleaseMemObject(output);

  return 0;
}

