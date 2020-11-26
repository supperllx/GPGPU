#include <stdio.h>
#include <string.h>
#include <CL/cl.h>

#define DATA_SIZE 10

char* loadProgSource(const char* filename, const char* preamble, size_t *sz) {
  FILE* fptr = NULL;
  size_t szSource, szPreamble, howmany;
  char* sourceString;

  // Open the OpenCL source code file
  fptr = fopen(filename, "r");
  szPreamble = strlen(preamble);

  // Get the length of the source code
  fseek(fptr, 0, SEEK_END);
  szSource = ftell(fptr);
  fseek(fptr, 0, SEEK_SET);

  // Allocate a buffer for the source code string and read it in
  sourceString = (char *) calloc(szSource + szPreamble+1, sizeof(char));
  howmany = fread((sourceString) + szPreamble, szSource, 1, fptr);
  fclose(fptr);
  *sz = szSource + szPreamble;
  sourceString[szSource + szPreamble] = '\0';
  return sourceString;
}

int main(void) {
  cl_platform_id platform_id;
  cl_uint num_of_platforms = 0;
  cl_uint num_of_devices = 0;
  cl_device_id device_id;
  cl_context_properties properties[3];
  cl_int err;
  cl_context context;
  cl_command_queue command_queue;
  char *kernelSource;
  size_t kernelSize;
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

  // Load kernel file, prepend static info, and return total kernel size
  kernelSource = loadProgSource("vecSquare_2.cl", "", &kernelSize);

  // Create a program from the kernel source code
  program = clCreateProgramWithSource(context, 1, (const char **) 
            &kernelSource, NULL, &err);

  // Compile the program
  if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
     printf("Error building program\n");
     return 1;
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

