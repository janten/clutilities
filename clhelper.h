//
//  clhelper.h
//  ncl
//
//  Created by Jan-Gerd Tenberge on 16.05.12.
//  Copyright (c) 2012 Jan-Gerd Tenberge. All rights reserved.
//
//  Include this header to easily create a global OpenCL context and run
//  your kernels there. Example usage:
//
//      initOpenCL();
//      cl_program program = buildCLProgram("source.cl");
//      cl_kernel  kernel  = buildCLKernel(program, "myKernel");
//      cl_int     status  = enqueueCLKernel(kernel, 1024);
//      showCLError(status);
//      waitForCLOperations();
//      releaseCLHelper();

#ifndef ncl_clhelper_h
#define ncl_clhelper_h

#include <assert.h>
#include <iostream>

#if defined __APPLE__ || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

const char* clErrorString(cl_int error);
void showCLError (cl_int error);

// Use one global queue
cl_command_queue queue;
cl_context context;
cl_device_id device = 0;

// Load file contents, internal use only
const char* loadFile(const char *fname) {
    FILE * f = fopen (fname, "r");
    if (!f) { printf("loadFile: couldn't open %s\n", fname); exit(1); }
    fseek(f, 0L, SEEK_END); 
    size_t sz = ftell(f); 
    fseek(f, 0L, SEEK_SET);
    char * buf = (char*)malloc(sz+1);
    fread(buf, sz, 1, f);
    buf[sz] = 0;
    fclose(f);
    return (const char *) buf;
}

// Perform a blocking write to the given memory buffer
cl_int writeDataToCLBuffer (void *data, size_t bytes, cl_mem buffer) {
    cl_int error = CL_SUCCESS;
    printf("Performing blocking write of %lu bytes of data from %p\n", bytes, data);
    error = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, bytes, data, 0, NULL, NULL);
    showCLError(error);
    return error;
}

// Read data from given buffer. Will block program execution until successful.
cl_int readDataFromCLBuffer (void *data, size_t bytes, cl_mem buffer) {
    cl_int error = CL_SUCCESS;
    printf("Performing blocking read of %lu bytes of data to %p\n", bytes, data);
    error = clEnqueueReadBuffer(queue, buffer, true, 0, bytes, data, 0, NULL, NULL);
    showCLError(error);
    return error;
}

// Create an OpenCL memory buffer of given size.
cl_mem createCLBuffer (size_t bufferSizeInBytes) {
    cl_int error = CL_SUCCESS;
	cl_mem univ = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSizeInBytes, NULL, &error);
	showCLError(error);
    return univ;
}

// Compile OpenCL C source file into a cl_program with given defines
cl_program buildCLProgram (const char* filename, const char* defines) {
    cl_int error = CL_SUCCESS;
	const char* source = loadFile(filename);
	cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &error);
	showCLError(error);
	printf("Program created from %s\n", filename);

	error = clBuildProgram (program, 0, NULL, defines, NULL, NULL), 
	showCLError(error);
	printf("Program built from %s\n", filename);
    
    return program;
}

// Convenience method to call buildCLProgram if no defines are needed
cl_program buildCLProgram (const char* filename) {
    return buildCLProgram(filename, "");
}

// Build a kernel with given name contained in given program
cl_kernel buildCLKernel (cl_program program, const char* kernelName) {
    cl_int error = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel (program, kernelName, &error);
	showCLError(error);
    return kernel;
}

// Execute a given kernel with a given global work size.
cl_int enqueueCLKernel (cl_kernel kernel, const size_t globalWorkSize) {
    cl_int error = CL_SUCCESS;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
	showCLError(error);
    return error;
}

// Wait for all pending OpenCL operations to finish
cl_int waitForCLOperations () {
    cl_int error = clFinish(queue);
	showCLError(error);
    return error;
}

// Get the build log for a given cl_program
char* buildLogForCLProgram(cl_program program) {
	char* out = (char *)malloc(sizeof(char)*1024*64);
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(char)*1024*64, out, NULL);
    return out;
}

// Destroy command queue and context
void releaseCLHelper () {
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

// Initialize a global OpenCL context for the first GPU found
void initOpenCL() {
	cl_int error = 0;   // Used to handle error codes
	cl_platform_id platform;
	
	// Platform
	error = clGetPlatformIDs(1, &platform, NULL);
	showCLError(error);

	// Device
	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	showCLError(error);
    printf("Device: %p\n", device);
    
	// Context
	context = clCreateContext(0, 1, &device, NULL, NULL, &error);
	showCLError(error);

	// Command-queue
	queue = clCreateCommandQueue(context, device, 0, &error);
	showCLError(error);
	printf("Command queue created!\n");
}

// Display OpenCL error and optionally assert error == CL_SUCCESS
inline void showCLError (cl_int error, bool assert) {
    if (error == CL_SUCCESS) {
        return;
    }
    printf("OpenCL Error: %s\n", clErrorString(error));
    
    if (assert) {
        assert (error == CL_SUCCESS);
    }
}

// Convenience method to call showCLError without assertion
inline void showCLError (cl_int error) {
    showCLError(error, false);
}

// Get string representation of OpenCL error code
const char* clErrorString(cl_int error) {
    static const char* errorString[] = {
        "CL_SUCCESS",
        "CL_DEVICE_NOT_FOUND",
        "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES",
        "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE",
        "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE",
        "CL_MAP_FAILURE",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "CL_INVALID_VALUE",
        "CL_INVALID_DEVICE_TYPE",
        "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE",
        "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE",
        "CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER",
        "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM",
        "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION",
        "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX",
        "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS",
        "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET",
        "CL_INVALID_EVENT_WAIT_LIST",
        "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION",
        "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE",
    };
    
    const int errorCount = sizeof(errorString) / sizeof(errorString[0]);
    const int index = -error;
    return (index >= 0 && index < errorCount) ? errorString[index] : "";
}


#endif
