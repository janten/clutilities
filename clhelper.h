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

#if defined __APPLE__ || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

const char* clErrorString(cl_int error);
cl_int showCLError (cl_int error);

// Use one global queue
cl_command_queue queue;
cl_context context;
cl_device_id device = 0;

/**
 *  Get file contents.
 *
 *  @param fname Path to file that will be opened
 *
 *  @return The contents of the file at the specified path
 */
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

/**
 *  Perform a blocking write operation to the given buffer.
 *
 *  @param data   A pointer to the data to be copied, in host memory
 *  @param bytes  Amount of data in bytes to be copied
 *  @param buffer Target buffer where the data will be copied to
 *
 *  @return OpenCL error value
 */
cl_int writeDataToCLBuffer (void *data, size_t bytes, cl_mem buffer) {
    cl_int error = CL_SUCCESS;
    printf("Performing blocking write of %lu bytes of data from %p\n", bytes, data);
    error = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, bytes, data, 0, NULL, NULL);
    showCLError(error);
    return error;
}

/**
 *  Perform a blocking read operation from the given buffer.
 *
 *  @param data   A pointer in host memory
 *  @param bytes  Amount of data in bytes to be copied
 *  @param buffer Source buffer where the data will be copied from
 *
 *  @return OpenCL error value
 */
cl_int readDataFromCLBuffer (void *data, size_t bytes, cl_mem buffer) {
    cl_int error = CL_SUCCESS;
    printf("Performing blocking read of %lu bytes of data to %p\n", bytes, data);
    error = clEnqueueReadBuffer(queue, buffer, true, 0, bytes, data, 0, NULL, NULL);
    showCLError(error);
    return error;
}

/**
 *  Create a buffer object.
 *
 *  @param bufferSizeInBytes Desired size of the buffer in bytes
 *
 *  @return A buffer object
 */
cl_mem createCLBuffer (size_t bufferSizeInBytes) {
    cl_int error = CL_SUCCESS;
	cl_mem univ = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSizeInBytes, NULL, &error);
	showCLError(error);
    return univ;
}

/**
 *  Build a OpenCL program by compiling the source code from the specified file
 *  using the given defines.
 *
 *  @param filename Path to the kernel source file
 *  @param defines  Custom defines to use during compilation. May be NULL.
 *
 *  @return A program
 */
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

/**
 *  Build a kernel that is part of the given program.
 *
 *  @param program    The program which contains the kernel
 *  @param kernelName The name of the kernel
 *
 *  @return A kernel object
 */
cl_kernel buildCLKernel (cl_program program, const char* kernelName) {
    cl_int error = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel (program, kernelName, &error);
	showCLError(error);
    return kernel;
}

/**
 *  Enqueue a given kernel for execution
 *
 *  @param kernel         The kernel to be executed
 *  @param globalWorkSize The amount of threads to be run
 *
 *  @return An error value
 */
cl_int enqueueCLKernel (cl_kernel kernel, const size_t globalWorkSize) {
    cl_int error = CL_SUCCESS;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
	showCLError(error);
    return error;
}


/**
 *  Block the execution of host code until all OpenCL operations have finished.
 *
 *  @return An error value
 */
cl_int waitForCLOperations () {
    cl_int error = clFinish(queue);
	showCLError(error);
    return error;
}

/**
 *  Retrieve the build log for a program
 *
 *  @param program The program to retrieve the build log for
 *
 *  @return The log
 */
char* buildLogForCLProgram(cl_program program) {
	char* out = (char *)malloc(sizeof(char)*1024*64);
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(char)*1024*64, out, NULL);
    return out;
}

/**
 *  Release internal resources
 */
void releaseCLHelper () {
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}


/**
 *  Initialize globally used resources
 */
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

/**
 *  Display information about the given error code in human readable form.
 *
 *  @param error An OpenCL error value
 *
 *  @return The input value, for convenience
 */
inline cl_int showCLError (cl_int error) {
    if (error != CL_SUCCESS) {
		printf("OpenCL Error: %s\n", clErrorString(error));
    }
    
    return error;
}

/**
 *  Return a human readable string representation for common error values.
 *
 *  @param error An OpenCL error code
 *
 *  @return The error code's readable name
 */
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
