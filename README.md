clutilities
===========

Various OpenCL helpers and utilities, all made for Mac OS X.

clinfo
------
Program to enumerate and dump all of the OpenCL information for a machine (or at least for a specific run-time).

clhelper
--------
Include this header to easily create a global OpenCL context and run your kernels there. Example usage:

	initOpenCL();
	cl_program program = buildCLProgram("source.cl");
	cl_kernel  kernel  = buildCLKernel(program, "myKernel");
	cl_int     status  = enqueueCLKernel(kernel, 1024);
	showCLError(status);
	waitForCLOperations();
	releaseCLHelper();

A simple simulation for the n-body problem using the leapfrog integration scheme is included.