//
//  main.cpp
//  Aufgabe4
//
//  Created by Jan-Gerd Tenberge, Manuel Radek on 26.04.12.
//  Copyright (c) 2012 Manuel Radek. All rights reserved.
//
//  Simple demonstration application using clhelper.h to
//  simulate a three body problem with leapfrog integration
//  scheme on you GPU.
//
//  On Mac OS X compile with
//      clang++ main.cpp -framework OpenCL -o ncl
//
//  On Linux use
//      g++ main.cpp -lOpenCL -o ncl

#include <iostream>
#include <cmath>
#include <assert.h>
#include <sys/time.h>

#include "../clhelper.h"

typedef struct {
    cl_float x_old, y_old, z_old;
    cl_float x,y,z;
    cl_float vx,vy,vz;
    cl_float ax,ay,az;
    cl_float mass;
} Planet;

void leap_frog(Planet* universe, double dt, int nTimes);
void accelerations(Planet* universe, int nTimes);

int main(int argc, const char * argv[]) {
    // Simulation parameters
    cl_float T  = 100;  // Maximum time
    cl_float dt = 0.01; // Length of timestep
    int planets = 3;    // Number of bodies (sun, earth, moon)

    Planet universe[planets];
    Planet output[planets];
    memset(universe, 0, planets*sizeof(Planet));
    printf("Size of universe: %lu bytes\n", sizeof(universe));
    
    // Initialisation
    
    // Sun at center - according to G. Galilei
    universe[0].x_old = 0;
    universe[0].y_old = 0;
    universe[0].z_old = 0;
    universe[0].mass  = 5;
    
    // Earth
    universe[1].x_old = 40;
    universe[1].y_old =  0;
    universe[1].z_old =  0;
    universe[1].mass  = 81.3;
    universe[1].vy    =  2;
    
    // Moon
    universe[2].x_old = 44;
    universe[2].y_old =  0;
    universe[2].z_old =  0;
    universe[2].mass  = 10;
    universe[2].vy    = -0.8;
    
    printf("Position of sun:   %9.6f, %9.6f, %9.6f\n", universe[0].x_old, universe[0].y_old, universe[0].z_old);
	printf("Position of earth: %9.6f, %9.6f, %9.6f\n", universe[1].x_old, universe[1].y_old, universe[1].z_old);
	printf("Position of moon:  %9.6f, %9.6f, %9.6f\n", universe[2].x_old, universe[2].y_old, universe[2].z_old);

	// Initialize OpenCL and create leap_frog integration kernel
    initOpenCL();
	cl_program program = buildCLProgram("calc.cl");
    cl_kernel kernel = buildCLKernel(program, "leap_frog");

    // Create memory buffer andy copy universe
	cl_mem univ = createCLBuffer(sizeof(universe));
	writeDataToCLBuffer(universe, sizeof(universe), univ);
    waitForCLOperations();
    
    // Set kernel arguments
    cl_int error = CL_SUCCESS;
	error = clSetKernelArg (kernel, 0, sizeof(cl_mem), &univ);
    error = clSetKernelArg (kernel, 1, sizeof(cl_float), &dt);
	error = clSetKernelArg (kernel, 2, sizeof(cl_int), &planets);
	error = clSetKernelArg (kernel, 3, sizeof(cl_float), &T);
	showCLError(error);
    
    // Run kernel
	const size_t globalSize = planets;
	enqueueCLKernel(kernel, globalSize);
    waitForCLOperations();
    
    // Read results
	readDataFromCLBuffer(output, sizeof(universe), univ);
    waitForCLOperations();

    // Clean up
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(univ);
    releaseCLHelper();

	printf("%lu Bytes of data read from graphics card\n", sizeof(universe));

	printf("Position of sun:   %9.6f, %9.6f, %9.6f\n", output[0].x, output[0].y, output[0].z);
	printf("Position of earth: %9.6f, %9.6f, %9.6f\n", output[1].x, output[1].y, output[1].z);
	printf("Position of moon:  %9.6f, %9.6f, %9.6f\n", output[2].x, output[2].y, output[2].z);
    
    return 0;
}