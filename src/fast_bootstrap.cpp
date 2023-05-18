#include <Rcpp.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <random_generator/random_generator.h>
#include <opencl_utilities.h>
#include <fstream>
#include <string>
#include <stdint.h>
#include <stdio.h>
#include <cmath>
#include <unistd.h>

template <typename T>
class opencl_bootstrap_manager {
  
  public:

    opencl_bootstrap_manager(int bootstrap_samples, int seed) : bootstrap_samples(bootstrap_samples), seed(seed) {
      set_local_item_size(32);
      setup_device();
    }
  
    void set_local_item_size(int item_size) {
      local_item_size = (size_t) item_size;
      global_item_size = (size_t) local_item_size * ceil( ((float)bootstrap_samples) / ((float)local_item_size) );
    }

    Rcpp::NumericVector get_bootstrapped_means(Rcpp::NumericVector x) {
      Rcpp::NumericVector out_r(bootstrap_samples);
      
      int nr_of_values = x.length();
      int nr_of_values_not_na = nr_of_values;
      T* h_values = new T[nr_of_values]();
      int idx = 0;
      for(int i = 0; i < nr_of_values; i++) {
        if (Rcpp::NumericVector::is_na(x[i])) {
          nr_of_values_not_na--;
        } else {
          h_values[idx] = x[i];
          idx++;
        }
      }
      if (nr_of_values_not_na == 0) {
        for(int i = 0; i < bootstrap_samples; i++) {
          out_r[i] = Rcpp::NumericVector::get_na();
        }
        return(out_r);
      }
      
      
      T* h_out = new T[bootstrap_samples];
      
      calc_bootstrap(h_values, h_out, nr_of_values_not_na);

      
      for(int i = 0; i < bootstrap_samples; i++) {
        out_r[i] = h_out[i];
      }
      
      delete [] h_values;
      delete [] h_out;
      
      return(out_r);
    }
  
    
    ~opencl_bootstrap_manager() {
      delete[] rand_states;
      CHECK_CL_ERROR(clFinish(command_queue));
      CHECK_CL_ERROR(clReleaseCommandQueue(command_queue));
      CHECK_CL_ERROR(clReleaseProgram(program));
      CHECK_CL_ERROR(clReleaseContext(context));
      CHECK_CL_ERROR(clReleaseKernel(bootstrap_kernel));
      CHECK_CL_ERROR(clReleaseMemObject(buffer_rand_states));
      CHECK_CL_ERROR(clReleaseMemObject(buffer_output));
    };
    
    
  private:
    int bootstrap_samples;
    cl_device_id device_id;
    int seed;
    size_t global_item_size;
    size_t local_item_size;
    kernel_source kernel_source_code;
    xorwow_state* rand_states;
    cl_program program;
    cl_context context;
    cl_kernel bootstrap_kernel;
    cl_command_queue command_queue;
    cl_mem buffer_output;
    cl_mem buffer_rand_states;
    
    void set_default_device_id() {
      cl_platform_id platform_id = NULL;
      cl_device_id default_device_id = NULL;
      cl_uint num_devices;
      cl_uint num_platforms;
      CHECK_CL_ERROR(clGetPlatformIDs(1, &platform_id, &num_platforms));
      CHECK_CL_ERROR(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &default_device_id, &num_devices));
      device_id = default_device_id;
    }
    
    void set_kernel_source() {
      kernel_source_code = get_kernel_source();
    }
    
    void init_rand_states() {
      rand_states = new xorwow_state[bootstrap_samples];
      for(int i = 0; i < bootstrap_samples; i++) {
        init_xorwow(rand_states + i, seed, (unsigned long long) i);
      }
    }

    void setup_device()
    {
      cl_int err;
      
      set_default_device_id();
      set_kernel_source();
      init_rand_states();

      context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
      CHECK_CL_ERROR_AFTER(err);

      program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source_code.str, (const size_t *)&kernel_source_code.size, &err);
      CHECK_CL_ERROR_AFTER(err);

      err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
      CHECK_CL_PROGRAM_ERROR(err, program, device_id);
      CHECK_CL_ERROR_AFTER(err);
      
      bootstrap_kernel = clCreateKernel(program, "bootstrap_kernel", &err);
      CHECK_CL_ERROR_AFTER(err);
      
      command_queue = clCreateCommandQueue(context, device_id, 0, &err);
      CHECK_CL_ERROR_AFTER(err);
      
      buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bootstrap_samples * sizeof(T), NULL, &err);
      CHECK_CL_ERROR_AFTER(err);
      
      buffer_rand_states = clCreateBuffer(context, CL_MEM_READ_ONLY, bootstrap_samples * sizeof(xorwow_state), NULL, &err);
      CHECK_CL_ERROR_AFTER(err);
      CHECK_CL_ERROR(clEnqueueWriteBuffer(command_queue, buffer_rand_states, CL_TRUE, 0, bootstrap_samples * sizeof(xorwow_state), rand_states, 0, NULL, NULL));
      
      
      CHECK_CL_ERROR(clSetKernelArg(bootstrap_kernel, 0, sizeof(cl_mem), (void *)&buffer_rand_states));
      CHECK_CL_ERROR(clSetKernelArg(bootstrap_kernel, 1, sizeof(int), (void *)&bootstrap_samples));
      CHECK_CL_ERROR(clSetKernelArg(bootstrap_kernel, 2, sizeof(cl_mem), (void *)&buffer_output));
      
    }
    
    void calc_bootstrap(T* h_values, T* h_output, int nr_of_values) {
      
      cl_int err;

      cl_mem d_values = clCreateBuffer(context, CL_MEM_READ_ONLY, nr_of_values * sizeof(T), NULL, &err);
      CHECK_CL_ERROR_AFTER(err);

      CHECK_CL_ERROR(clEnqueueWriteBuffer(command_queue, d_values, CL_TRUE, 0, nr_of_values * sizeof(T), h_values, 0, NULL, NULL));
      CHECK_CL_ERROR(clSetKernelArg(bootstrap_kernel, 3, sizeof(cl_mem), (void *)&d_values));
      CHECK_CL_ERROR(clSetKernelArg(bootstrap_kernel, 4, sizeof(int), (void *)&nr_of_values));
      CHECK_CL_ERROR(clEnqueueNDRangeKernel(command_queue, bootstrap_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL));
      CHECK_CL_ERROR(clEnqueueReadBuffer(command_queue, buffer_output, CL_TRUE, 0, bootstrap_samples * sizeof(T), h_output, 0, NULL, NULL));
      
      CHECK_CL_ERROR(clReleaseMemObject(d_values));
    }
};

// [[Rcpp::export]]
Rcpp::NumericVector test_rand_gen_host(const int n, const int seed = 0) {
  Rcpp::NumericVector output(n);
  xorwow_state *state = (xorwow_state*)malloc(n * sizeof(xorwow_state));
  for(int i = 0; i < n; i++) {
    init_xorwow(state + i, seed, i);
    output[i] = rand(state + i);
  }
  return(output);
}

typedef opencl_bootstrap_manager<float> opencl_bootstrap_manager_float;

RCPP_EXPOSED_CLASS_NODECL(opencl_bootstrap_manager_float)
RCPP_MODULE(opencl_bootstrap_manager_float) {
  Rcpp::class_<opencl_bootstrap_manager_float>("opencl_bootstrap_manager_float")
  
  .constructor<int,int>()
  .method("get_bootstrapped_means", &opencl_bootstrap_manager_float::get_bootstrapped_means, "get bootstrapped means for numeric vector")
  .method("set_local_item_size" ,&opencl_bootstrap_manager_float::set_local_item_size, "set opencl local item size (default is 32)")
  ;
}
