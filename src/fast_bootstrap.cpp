#include <Rcpp.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <opencl_utilities.h>
#include <fstream>
#include <string>
#include <stdint.h>
#include <stdio.h>
#include <cmath>
#include <unistd.h>

typedef struct t_xorwow_state {
  cl_uint x[5];
  cl_uint d;
} xorwow_state;

template <typename T>
class opencl_bootstrap_manager {
  
  public:

    opencl_bootstrap_manager(int nr_bootstraps_, int seed_)
    {
      set_local_item_size(32);
      set_parameters(nr_bootstraps_, seed_);
    }
  
    void set_parameters(int nr_bootstraps_, int seed_) {
      if (command_queue) {
        CHECK_CL_ERROR(clFinish(command_queue));
      }
      if (buffer_rand_states) {
        CHECK_CL_ERROR(clReleaseMemObject(buffer_rand_states));
      }
      if (buffer_output) {
        CHECK_CL_ERROR(clReleaseMemObject(buffer_output));
      }
      
      nr_bootstraps = nr_bootstraps_;
      seed = seed_;
      setup_device();
    }
  
    void set_local_item_size(int item_size) {
      local_item_size = (size_t) item_size;
      global_item_size = (size_t) local_item_size * ceil( ((float) nr_bootstraps) / ((float) local_item_size) );
    }

    std::vector<T> get_bootstrapped_means(std::vector<T> x) {
      std::vector<T> h_out(nr_bootstraps);
      calc_bootstrap_on_gpu(&x[0], &h_out[0], x.size());
      return(h_out);
    }
  
    ~opencl_bootstrap_manager() {

    };
    
    void cleanup_device() {
      CHECK_CL_ERROR(clFinish(command_queue));
      CHECK_CL_ERROR(clReleaseCommandQueue(command_queue));
      CHECK_CL_ERROR(clReleaseProgram(program));
      CHECK_CL_ERROR(clReleaseContext(context));
      CHECK_CL_ERROR(clReleaseKernel(bootstrap_kernel));
      CHECK_CL_ERROR(clReleaseKernel(init_xorwow_kernel_kernel));
      CHECK_CL_ERROR(clReleaseMemObject(buffer_rand_states));
      CHECK_CL_ERROR(clReleaseMemObject(buffer_output));
    }
    
    std::vector<unsigned int> test_rand_gen_device(int n = 10) {
      cl_int err;
      const size_t cl_n = n;
      std::vector<unsigned int> output(n);
      cl_kernel gen_random_kernel_int = clCreateKernel(program, "gen_random_kernel_int", &err);
      cl_mem buffer_output_test = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(unsigned int), NULL, &err);
      
      CHECK_CL_ERROR(clSetKernelArg(gen_random_kernel_int, 0, sizeof(cl_mem), (void *)&buffer_rand_states));
      CHECK_CL_ERROR(clSetKernelArg(gen_random_kernel_int, 1, sizeof(cl_mem), (void *)&buffer_output_test));
      CHECK_CL_ERROR(clSetKernelArg(gen_random_kernel_int, 2, sizeof(int), (void *)&n));
      CHECK_CL_ERROR(clEnqueueNDRangeKernel(command_queue, gen_random_kernel_int, 1, NULL, &cl_n, &cl_n, 0, NULL, NULL));

      CHECK_CL_ERROR(clEnqueueReadBuffer(command_queue, buffer_output_test, CL_TRUE, 0, n * sizeof(unsigned int), &output[0], 0, NULL, NULL));
      return(output);
    }
    
  private:
    
    int nr_bootstraps;
    cl_device_id device_id;
    int seed;
    size_t global_item_size;
    size_t local_item_size;
    kernel_source kernel_source_code;
    cl_program program;
    cl_context context;
    cl_kernel bootstrap_kernel;
    cl_kernel init_xorwow_kernel_kernel;
    cl_command_queue command_queue = NULL;
    cl_mem buffer_output = NULL;
    cl_mem buffer_rand_states = NULL;
    
    
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
      kernel_source_code = get_kernel_source("inst/include/kernels.cl");
    }
    
    void init_rand_states_device() {
      cl_int err;
      
      buffer_rand_states = clCreateBuffer(context, CL_MEM_READ_WRITE, nr_bootstraps * sizeof(xorwow_state), NULL, &err);

      init_xorwow_kernel_kernel = clCreateKernel(program, "init_xorwow_kernel", &err);
      CHECK_CL_ERROR_AFTER(err);
      CHECK_CL_ERROR(clSetKernelArg(init_xorwow_kernel_kernel, 0, sizeof(cl_mem), (void *)&buffer_rand_states));
      CHECK_CL_ERROR(clSetKernelArg(init_xorwow_kernel_kernel, 1, sizeof(int), (void *)&nr_bootstraps));
      CHECK_CL_ERROR(clSetKernelArg(init_xorwow_kernel_kernel, 2, sizeof(int), (void *)&seed));

      CHECK_CL_ERROR(clEnqueueNDRangeKernel(command_queue, init_xorwow_kernel_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL));
      
    }

    void setup_device()
    {
      cl_int err;
      
      set_default_device_id();
      set_kernel_source();

      context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
      CHECK_CL_ERROR_AFTER(err);

      program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source_code.str, (const size_t *)&kernel_source_code.size, &err);
      CHECK_CL_ERROR_AFTER(err);

      err = clBuildProgram(program, 1, &device_id, "-Iinst/include", NULL, NULL);
      CHECK_CL_PROGRAM_ERROR(err, program, device_id);
      CHECK_CL_ERROR_AFTER(err);
      
      bootstrap_kernel = clCreateKernel(program, "bootstrap_kernel", &err);
      CHECK_CL_ERROR_AFTER(err);
      
      command_queue = clCreateCommandQueue(context, device_id, 0, &err);
      CHECK_CL_ERROR_AFTER(err);
      
      buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, nr_bootstraps * sizeof(T), NULL, &err);
      CHECK_CL_ERROR_AFTER(err);
      
      init_rand_states_device();
      
      CHECK_CL_ERROR(clSetKernelArg(bootstrap_kernel, 0, sizeof(cl_mem), (void *)&buffer_rand_states));
      CHECK_CL_ERROR(clSetKernelArg(bootstrap_kernel, 1, sizeof(int), (int *)&nr_bootstraps));
      CHECK_CL_ERROR(clSetKernelArg(bootstrap_kernel, 2, sizeof(cl_mem), (void *)&buffer_output));
      
    }
    
    void calc_bootstrap_on_gpu(T* values, T* h_out, int nr_values) {
      
      cl_int err;
      
      cl_mem d_values = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nr_values * sizeof(T), values, &err);
      CHECK_CL_ERROR_AFTER(err);
      CHECK_CL_ERROR(clSetKernelArg(bootstrap_kernel, 3, sizeof(cl_mem), (void *)&d_values));
      CHECK_CL_ERROR(clSetKernelArg(bootstrap_kernel, 4, sizeof(int), (void *)&nr_values));
      CHECK_CL_ERROR(clEnqueueNDRangeKernel(command_queue, bootstrap_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL));
      CHECK_CL_ERROR(clEnqueueReadBuffer(command_queue, buffer_output, CL_TRUE, 0, nr_bootstraps * sizeof(T), h_out, 0, NULL, NULL));
      
      CHECK_CL_ERROR(clReleaseMemObject(d_values));
    }
};


typedef opencl_bootstrap_manager<float> opencl_bootstrap_manager_float;

void finalizer_opencl_bootstrap_manager(opencl_bootstrap_manager_float* ptr){
  ptr->cleanup_device();
}

RCPP_EXPOSED_CLASS_NODECL(opencl_bootstrap_manager_float)
RCPP_MODULE(opencl_bootstrap_manager_float) {
  Rcpp::class_<opencl_bootstrap_manager_float>("opencl_bootstrap_manager_float")
  
  .constructor<int,int>("sets the nr of bootstrap samples and the seed")
  .method("get_bootstrapped_means", &opencl_bootstrap_manager_float::get_bootstrapped_means, "get bootstrapped means for numeric vector")
  .method("set_local_item_size" ,&opencl_bootstrap_manager_float::set_local_item_size, "set opencl local item size (default is 32)")
  .method("set_parameters", &opencl_bootstrap_manager_float::set_parameters, "set the nr of bootstrap samples and the seed")
  .method("test_rand_gen_device", &opencl_bootstrap_manager_float::test_rand_gen_device, "test random numbers generated from device")
  .finalizer(finalizer_opencl_bootstrap_manager )
  ;
  
  Rcpp::function("print_opencl_platforms", &print_opencl_platforms, "print all available opencl platforms");
  Rcpp::function("print_opencl_devices", &print_opencl_devices, "print all available opencl devices");
}

