#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x900000) // for reading in kernels

const char *getErrorString(cl_int error)
{
  switch(error){
  // run-time and JIT compiler errors
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  default: return "Unknown OpenCL error";
  }
}

#define CHECK_CL_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, char const* const func, char const* const file, int const line)
{
  if (err != CL_SUCCESS)
  {
    Rcpp::Rcerr << "OpenCL runtime error at: " << file << ":" << line << std::endl;
    Rcpp::Rcerr << getErrorString(err) << " " << func << std::endl;
    Rcpp::stop("");
  }
}
#define CHECK_CL_ERROR_AFTER(val) checkAfter((val), __FILE__, __LINE__)
template <typename T>
void checkAfter(T err, char const* const file, int const line)
{
  if (err != CL_SUCCESS)
  {
    Rcpp::Rcerr << "OpenCL runtime error at: " << file << ":" << line << std::endl;
    Rcpp::Rcerr << getErrorString(err) << std::endl;
    Rcpp::stop("");
  }
}

template <typename T>
void CHECK_CL_PROGRAM_ERROR(T err, cl_program program, cl_device_id device_id) {
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char *) malloc(log_size);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    printf("%s\n", log);
    Rcpp::stop("");
  }
}

typedef struct kernel_source {
  char *str;
  size_t size;
} kernel_source;

kernel_source get_kernel_source(const char* path_to_file)
{
  kernel_source s;
  FILE *fp;
  fp = fopen(path_to_file, "r");
  s.str = (char*)malloc(MAX_SOURCE_SIZE);
  s.size = fread( s.str, 1, MAX_SOURCE_SIZE, fp);
  fclose( fp );

  return s;
}

// [[Rcpp::export]]
void print_opencl_devices() {
  
  char* value;
  size_t valueSize;
  cl_uint platformCount;
  cl_platform_id* platforms;
  cl_uint deviceCount;
  cl_device_id* devices;
  cl_uint maxComputeUnits;
  
  // get all platforms
  clGetPlatformIDs(0, NULL, &platformCount);
  platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
  clGetPlatformIDs(platformCount, platforms, NULL);
  
  for (unsigned int i = 0; i < platformCount; i++) {
    
    // get all devices
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
    
    // for each device print critical attributes
    for (unsigned int j = 0; j < deviceCount; j++) {
      
      // print device name
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
      value = (char*) malloc(valueSize);
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
      printf("%d. Device: %s\n", j+1, value);
      free(value);
      
      // print hardware device version
      clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
      value = (char*) malloc(valueSize);
      clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
      printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
      free(value);
      
      // print software driver version
      clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
      value = (char*) malloc(valueSize);
      clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
      printf(" %d.%d Software version: %s\n", j+1, 2, value);
      free(value);
      
      // print c version supported by compiler for device
      clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
      value = (char*) malloc(valueSize);
      clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
      printf(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
      free(value);
      
      // print parallel compute units
      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                      sizeof(maxComputeUnits), &maxComputeUnits, NULL);
      printf(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);
      
    }
    
    free(devices);
    
  }
  
  free(platforms);
  
}

// [[Rcpp::export]]
void print_opencl_platforms() {
  
  char* info;
  size_t infoSize;
  cl_uint platformCount;
  cl_platform_id *platforms;
  const char* attributeNames[5] = { "Name", "Vendor", "Version", "Profile", "Extensions" };
  const cl_platform_info attributeTypes[5] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
                                               CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };
  const int attributeCount = sizeof(attributeNames) / sizeof(char*);
  
  // get platform count
  clGetPlatformIDs(5, NULL, &platformCount);
  
  // get all platforms
  platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
  clGetPlatformIDs(platformCount, platforms, NULL);
  
  // for each platform print all attributes
  for (unsigned int i = 0; i < platformCount; i++) {
    
    printf("\n %d. Platform \n", i+1);
    
    for (unsigned int j = 0; j < attributeCount; j++) {
      
      // get platform attribute value size
      clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
      info = (char*) malloc(infoSize);
      
      // get platform attribute value
      clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);
      
      printf("  %d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);
      free(info);
      
    }
    
    printf("\n");
    
  }
  
  free(platforms);
  
}