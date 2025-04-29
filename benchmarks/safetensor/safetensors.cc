#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <liburing.h>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>
#include <string>
#include <sys/types.h>
#include <unistd.h>

#if !defined(SAFETENSORS_CPP_NO_IMPLEMENTATION)
#define SAFETENSORS_CPP_IMPLEMENTATION
#endif

#include "safetensors.hh"
#include "cufile_sample_utils.h"
#include "phoenix.h"

#define CUFILE_CHECK_ERROR(status) { \
    if (status.err != CU_FILE_SUCCESS) { \
        std::cerr << "(" << __func__ << ":" << __LINE__ << ")"; \
        std::cerr << "cufile error: " << cuFileGetErrorString(status) << "\n"; \
        exit(-1); \
    } \
} \

#define CUDA_CHECK_ERROR(status) { \
    if (status != cudaSuccess) { \
        std::cerr << "(" << __func__ << ":" << __LINE__ << ")"; \
        std::cerr << "cuda error: " << GetCudaErrorString(status) << "\n"; \
        exit(-1); \
    } \
} \

#define PHXFS_CHECK_ERROR(status) { \
    if (status != 0) { \
      std::cerr << "(" << __func__ << ":" << __LINE__ << ")"; \
      std::cerr << "phxfs error: " << status << "\n"; \
      exit(-1); \
    } \
} \

#define ROUND_UP(x, y) (((x) + (y)-1) & ~((y)-1))
#define ROUND_DOWN(x, y) ((x) & ~((y)-1))
#define GPU_PAGE_SIZE 64 * 1024
#define PAGE_SIZE 4096

static int device_id = 0;
static inline uint64_t tenser_to_device_phxfs(safetensors::safetensors_t &st, int fd){
  std::string key;
  safetensors::tensor_t tensor;
  uint64_t total_file_size = 0;
  CUDA_CHECK_ERROR(cudaSetDevice(device_id));
  for (size_t idx = 0; idx < st.tensors.size(); idx++) {
    key = st.tensors.keys()[idx];
    st.tensors.at(idx, &tensor);
    ssize_t offset = tensor.data_offsets[0] + 8 + st.header_size;
    ssize_t size = tensor.data_offsets[1] - tensor.data_offsets[0];

    // force align to 4K
    ssize_t aligned_offset = offset & ~(4096 - 1);
    ssize_t aligned_end = (offset + size + 4095) & ~(4096 - 1);
    ssize_t aligned_size = aligned_end - aligned_offset;
    ssize_t alloc_size = ROUND_UP(aligned_size, GPU_PAGE_SIZE);

    if (aligned_offset > (ssize_t)((safetensors::detail::safetensors_file *)st.st_file)->size) {
      std::cerr << "aligned_offset is " << aligned_offset << ", file size is " << ((safetensors::detail::safetensors_file *)st.st_file)->size << "\n";
      // return -1;
      continue;
    }

    CUDA_CHECK_ERROR(cudaMalloc(&tensor.dev_ptr, alloc_size));
    
    void *host_ptr = nullptr;
    auto ret = phxfs_regmem(device_id, tensor.dev_ptr, alloc_size, &host_ptr);
    if (ret){
      std::cerr << "phxfs_regmem error, ret is " << ret << ", size is " << alloc_size << "\n";
      return -1;
    }

    ssize_t result = phxfs_read({.fd = fd, .deviceID = device_id},
      tensor.dev_ptr, 0,
      aligned_size, aligned_offset);

    if (result != aligned_size && result < 0) {
      std::cerr << "read_thread error, result is " << result << ", size is " << aligned_size << "\n";
      return -1;
    }
    if (result == 0) {
      // End of file reached
      break;
    }
    total_file_size += size;
    PHXFS_CHECK_ERROR(phxfs_deregmem(device_id, tensor.dev_ptr, alloc_size));
  }
  return total_file_size;
}
  
static inline int read_tensor_phxfs(std::string &filename, uint64_t *done_size) {
  safetensors::safetensors_t st;

  std::string warn, err;

  auto ret = safetensors::mmap_from_file(filename, &st, &warn, &err, true);

  int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0644);
  if (fd < 0) {
    std::cerr << "Failed to open file: " << filename << "\n";
    return EXIT_FAILURE;
  }

  if (warn.size()) {
    std::cout << "WARN: " << warn << "\n";
  }

  if (!ret) {
    std::cerr << "Failed to load: " << filename << "\n";
    std::cerr << "  ERR: " << err << "\n";
    return EXIT_FAILURE;
  }

  if (!safetensors::validate_data_offsets(st, err)) {
    std::cerr << "Invalid data_offsets\n";
    std::cerr << err << "\n";
    return EXIT_FAILURE;
  }

  auto size = tenser_to_device_phxfs(st, fd);

  *done_size += size;
  return EXIT_SUCCESS;
}
  
int load_safetensors_phxfs(std::string &dir) {
    std::string warn, err;
    std::vector<std::string> file_paths;
    safetensors::safetensors_t st;
    struct timespec start, end;
  
    CUDA_CHECK_ERROR(cudaSetDevice(device_id));
    PHXFS_CHECK_ERROR(phxfs_open(device_id));
  
    std::string files = std::filesystem::path(dir).string();
    for (const auto& entry : std::filesystem::directory_iterator(dir)){
      std::string filename = entry.path().filename().string();
      if (filename.size() >= 12 && filename.substr(filename.size() - 12) == ".safetensors") {
        file_paths.push_back(std::string(dir) + "/" + filename);
      }
    }
  
    uint64_t done_size = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (auto &file : file_paths) {
      std::cout << "Processing file: " << file << "\n";
      if (read_tensor_phxfs(file, &done_size) != 0) {
        std::cerr << "Failed to read tensor from file: " << file << "\n";
        return EXIT_FAILURE;
      }
    }
  
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    std::cout << "Elapsed time: " << elapsed << " seconds\n";
    std::cout << "Total size: " << (1.0 * done_size / (1024 * 1024 * 1024)) << " GB\n";
    std::cout << "Throughput: " << (1.0 * done_size / (1024 * 1024 * 1024)) / elapsed << " GB/s\n";
  
  
    for (size_t idx = 0; idx < st.tensors.size(); idx++) {
      std::string key = st.tensors.keys()[idx];
      safetensors::tensor_t tensor;
      st.tensors.at(idx, &tensor);
  
      if (tensor.dev_ptr !=nullptr)
        CUDA_CHECK_ERROR(cudaFree(tensor.dev_ptr));
    }
  
    PHXFS_CHECK_ERROR(phxfs_close(device_id));
  
    return EXIT_SUCCESS;
  }
  


static inline uint64_t tenser_to_device_gds(safetensors::safetensors_t &st, CUfileHandle_t &cf_handle){
    std::string key;
    safetensors::tensor_t tensor;
    uint64_t total_file_size = 0;
    CUDA_CHECK_ERROR(cudaSetDevice(0));
    for (size_t idx = 0; idx < st.tensors.size(); idx++) {
      key = st.tensors.keys()[idx];
      st.tensors.at(idx, &tensor);
      ssize_t length = tensor.data_offsets[1] - tensor.data_offsets[0];
      CUDA_CHECK_ERROR(cudaMalloc(&tensor.dev_ptr, length));
      CUFILE_CHECK_ERROR(cuFileBufRegister(tensor.dev_ptr, length, 0));
      ssize_t result = cuFileRead(cf_handle,
        tensor.dev_ptr, length,
        tensor.data_offsets[0] + 8 + st.header_size, 0);
      if (result != length) {
        std::cerr << "read_thread error, result is " << result << ", size is " << length << "\n";
        return -1;
      }
      if (result == 0) {
        // End of file reached
        break;
      }
      total_file_size += length;
      CUFILE_CHECK_ERROR(cuFileBufDeregister(tensor.dev_ptr));
    }
    return total_file_size;
  }

static inline int read_tensor_gds(std::string &filename, uint64_t *done_size) {
    safetensors::safetensors_t st;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    std::string warn, err;
  
    auto ret = safetensors::mmap_from_file(filename, &st, &warn, &err, true);
  
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (fd < 0) {
      std::cerr << "Failed to open file: " << filename << "\n";
      return EXIT_FAILURE;
    }
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  
    CUFILE_CHECK_ERROR(cuFileHandleRegister(&cf_handle, &cf_descr));
    if (warn.size()) {
      std::cout << "WARN: " << warn << "\n";
    }
  
    if (!ret) {
      std::cerr << "Failed to load: " << filename << "\n";
      std::cerr << "  ERR: " << err << "\n";
      return EXIT_FAILURE;
    }
  
    if (!safetensors::validate_data_offsets(st, err)) {
      std::cerr << "Invalid data_offsets\n";
      std::cerr << err << "\n";
      return EXIT_FAILURE;
    }
  
    auto size = tenser_to_device_gds(st, cf_handle);
  
    *done_size += size;
    return EXIT_SUCCESS;
  }


int load_safetensors_gds(std::string &dir) {
    std::string warn, err;
    std::vector<std::string> file_paths;
    safetensors::safetensors_t st;
    struct timespec start, end;
  
    std::string files = std::filesystem::path(dir).string();
  
    CUDA_CHECK_ERROR(cudaSetDevice(0));
    CUFILE_CHECK_ERROR(cuFileDriverOpen());
  
    for (const auto& entry : std::filesystem::directory_iterator(dir)){
      std::string filename = entry.path().filename().string();
      if (filename.size() >= 12 && filename.substr(filename.size() - 12) == ".safetensors") {
        file_paths.push_back(std::string(dir) + "/" + filename);
      }
    }
  
    uint64_t done_size = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (auto &file : file_paths) {
      std::cout << "Processing file: " << file << "\n";
      if (read_tensor_gds(file, &done_size) != 0) {
        std::cerr << "Failed to read tensor from file: " << file << "\n";
        return EXIT_FAILURE;
      }
    }
  
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    std::cout << "Elapsed time: " << elapsed << " seconds\n";
    std::cout << "Total size: " << (1.0 * done_size / (1024 * 1024 * 1024)) << " GB\n";
    std::cout << "Throughput: " << (1.0 * done_size / (1024 * 1024 * 1024)) / elapsed << " GB/s\n";
  
    for (size_t idx = 0; idx < st.tensors.size(); idx++) {
      std::string key = st.tensors.keys()[idx];
      safetensors::tensor_t tensor;
      st.tensors.at(idx, &tensor);
  
      if (tensor.dev_ptr !=nullptr)
        CUDA_CHECK_ERROR(cudaFree(tensor.dev_ptr));
    }
  
    CUFILE_CHECK_ERROR(cuFileDriverClose());
  
    return EXIT_SUCCESS;
}

static inline uint64_t tenser_to_device_native(safetensors::safetensors_t &st){
  std::string key;
  safetensors::tensor_t tensor;
  uint64_t total_file_size = 0;
  CUDA_CHECK_ERROR(cudaSetDevice(0));
  for (size_t idx = 0; idx < st.tensors.size(); idx++) {
    key = st.tensors.keys()[idx];
    st.tensors.at(idx, &tensor);
    size_t length = tensor.data_offsets[1] - tensor.data_offsets[0] + 1;
    CUDA_CHECK_ERROR(cudaMalloc(&tensor.dev_ptr, length));
    CUDA_CHECK_ERROR(cudaMemcpy(
        tensor.dev_ptr, st.databuffer_addr + tensor.data_offsets[0], length,
        cudaMemcpyHostToDevice));
    total_file_size += length;
  }
  return total_file_size;
}

static inline int read_tensor_native(std::string &filename, uint64_t *done_size) {
    safetensors::safetensors_t st;
    std::string warn, err;
  
    auto ret = safetensors::mmap_from_file(filename, &st, &warn, &err, false);
  
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (fd < 0) {
      std::cerr << "Failed to open file: " << filename << "\n";
      return EXIT_FAILURE;
    }
  
    if (!ret) {
      std::cerr << "Failed to load: " << filename << "\n";
      std::cerr << "  ERR: " << err << "\n";
      return EXIT_FAILURE;
    }
  
    if (!safetensors::validate_data_offsets(st, err)) {
      std::cerr << "Invalid data_offsets\n";
      std::cerr << err << "\n";
      return EXIT_FAILURE;
    }
  
    auto size = tenser_to_device_native(st);
  
    *done_size += size;
    return EXIT_SUCCESS;
  }

int load_safetensors_native(std::string &dir) {
    std::string warn, err;
    std::vector<std::string> file_paths;
    safetensors::safetensors_t st;
    struct timespec start, end;
  
    CUDA_CHECK_ERROR(cudaSetDevice(0));
  
    // CUDA_CHECK_ERROR(status)
  
    std::string files = std::filesystem::path(dir).string();
    for (const auto& entry : std::filesystem::directory_iterator(dir)){
      std::string filename = entry.path().filename().string();
      if (filename.size() >= 12 && filename.substr(filename.size() - 12) == ".safetensors") {
        file_paths.push_back(std::string(dir) + "/" + filename);
      }
    }
  
    uint64_t done_size = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (auto &file : file_paths) {
      std::cout << "Processing file: " << file << "\n";
      if (read_tensor_native(file, &done_size) != 0) {
        std::cerr << "Failed to read tensor from file: " << file << "\n";
        return EXIT_FAILURE;
      }
    }
  
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    std::cout << "Elapsed time: " << elapsed << " seconds\n";
    std::cout << "Total size: " << (1.0 * done_size / (1024 * 1024 * 1024)) << " GB\n";
    std::cout << "Throughput: " << (1.0 * done_size / (1024 * 1024 * 1024)) / elapsed << " GB/s\n";
  
    for (size_t idx = 0; idx < st.tensors.size(); idx++) {
      std::string key = st.tensors.keys()[idx];
      safetensors::tensor_t tensor;
      st.tensors.at(idx, &tensor);
  
      if (tensor.dev_ptr !=nullptr)
        CUDA_CHECK_ERROR(cudaFree(tensor.dev_ptr));
    }
  
    return EXIT_SUCCESS;
}



int main(int argc, char *argv[]) {
    if (argc != 4){
        std::cout << "Usage: " << argv[0] << " <dir> <type> <device>" << std::endl;
        return -1;
    }
    std::string dir = argv[1];
    int type = atoi(argv[2]);
    device_id = atoi(argv[3]);
    
    switch (type) {
        case 0:
            load_safetensors_phxfs(dir);
            break;
        case 1:
            load_safetensors_gds(dir);
            break;
        case 2:
            load_safetensors_native(dir);
            break;
        default:
            std::cout << "Invalid type" << std::endl;
            return -1;
    }
}