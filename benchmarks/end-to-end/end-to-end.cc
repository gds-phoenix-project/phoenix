
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <cstddef>
#include <string>
#include <cuda_runtime.h>
#include <vector>
#include "cufile_sample_utils.h"
#include "phoenix.h"
#include "cufile.h"


struct IOParams{
    int loop_idx;
    int device_id;
    ssize_t io_size;
    std::string file_path;
    std::vector<unsigned long long> latency_vec;
};

static inline void loop_gds_large(struct IOParams *params){
    struct timespec start, end;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    CUfileError_t status;
    int file_fd;
    void *gpu_buffer = NULL;
    ssize_t result;

    file_fd = open(params->file_path.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0664);
    if (file_fd < 0){
        printf("open file failed\n");
        exit(1);
    }
    check_cudaruntimecall(cudaSetDevice(0));

    check_cudaruntimecall(cudaMalloc(&gpu_buffer, params->io_size));
    check_cudaruntimecall(cudaMemset(gpu_buffer, 0xab, params->io_size));
    check_cudaruntimecall(cudaStreamSynchronize(0));

    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = file_fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    clock_gettime(CLOCK_MONOTONIC, &start);
    status = cuFileDriverOpen();
    
    if (status.err != CU_FILE_SUCCESS){
        printf("cuFileDriverOpen failed\n");
        exit(1);
    }
    
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS){
        printf("cuFileHandleRegister failed\n");
        exit(1);
    }
    
    status = cuFileBufRegister(gpu_buffer, params->io_size, 0);
    if (status.err != CU_FILE_SUCCESS){
        printf("cuFileBufRegister failed\n");
        exit(1);
    }
    
    result = cuFileRead(cf_handle, gpu_buffer, params->io_size, 
        params->loop_idx * params->io_size, 0);
    if (result != params->io_size){
        printf("cuFileRead failed\n");
        exit(1);
    }
    
    cuFileBufDeregister(gpu_buffer);
    cuFileHandleDeregister(cf_handle);
    cuFileDriverClose();
    clock_gettime(CLOCK_MONOTONIC, &end);

    params->latency_vec.push_back((end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec));
    check_cudaruntimecall(cudaFree(gpu_buffer));
    check_cudaruntimecall(cudaStreamSynchronize(0));
}

static inline void loop_phxfs_large(struct IOParams *params){
    struct timespec start, end;
    int file_fd;
    int device_id = 0;
    void *gpu_buffer;
    void *target_addr;
    int ret;
    ssize_t result;
    
    check_cudaruntimecall(cudaSetDevice(device_id));
    file_fd = open(params->file_path.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0664);
    if (file_fd < 0){
        printf("open file failed\n");
        exit(1);
    }

    check_cudaruntimecall(cudaMalloc(&gpu_buffer, params->io_size));
    check_cudaruntimecall(cudaMemset(gpu_buffer, 0xab, params->io_size));
    check_cudaruntimecall(cudaStreamSynchronize(0));

    clock_gettime(CLOCK_MONOTONIC, &start);
    ret = phxfs_open(0);
    if (ret){
        printf("phxfs open failed\n");
        exit(1);
    }
    
    ret = phxfs_regmem(device_id, gpu_buffer, params->io_size, &target_addr);

    if (ret){
        printf("phxfs regmem failed\n");
        exit(1);
    }

    result = phxfs_read({.fd = file_fd, .deviceID = device_id}, gpu_buffer, 0, params->io_size, params->loop_idx * params->io_size);
    if (result != params->io_size){
        printf("phxfs read failed\n");
        exit(1);
    }

    ret = phxfs_deregmem(device_id, gpu_buffer, params->io_size);
    if (ret){
        printf("phxfs unregmem failed\n");
        exit(1);
    }

    phxfs_close(device_id);
    clock_gettime(CLOCK_MONOTONIC, &end);
    close(file_fd);
    check_cudaruntimecall(cudaFree(gpu_buffer));
    params->latency_vec.push_back((end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec));
}

static ssize_t loop_cnt = 10;
int main(int argc, char* argv[]){
    if (argc != 4){
        printf("Usage: %s <file_path> <io_size> <mode>\n", argv[0]);
        return -1;
    }
    int mode = atoi(argv[3]);
    struct IOParams params;
    params.file_path = argv[1];
    params.io_size = atoll(argv[2]);
    params.loop_idx = 0;
    for (auto i = 0; i < loop_cnt; i++){
        if (mode == 0){
            loop_phxfs_large(&params);
        } else {
            loop_gds_large(&params);
        }
    }
    unsigned long long total_time = 0;
    for (size_t i = 0; i < params.latency_vec.size(); i++){
        total_time += params.latency_vec[i];
    }
    total_time /= 1000.0;
    std::cout << "Test mode: " << (mode == 0 ? "PHXFS" : "GDS") << std::endl;
    std::cout << "Total IO operations: " << params.latency_vec.size() << std::endl;
    std::cout << "IO size: " << params.io_size << std::endl;
    std::cout << "Total loop count: " << loop_cnt << std::endl;
    std::cout << "Average time: " << std::fixed << (double)total_time / params.latency_vec.size() << " us" << std::endl;
    return 0;
}