#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cufile.h>
#include <builtin_types.h>
#include <iostream>
#include <pthread.h>
#include <string>
#include <sys/types.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include "cufile_sample_utils.h"

#include "phxfs_utils.h"
#include "phxfs.h"

static int TEST_REPEAT = 10;
static size_t IO_SIZE = 1024 * 64;
const char *file_path = "/mnt/phxfs/test.data";


void loop_gds_test(){
    struct timespec start, end;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    CUfileError_t status;
    unsigned long long  completed_bytes = 0;
    unsigned long iter_time = 0;
    int file_fd;
    void *gpu_buffer[TEST_REPEAT];
    ssize_t result;


    check_cudaruntimecall(cudaSetDevice(0));

    status = cuFileDriverOpen();
    
    if (status.err != CU_FILE_SUCCESS){
        printf("cuFileDriverOpen failed\n");
        exit(1);
    }

    file_fd = open(file_path, O_RDWR | O_CREAT | O_DIRECT, 0664);
    if (file_fd < 0){
        printf("open file failed\n");
        exit(1);
    }
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = file_fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    
    
    if (status.err != CU_FILE_SUCCESS){
        printf("cuFileHandleRegister failed\n");
        exit(1);
    }

    for (int i=0; i<TEST_REPEAT; i++){
        clock_gettime(CLOCK_MONOTONIC, &start);
        check_cudaruntimecall(cudaMalloc(&gpu_buffer[i], IO_SIZE));
        clock_gettime(CLOCK_MONOTONIC, &end);
        iter_time += (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
        check_cudaruntimecall(cudaMemset(gpu_buffer[i], 0x00, IO_SIZE));
        check_cudaruntimecall(cudaStreamSynchronize(0));
    }
    std::cout << "cudaMalloc: " << 1.0 * iter_time / TEST_REPEAT << " ns" << std::endl;
    iter_time = 0;


    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i=0; i<TEST_REPEAT; i++){
        status = cuFileBufRegister(gpu_buffer[i], IO_SIZE, 0);
        if (status.err != CU_FILE_SUCCESS){
            printf("cuFileBufRegister failed\n");
            exit(1);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    iter_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    std::cout << "cuFileBufRegister: " << 1.0 * iter_time / TEST_REPEAT << " ns" << std::endl;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i=0; i<TEST_REPEAT; i++){
        result = cuFileRead(cf_handle, 
            gpu_buffer[i], IO_SIZE, 
            completed_bytes, 0);
        if (result < 0) {
            printf("cuFileRead failed\n");
            exit(1);
        }
        completed_bytes += result;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    iter_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    std::cout << "cuFileRead: " << 1.0 * iter_time / TEST_REPEAT << " ns" << std::endl;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i=0; i<TEST_REPEAT; i++){
        cuFileBufDeregister(gpu_buffer[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    iter_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    std::cout << "cuFileBufDeregister: " << 1.0 * iter_time / TEST_REPEAT << " ns" << std::endl;
    
    
    cuFileHandleDeregister(cf_handle);

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i=0; i<TEST_REPEAT; i++){
        check_cudaruntimecall(cudaFree(gpu_buffer[i]));
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    iter_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    std::cout << "cudaFree: " << 1.0 * iter_time / TEST_REPEAT << " ns" << std::endl;

    cuFileDriverClose();
}

void loop_phxfs_test(){
    struct timespec start, end;
    unsigned long long completed_bytes = 0;
    unsigned long iter_time = 0;
    int file_fd;
    void **gpu_buffer = new void*[TEST_REPEAT];
    void **target_addr = new void*[TEST_REPEAT];
    int ret;

    check_cudaruntimecall(cudaSetDevice(0));
    ret = phxfs_open(0);

    file_fd = open(file_path, O_RDWR | O_CREAT | O_DIRECT, 0664);

    if (file_fd < 0){
        printf("open file failed\n");
        exit(1);
    }

    for (int i=0; i<TEST_REPEAT; i++){
        check_cudaruntimecall(cudaMalloc(&gpu_buffer[i], IO_SIZE));
        check_cudaruntimecall(cudaMemset(gpu_buffer[i], 0x00, IO_SIZE));
        check_cudaruntimecall(cudaStreamSynchronize(0));
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i=0; i<TEST_REPEAT; i++){
        ret = phxfs_regmem(0, gpu_buffer[i], IO_SIZE, &target_addr[i]);
        if (ret){
            printf("phxfs regmem failed\n");
            exit(1);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    iter_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    std::cout << "phxfs_regmem: " << 1.0 * iter_time / TEST_REPEAT << " ns" << std::endl;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i=0; i<TEST_REPEAT; i++){
        ssize_t result = pread(file_fd, target_addr[i], IO_SIZE, completed_bytes);
        if (result < 0){
            printf("pread failed\n");
            exit(1);
        }
        completed_bytes += result;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    iter_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    std::cout << "pread: " << 1.0 * iter_time / TEST_REPEAT << " ns" << std::endl;

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = TEST_REPEAT - 1; i >= 0; i--){
        ret = phxfs_unregmem(file_fd, gpu_buffer[i], IO_SIZE);
        if (ret){
            printf("phxfs unregmem failed, %d\n", i);
            exit(1);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    iter_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    std::cout << "phxfs_unregmem: " << 1.0 * iter_time / TEST_REPEAT << " ns" << std::endl;

    for (int i=0; i<TEST_REPEAT; i++){
        check_cudaruntimecall(cudaFree(gpu_buffer[i]));
    }

    close(file_fd);
    phxfs_close(0);
}

void muti_file_register(){
    struct timespec start, end;
    CUfileDescr_t cf_descr[20];
    CUfileHandle_t cf_handle[20];
    CUfileError_t status;
    unsigned long total_time = 0;
    int file_fd[20];

    check_cudaruntimecall(cudaSetDevice(0));

    status = cuFileDriverOpen();

    for (int i = 0; i < 20;i++){
        std::string file_predix = "/mnt/phxfs/test/" + std::to_string(i) + ".data";
        file_fd[i] = open(file_path, O_RDWR | O_CREAT | O_DIRECT, 0664);
        if (file_fd[i] < 0){
            printf("open file failed\n");
            exit(1);
        }
    }
    for (int i = 0; i < 10;i++){
        memset(&cf_descr[i], 0, sizeof(CUfileDescr_t));
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < 10;i++){
        cf_descr[i].handle.fd = file_fd[i];
        cf_descr[i].type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        status = cuFileHandleRegister(&cf_handle[i], &cf_descr[i]);
        if (status.err != CU_FILE_SUCCESS){
            printf("cuFileHandleRegister failed\n");
            exit(1);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    total_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    std::cout << "cuFileHandleRegister: " << 1.0 * total_time / 10 << " ns" << std::endl;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < 10;i++){
        cuFileHandleDeregister(cf_handle[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    total_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    std::cout << "cuFileHandleDeregister: " << 1.0 * total_time / 10 << " ns" << std::endl;
}

#define LOOP (1L << 30) * 50
static unsigned long long all_done_bytes = 0;
std::vector<unsigned long long> latency_vec;

void loop_phxfs_once(){
    struct timespec start, end;
    phxfs_xfer_addr_list_t xfer_addr;
    int file_fd;
    int device_id = 0;
    size_t internal_bytes = 0;
    void *gpu_buffer;
    void *target_addr;
    int ret;
    

    check_cudaruntimecall(cudaSetDevice(device_id));
    file_fd = open(file_path, O_RDWR | O_CREAT | O_DIRECT, 0664);
    if (file_fd < 0){
        printf("open file failed\n");
        exit(1);
    }


    check_cudaruntimecall(cudaMalloc(&gpu_buffer, IO_SIZE));
    check_cudaruntimecall(cudaMemset(gpu_buffer, 0xab, IO_SIZE));
    check_cudaruntimecall(cudaStreamSynchronize(0));


    clock_gettime(CLOCK_MONOTONIC, &start);
    ret = phxfs_open(0);
    if (ret){
        printf("phxfs open failed\n");
        exit(1);
    }
    
    ret = phxfs_regmem(file_fd, gpu_buffer, IO_SIZE, &target_addr);

    if (target_addr == NULL){
        printf("phxfs regmem failed\n");
        exit(1);
    }
    phxfs_do_xfer_addr(device_id, (phxfs_io_para_s){
        .buf = gpu_buffer,
        .buf_offset = 0,
        .nbyte = IO_SIZE
    }, &xfer_addr);


    for (int i = 0;i < xfer_addr.nr_xfer_addrs; i++){
        ssize_t result = pread(file_fd, xfer_addr.x_addrs[i].target_addr, xfer_addr.x_addrs[i].nbyte, all_done_bytes);
        if (result < 0){
            printf("pread failed\n");
            exit(1);
        }
        all_done_bytes += result;
        all_done_bytes%=LOOP;
        internal_bytes+=result;
    }
    if (internal_bytes != IO_SIZE){
        printf("pread failed\n");
        exit(1);
    }
    
    ret = phxfs_unregmem(device_id, gpu_buffer, IO_SIZE);
    if (ret){
        printf("phxfs unregmem failed\n");
        exit(1);
    }

    phxfs_close(device_id);
    clock_gettime(CLOCK_MONOTONIC, &end);
    close(file_fd);
    check_cudaruntimecall(cudaFree(gpu_buffer));
    latency_vec.push_back((end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec));
    all_done_bytes += internal_bytes;
    all_done_bytes%=LOOP;
}

void loop_gds_once(){
    struct timespec start, end;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    CUfileError_t status;
    int file_fd;
    void *gpu_buffer = NULL;
    ssize_t result;

    file_fd = open(file_path, O_RDWR | O_CREAT | O_DIRECT, 0664);
    if (file_fd < 0){
        printf("open file failed\n");
        exit(1);
    }
    check_cudaruntimecall(cudaSetDevice(0));

    check_cudaruntimecall(cudaMalloc(&gpu_buffer, IO_SIZE));
    check_cudaruntimecall(cudaMemset(gpu_buffer, 0xab, IO_SIZE));
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
    
    status = cuFileBufRegister(gpu_buffer, IO_SIZE, 0);
    if (status.err != CU_FILE_SUCCESS){
        printf("cuFileBufRegister failed\n");
        exit(1);
    }
    

    result = cuFileRead(cf_handle, gpu_buffer, IO_SIZE, 
            all_done_bytes, 0);
    if (result != (ssize_t)IO_SIZE){
        printf("cuFileRead failed\n");
        exit(1);
    }
    
    cuFileBufDeregister(gpu_buffer);
    cuFileHandleDeregister(cf_handle);
    cuFileDriverClose();
    clock_gettime(CLOCK_MONOTONIC, &end);

    all_done_bytes+=result;
    all_done_bytes%=LOOP;
    latency_vec.push_back((end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec));
    check_cudaruntimecall(cudaFree(gpu_buffer));
    check_cudaruntimecall(cudaStreamSynchronize(0));
}

void get_once_performance(std::vector<unsigned long long> &latency_vec){
    unsigned long long total_time = 0;
    for (size_t i = 0; i < latency_vec.size(); i++){
        total_time += latency_vec[i];
    }
    std::cout << "average latency: " << 1.0 * total_time / latency_vec.size() << " ns" << std::endl;
}

int main(int argc, char *argv[]) {
    TEST_REPEAT = 10;
    int type, once;
    std::cout << std::fixed;
    if (argc != 3){
        printf("Usage: %s <type> <io_size> <once|muti>\n", argv[0]);
        exit(1);
    }

    type = atoi(argv[1]);
    IO_SIZE = atoi(argv[2]);
    once = atoi(argv[3]);
    if (IO_SIZE > 1024){
        TEST_REPEAT = 100;
    }
    IO_SIZE = IO_SIZE * 1024;

    if (once){
        if (type == 0)
            loop_gds_test();
        else
            loop_phxfs_test();
    }else{
        if (type == 0){
            for (int i = 0; i < TEST_REPEAT; i++){
                loop_gds_once();
            }
        }else{
            for (int i = 0; i < TEST_REPEAT; i++){
                loop_phxfs_once();
            }
        }
        get_once_performance(latency_vec);
    }

    return 0;
}