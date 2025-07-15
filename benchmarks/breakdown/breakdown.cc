#include <cstddef>
#include <cstdint>
#include <cstdio>
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

#include "phxfs_utils.h"
#include "phoenix.h"
#include "cufile_sample_utils.h"

static int TEST_REPEAT = 1000;
static std::string file_path = "/mnt/phxfs1/6";
static uint64_t io_size = 4ull * 1024 * 1024 * 1024;
static std::vector<unsigned long long> latency_vec;


#define get_time(ts) clock_gettime(CLOCK_MONOTONIC, &ts);
#define get_time_diff(start, end) do{\
    latency_vec.push_back((end.tv_sec - start.tv_sec) * 1000000000LL + end.tv_nsec - start.tv_nsec);\
}while(0)

static std::string sync_op_name[] = {
    "cuFileDriverOpen",
    "cuFileHandleRegister",
    "cudaMalloc",
    "cuFileBufRegister",
    "cuFileRead",
    "cuFileBufDeregister",
    "cuFileHandleDeregister",
    "cudaFree",
    "cuFileDriverClose"
};
#define sync_time_size 9


void gds_io_test(){
    struct timespec start, end;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    CUfileError_t status;
    int file_fd;
    void *gpu_buffer;
    ssize_t result;

    check_cudaruntimecall(cudaSetDevice(0));

    get_time(start);
    status = cuFileDriverOpen();
    get_time(end);
    get_time_diff(start, end);

    if (status.err != CU_FILE_SUCCESS){
        printf("cuFileDriverOpen failed\n");
        exit(1);
    }

    file_fd = open(file_path.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0664);
    if (file_fd < 0){
        printf("open file failed\n");
        exit(1);
    }
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = file_fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    get_time(start);
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    get_time(end);
    get_time_diff(start, end);

    if (status.err != CU_FILE_SUCCESS){
        printf("cuFileHandleRegister failed\n");
        exit(1);
    }
    gpu_buffer = NULL;
    get_time(start);
    check_cudaruntimecall(cudaMalloc(&gpu_buffer, io_size));
    get_time(end);
    get_time_diff(start, end);

    check_cudaruntimecall(cudaMemset(gpu_buffer, 0x00, io_size));
    check_cudaruntimecall(cudaStreamSynchronize(0));

    get_time(start);
    status = cuFileBufRegister(gpu_buffer, io_size, 0);
    get_time(end);
    get_time_diff(start, end);

    if (status.err != CU_FILE_SUCCESS){
        printf("cuFileBufRegister failed\n");
        exit(1);
    }

    get_time(start);
    result = cuFileRead(cf_handle, gpu_buffer, io_size, 0, 0);
    get_time(end);
    get_time_diff(start, end);

    if (result < 0) {
        printf("cuFileRead failed\n");
        exit(1);
    }


    get_time(start);
    cuFileBufDeregister(gpu_buffer);
    get_time(end);
    get_time_diff(start, end);

    get_time(start);
    cuFileHandleDeregister(cf_handle);
    get_time(end);
    get_time_diff(start, end);

    get_time(start);
    check_cudaruntimecall(cudaFree(gpu_buffer));
    get_time(end);
    get_time_diff(start, end);

    get_time(start);
    cuFileDriverClose();
    get_time(end);
    get_time_diff(start, end);

}

void get_breakdown(void (*func)(), const std::string* op_name, long int op_cnt){
    unsigned long long *times;
    int repeat = 0;

    times = new unsigned long long[op_cnt];
    latency_vec.clear();
    latency_vec.reserve(TEST_REPEAT * op_cnt);
    std::cout << "Start to get breakdown" << std::endl;
    for (int i = 0; i < TEST_REPEAT; i++){
        func();
    }

    std::cout << latency_vec.size() << std::endl;
    repeat = latency_vec.size() / op_cnt;
    std::cout << "repeat: " << repeat << std::endl;
    for (unsigned long int  i = 0;i < (latency_vec.size() / op_cnt); i++){
        for (int j = 0; j < op_cnt; j++){
            times[j] += latency_vec[i * op_cnt + j];
        }
    }
    for (int i = 0; i < op_cnt; i++){
        std::cout << op_name[i] << ": " << times[i] / TEST_REPEAT << " ns" << std::endl;
    }
}

static std::string phxfs_op_name[] = {
    "phxfs_open",
    "phxfs_regmem",
    "pread",
    "phxfs_deregmem",
    "phxfs_close"
};

#define phxfs_op_size 5

void phxfs_io_test(){
    struct timespec start, end;
    phxfs_fileid_t fid;
    int ret, file_fd, device_id;
    uint64_t real_size = io_size;
    void *gpu_buffer;
    void *target_addr = NULL;

    device_id = 0;

    check_cudaruntimecall(cudaSetDevice(device_id));

    file_fd = open(file_path.c_str(),  O_CREAT | O_RDWR | O_DIRECT, 0644);

    if (file_fd < 0) {
        perror("Open file error");
        return;
    }
    get_time(start);
    ret = phxfs_open(device_id);
    get_time(end);
    get_time_diff(start, end);

    if (ret != 0) {
        pr_error("phxfs init failed: " << ret);
        return;
    }

    fid.fd = file_fd;
    fid.deviceID = device_id;

    if (io_size < 64 * 1024) {
        real_size = 64 * 1024;
    }

    check_cudaruntimecall(cudaMalloc(&gpu_buffer, real_size));
    check_cudaruntimecall(cudaMemset(gpu_buffer, 0x00, real_size));
    check_cudaruntimecall(cudaStreamSynchronize(0));

    get_time(start);
    ret = phxfs_regmem(fid.deviceID, gpu_buffer, real_size, &target_addr);
    get_time(end);
    get_time_diff(start, end);

    if (ret){
        pr_error("phxfs regmem failed: " << ret);
        return;
    }

    get_time(start);
    ssize_t result = phxfs_read(fid, gpu_buffer, 0, io_size, 0);
    get_time(end);
    get_time_diff(start, end);

    if (result < 0){
        perror("Read file error");
        return;
    }

    get_time(start);
    ret = phxfs_deregmem(fid.deviceID, gpu_buffer, real_size);
    get_time(end);
    get_time_diff(start, end);

    if (ret){
        pr_error("phxfs unregmem failed");
        return;
    }

    check_cudaruntimecall(cudaFree(gpu_buffer));

    get_time(start);
    phxfs_close(fid.deviceID);
    get_time(end);
    get_time_diff(start, end);

    close(file_fd);
}

int main(int argc, char *argv[]) {
    int type = 0;
    if (argc > 3){
        file_path = argv[1];
        type = atoi(argv[2]);
        io_size = atoll(argv[3]);
        TEST_REPEAT = atoi(argv[4]);
    } else {
        std::cout << "Usage: " << argv[0] << " <file_path> <type> <io_size> <repeat>" << std::endl;
        return -1;
    }
    io_size *= 1024;
    std::cout << file_path<< std::endl;
    std::cout << "io size" << io_size;
    if (type == 1)
        get_breakdown(gds_io_test, sync_op_name, sync_time_size);
    else
        get_breakdown(phxfs_io_test, phxfs_op_name, phxfs_op_size);
    return 0;
}