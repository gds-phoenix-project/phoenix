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
#include "cufile_sample_utils.h"
#include "phxfs_utils.h"
#include "phxfs.h"

static int TEST_REPEAT = 10000;
const char *file_path = "/mnt/phxfs/test.data";


void register_only(size_t size){
    struct timespec start, end;
    uint64_t total_time = 0;
    CUfileError_t status;
    void **gpu_buffer = new void *[TEST_REPEAT + 10];

    for (int i=0; i<TEST_REPEAT; i++){
        check_cudaruntimecall(cudaMalloc(&gpu_buffer[i], size));
        check_cudaruntimecall(cudaMemset(gpu_buffer[i], 0x00, size));
        check_cudaruntimecall(cudaStreamSynchronize(0));
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i=0; i<TEST_REPEAT; i++){
        status = cuFileBufRegister(gpu_buffer[i], size, 0);
        if (status.err != CU_FILE_SUCCESS){
            printf("cuFileBufRegister failed\n");
            exit(1);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    total_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    printf("register_only: %ld\n", total_time / TEST_REPEAT);

    
    for (int i=0; i<TEST_REPEAT; i++){
        cuFileBufDeregister(gpu_buffer[i]);
        check_cudaruntimecall(cudaFree(gpu_buffer[i]));
    }
    delete [] gpu_buffer;

}

void malloc_only(size_t size){
    struct timespec start, end;
    uint64_t total_time = 0;
    void **gpu_buffer = new void *[TEST_REPEAT];

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i=0; i<TEST_REPEAT; i++){
        check_cudaruntimecall(cudaMalloc(&gpu_buffer[i], size));
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    total_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    printf("malloc_only: %ld\n", total_time / TEST_REPEAT);
    for (int i=0; i<TEST_REPEAT; i++){
        check_cudaruntimecall(cudaFree(gpu_buffer[i]));
    }
}


void driver_open_only(){
    struct timespec start, end;
    unsigned long long total_time = 0;
    for (int i=0; i<TEST_REPEAT; i++){
        printf("repeat %d\n", i);
        clock_gettime(CLOCK_MONOTONIC,&start);
        cuFileDriverOpen();
        clock_gettime(CLOCK_MONOTONIC,&end);
        total_time += (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
        cuFileDriverClose();
    }
    printf("driver_open_only: %lld\n", total_time / TEST_REPEAT);
}

void file_handle_register_only(){
    struct timespec start, end;
    unsigned long long total_time = 0;
    CUfileHandle_t cf_handle;
    CUfileError_t status;
    CUfileDescr_t cf_descr;
    int file_fd;
    file_fd = open(file_path, O_RDWR | O_CREAT | O_DIRECT, 0664);
    if (file_fd < 0){
        printf("open file failed\n");
        exit(1);
    }
    
    for (int i=0; i<TEST_REPEAT; i++){
        memset(&cf_descr, 0, sizeof(CUfileDescr_t));
        cf_descr.handle.fd = file_fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        clock_gettime(CLOCK_MONOTONIC, &start);
        status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS){
            printf("cuFileHandleRegister failed\n");
            exit(1);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_time += (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
        cuFileHandleDeregister(cf_handle);
    }
    printf("file_handle_register_only: %lld\n", total_time / TEST_REPEAT);
}

#define MAX_BATCH_SIZE 128
static int io_size = ( 1 << 10) * 64;
static int batch_size = 16;
static std::vector<unsigned long long> latency_vec;


#define get_time(ts) clock_gettime(CLOCK_MONOTONIC, &ts);
#define get_time_diff(start, end) do{\
    latency_vec.push_back((end.tv_sec - start.tv_sec) * 1000000000LL + end.tv_nsec - start.tv_nsec);\
}while(0)

static std::string batch_op_name[] = {
    "cuFileDriverOpen",
    "cuFileHandleRegister",
    "cudaMalloc",
    "cuFileBufRegister",
    "cuFileBatchIOSetUp",
    "cuFileBatchIOSubmit",
    "cuFileBatchIOGetStatus",
    "cuFileBatchIODestroy",
    "cuFileBufDeregister",
    "cudaFree",
    "cuFileHandleDeregister",
    "cuFileDriverClose"
};

#define batch_time_size 12

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


void batch_io_test(){
    struct timespec start, end;
    CUfileDescr_t cf_descr[MAX_BATCH_SIZE];
    CUfileHandle_t cf_handle[MAX_BATCH_SIZE];
    CUfileIOParams_t io_batch_params[MAX_BATCH_SIZE];
    CUfileIOEvents_t io_batch_events[MAX_BATCH_SIZE];
    CUfileError_t errorBatch;
    CUfileBatchHandle_t batch_id;
    CUfileError_t status;
    void *devPtr[MAX_BATCH_SIZE];
    int fd[MAX_BATCH_SIZE];

    int num_completed = 0;
    unsigned nr = 0;

    check_cudaruntimecall(cudaSetDevice(0));

    get_time(start);
    status = cuFileDriverOpen();
    get_time(end);
    get_time_diff(start, end);

    if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "cufile driver open error: "
        << cuFileGetErrorString(status) << std::endl;
            return;
    }
    

    for (int i=0;i<batch_size;i++){
        fd[i] = open(file_path, O_CREAT | O_RDWR | O_DIRECT, 0664);
        if (fd[i] < 0) {
			std::cerr << "file open error:"
			<< cuFileGetErrorString(errno) << std::endl;
		}
    }
    memset((void *)cf_descr, 0, MAX_BATCH_SIZE * sizeof(CUfileDescr_t));

    get_time(start);
    for (int i=0;i<batch_size;i++){
        cf_descr[i].handle.fd = fd[i];
        cf_descr[i].type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        status = cuFileHandleRegister(&cf_handle[i], &cf_descr[i]);
        if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "file register error:"
            << cuFileGetErrorString(status) << std::endl;
            close(fd[i]);
        }
    }
    get_time(end);
    get_time_diff(start, end);

    get_time(start);
    for(int i = 0; i < batch_size; i++) {
		devPtr[i] = NULL; 
		check_cudaruntimecall(cudaMalloc(&devPtr[i], io_size));
	}
    get_time(end);
    get_time_diff(start, end);

    for (int i=0;i<batch_size;i++){
        check_cudaruntimecall(cudaMemset((void*)(devPtr[i]), 0xab, io_size));
		check_cudaruntimecall(cudaStreamSynchronize(0));	
    }


    get_time(start);
    for(int i = 0; i < batch_size; i++) {
		status = cuFileBufRegister(devPtr[i], io_size, 0);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "buffer register failed:"
				<< cuFileGetErrorString(status) << std::endl;
			goto out2;
		}
	}
    get_time(end);
    get_time_diff(start, end);

    
    for(int i = 0; i < batch_size; i++) {
		io_batch_params[i].mode = CUFILE_BATCH;
		io_batch_params[i].fh = cf_handle[i];
		io_batch_params[i].u.batch.devPtr_base = devPtr[i];
		io_batch_params[i].u.batch.file_offset = i * io_size;
		io_batch_params[i].u.batch.devPtr_offset = 0;
		io_batch_params[i].u.batch.size = io_size;
		io_batch_params[i].opcode = CUFILE_READ;
	}

    get_time(start);
    errorBatch = cuFileBatchIOSetUp(&batch_id, batch_size);
    	if(errorBatch.err != 0) {
		std::cerr << "Error in setting Up Batch" << std::endl;
		goto out3;
	}
    get_time(end);
    get_time_diff(start, end);

    get_time(start);
    errorBatch = cuFileBatchIOSubmit(batch_id, batch_size, io_batch_params, 0);
    get_time(end);
    get_time_diff(start, end);	
	if(errorBatch.err != 0) {
		std::cerr << "Error in IO Batch Submit" << std::endl;
		goto out3;
	}

    get_time(start);
    while(num_completed != batch_size) {
		memset(io_batch_events, 0, sizeof(*io_batch_events));
		nr = batch_size;
		errorBatch = cuFileBatchIOGetStatus(batch_id, batch_size, &nr, io_batch_events, NULL);	
		if(errorBatch.err != 0) {
			std::cerr << "Error in IO Batch Get Status" << std::endl;
			goto out4;
		}
		num_completed += nr;
	}
    get_time(end);
    get_time_diff(start, end);

out4:
    get_time(start);
	cuFileBatchIODestroy(batch_id);
    get_time(end);
    get_time_diff(start, end);

out3:
    get_time(start);
    for(int i = 0; i < batch_size; i++) {
		status = cuFileBufDeregister(devPtr[i]);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "buffer deregister failed:"
				<< cuFileGetErrorString(status) << std::endl;
		}
	}
    get_time(end);
    get_time_diff(start, end);
out2:
    get_time(start);
	for(int i = 0; i < batch_size; i++) {
		check_cudaruntimecall(cudaFree(devPtr[i]));
	}
    get_time(end);
    get_time_diff(start, end);

	get_time(start);
	for(int i = 0; i < batch_size; i++) {
		if (fd[i] > 0) {
			cuFileHandleDeregister(cf_handle[i]);
		}
	}
    get_time(end);
    get_time_diff(start, end);

    for (int i=0;i<batch_size;i++){
        close(fd[i]);
    }
	
    get_time(start);
    status = cuFileDriverClose();
    get_time(end);
    get_time_diff(start, end);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cufile driver close failed:"
			<< cuFileGetErrorString(status) << std::endl;
	}
}

void sync_io_test(){
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

    file_fd = open(file_path, O_RDWR | O_CREAT | O_DIRECT, 0664);
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
        std::cout << "i: " << i << std::endl;
        func();
    }

    std::cout << latency_vec.size() << std::endl;
    repeat = latency_vec.size() / op_cnt;
    std::cout << "repeat: " << repeat << std::endl;
    for (unsigned long int  i = 0;i < (latency_vec.size() / op_cnt); i++){
        std::cout << "i: " << i << std::endl;
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
    "phxfs_unregmem",
    "phxfs_uninit"
};

#define phxfs_op_size 5

void phxfs_io_test(){
    struct timespec start, end;
    int ret, file_fd;
    void *gpu_buffer;
    void *target_addr = NULL;

    check_cudaruntimecall(cudaSetDevice(0));

    file_fd = open(file_path,  O_CREAT | O_RDWR | O_DIRECT, 0644);

    if (file_fd < 0) {
        perror("Open file error");
        return;
    }
    get_time(start);
    ret = phxfs_open(0);
    get_time(end);
    get_time_diff(start, end);

    if (ret != 0) {
        pr_error("phxfs init failed: " << ret);
        return;
    }

    check_cudaruntimecall(cudaMalloc(&gpu_buffer, io_size));
    check_cudaruntimecall(cudaMemset(gpu_buffer, 0x00, io_size));
    check_cudaruntimecall(cudaStreamSynchronize(0));

    get_time(start);
    ret = phxfs_regmem(0, gpu_buffer, io_size, &target_addr);
    get_time(end);
    get_time_diff(start, end);

    if (ret){
        pr_error("phxfs regmem failed: " << ret);
        return;
    }

    get_time(start);
    ret = pread(file_fd, target_addr, io_size, 0);
    get_time(end);
    get_time_diff(start, end);

    if (ret < 0){
        perror("Read file error");
        return;
    }

    get_time(start);
    ret = phxfs_unregmem(0, gpu_buffer, io_size);
    get_time(end);
    get_time_diff(start, end);

    if (ret){
        pr_error("phxfs unregmem failed: " << ret);
        return;
    }

    check_cudaruntimecall(cudaFree(gpu_buffer));

    get_time(start);
    phxfs_close(0);
    get_time(end);
    get_time_diff(start, end);

    close(file_fd);
}

int main(int argc, char *argv[]) {
    int type = 0;
    if (argc > 2){
        type = atoi(argv[1]);
        TEST_REPEAT = atoi(argv[2]);
    }
    if (type == 0)
        get_breakdown(sync_io_test, sync_op_name, sync_time_size);
    else
        get_breakdown(phxfs_io_test, phxfs_op_name, phxfs_op_size);
    return 0;
}