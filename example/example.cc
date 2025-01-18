#include <sys/types.h>
#include <unistd.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <fcntl.h>
#include "phxfs.h"


const char *file_path = "/mnt/phxfs/test.data";
static int device_id = 0;
static size_t io_size = 64 * (1 << 10); // 64KB

int main() {
    phxfs_xfer_addr_list_t xfer_addr_list;
    void *gpu_buffer, *target_addr;
    int ret, i;
    int file_fd;
    ssize_t result; 

    file_fd = open(file_path, O_CREAT | O_RDWR | O_DIRECT, 0644);


    ret = phxfs_open(device_id);

    if (ret != 0) {
        printf("phxfs init failed: %d\n", ret);
        return 1;
    }

    cudaMalloc(&gpu_buffer, io_size);
    cudaMemset(gpu_buffer, 0x00, io_size);
    cudaStreamSynchronize(0);

    // target_addr for register buffer less than 1GB
    ret = phxfs_regmem(device_id, gpu_buffer, io_size, &target_addr);

    if (ret) {
        printf("phxfs regmem failed: %d\n", ret);
        return 1;
    }

    result = pread(file_fd, target_addr, io_size, 0);

    if (result < 0) {
        perror("Read file error");
        return 1;
    }

    ret = phxfs_unregmem(device_id, gpu_buffer, io_size);

    if (ret) {
        printf("phxfs unregmem failed: %d\n", ret);
        return 1;
    }

    cudaFree(gpu_buffer);

    phxfs_close(device_id);

    close(file_fd);
    return 0;
}