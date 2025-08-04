#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <liburing.h>
#include <sys/types.h>
#include <unistd.h>

#include "phoenix.h"


enum phxfs_op {
    PHXFS_OP_READ = 0,
    PHXFS_OP_WRITE = 1,
};

struct phxfs_data{
    int fd, op;
    struct phxfs_xfer_addr *xfer_addr;
    off_t file_offset;
    ssize_t *bytes_done;
};

void CUDART_CB phxfs_callback(void *user_data){
    auto* data = static_cast<phxfs_data*>(user_data);
    *data->bytes_done = 0;
    ssize_t file_offset = data->file_offset;
    for (uint32_t i = 0; i < data->xfer_addr->nr_xfer_addrs; i++) {
        auto xfer_addr = data->xfer_addr->x_addrs[i];
        if (data->op == PHXFS_OP_READ) {
            *data->bytes_done += pread(data->fd, xfer_addr.target_addr, 
                                xfer_addr.nbyte, file_offset);
        } else {
            *data->bytes_done += pwrite(data->fd, xfer_addr.target_addr, 
                                     xfer_addr.nbyte, file_offset);
        }
        file_offset += xfer_addr.nbyte;
    }
}
  

cudaError_t phxfs_async(phxfs_fileid_t fid, enum phxfs_op op,
                            void* buf,
                            size_t nbytes, off_t offset,
                            ssize_t *bytes_done,
                            CUstream stream){

    struct phxfs_xfer_addr* addrs = phxfs_do_xfer_addr(fid.deviceID, buf, 0, nbytes);
    if (!addrs) return cudaErrorHostMemoryNotRegistered;
    auto* data = new phxfs_data{
        .fd = fid.fd, .op = op,
        .xfer_addr = addrs, .file_offset = offset,
        .bytes_done = bytes_done
    };

    return cudaLaunchHostFunc(stream, phxfs_callback, data);
}

cudaError_t phxfs_read_async(phxfs_fileid_t fid,
                            void* buf,
                            size_t nbytes, off_t offset,
                            ssize_t *bytes_done,
                            CUstream stream) {
    return phxfs_async(fid, PHXFS_OP_READ, buf, nbytes, offset, bytes_done, stream);
}

cudaError_t phxfs_write_async(phxfs_fileid_t fid,
                             void* buf, 
                             size_t nbytes, off_t offset,
                             ssize_t* bytes_done,
                             CUstream stream) {
    return phxfs_async(fid, PHXFS_OP_WRITE, buf, nbytes, offset, bytes_done, stream);
}