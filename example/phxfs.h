#ifndef __PHXFS_H__
#define __PHXFS_H__
#include <cstddef>
#include <stdint.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

// for register buffer more than 1GB, because the limitation of mmap
struct xfer_addr{
    void *target_addr;
    size_t nbyte;
};

typedef struct phxfs_xfer_addr_list_s{
    int nr_xfer_addrs;
    union{
        struct xfer_addr x_addr;
        struct xfer_addr x_addrs[0];
    };
}phxfs_xfer_addr_list_t;

typedef struct phxfs_io_para_s{
    void *buf;
    off_t buf_offset;
    size_t nbyte;
}phxfs_io_para_t;

int phxfs_do_xfer_addr(int, phxfs_io_para_s, phxfs_xfer_addr_list_t *);


int phxfs_open(int deviceID);
int phxfs_close(int deviceID);
int phxfs_regmem(int deviceID, const void *addr, size_t len, void **target_addr);
int phxfs_unregmem(int deviceID, const void *addr, size_t len);
#endif
