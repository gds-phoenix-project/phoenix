#ifndef __PHOENIX_H__
#define __PHOENIX_H__

#include <linux/types.h>
#include <linux/blk-mq.h>
#include <linux/nvme.h>
#include <linux/memremap.h>
#include <linux/genalloc.h>
#include <linux/cdev.h>
#include <linux/pci.h>

#define MAX_DEV_NUM 16

struct pci_p2pdma {
    struct gen_pool *pool;
    bool p2pmem_published;
    struct xarray map_types;
};

struct pci_p2pdma_pagemap {
    struct dev_pagemap pgmap;
    struct pci_dev provider;
    u64 bus_offset;
};

struct phxfs_dev {
    struct pci_dev *dev; /*pci device */
    int domain;
    unsigned int bus;
    unsigned int devfn;
    u64 size; /* HBM pci bar 4 size */
    u64 paddr; /* HBM bus address space addr */
    struct device device; /* char device. */
    struct cdev cdev;
    int idx;
    struct pci_p2pdma_pagemap *p2p_pgmap; /* struct dev_pagemap pgmap; */
    void *dev_remap_addr;
    void __iomem *pci_mem_va; /* get from devm_memremap_pages */
    bool remap;
    unsigned int dev_page_size;
};

struct phxfs_ctrl {
    struct phxfs_dev phx_dev[MAX_DEV_NUM];
    int dev_num;
};

struct find_info {
    void __iomem *start;
    char *target;
    u64 len;
    u64 result;
    u64 offset;
    int thread_id;
    bool found;
};

struct phxfs_dev_info_s {
    u64 dev_id;
} __attribute__((packed, aligned(8)));
typedef struct phxfs_dev_info_s phx_dev_info_t;

struct phxfs_ioctl_map_s {
    struct phxfs_dev_info_s dev;
    u64 c_vaddr;
    u64 c_size;
    u64 n_vaddr;
    u64 n_size;
    u64 end_addr;
    u32 sbuf_block;
} __attribute__((packed, aligned(8)));
typedef struct phxfs_ioctl_map_s phxfs_ioctl_map_t;

struct phxfs_ioctl_io_s {
    u64 cpuvaddr; /* cpu vaddr */
    loff_t offset; /* file offset */
    u64 size; /* Read/Write length */
    u64 end_fence_value; /* End fence value for DMA completion */
    s64 ioctl_return;
    int fd; /* File descriptor */
} __attribute__((packed, aligned(8)));
typedef struct phxfs_ioctl_io_s phxfs_ioctl_io_t;

struct phxfs_ioctl_ret_s {
    s64 ret;
    u8 padding[40];
} __attribute__((packed, aligned(8)));
typedef struct phxfs_ioctl_ret_s phxfs_ioctl_ret_t;

union phxfs_ioctl_para_s {
    struct phxfs_ioctl_map_s map_param;
    struct phxfs_ioctl_io_s io_para;
    struct phxfs_ioctl_ret_s ret;
} __attribute__((packed, aligned(8)));
typedef union phxfs_ioctl_para_s phxfs_ioctl_para_t;


#define PHXFS_IOCTL 0x88 /* 0x4c */
#define PHXFS_IOCTL_MAP _IOW(PHXFS_IOCTL, 1, struct phxfs_ioctl_map_s)
#define PHXFS_IOCTL_UNMAP _IOW(PHXFS_IOCTL, 2, struct phxfs_ioctl_map_s)

void phxfs_map_dev_release(phxfs_ioctl_map_t *map_param, u64 devaddr, u64 dev_len, u64 cpuvaddr, u64 length);

struct devmm_svm_process_id {
    int32_t hostpid;
    union {
        uint16_t devid;
        uint16_t vm_id;
    };
    uint16_t vfid;
};

#endif