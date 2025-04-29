#ifndef __PHXFS_MEM_H__
#define __PHXFS_MEM_H__

#include <linux/types.h>
#include <linux/mm.h>
#include <linux/file.h>
#include "phxfs.h" 

#define PAGE_SHIFT 12
#define PHXFS_MIN_BASE_INDEX ((unsigned long)1L<<32)
#define PHXFS_MAX_SHADOW_PAGES 4096
#define PHXFS_MAX_SHADOW_ALLOCS_ORDER 12

struct phxfs_mem_find_info {
    u64 devaddr;
    u64 cpuvaddr;
    u64 len;
    bool found;
};

struct phxfs_mmap_buffer {
    atomic_t ref;
    struct hlist_node hash_link;
    u64 c_vaddr; // mmap cpu vaddr
    u64 map_len; // mmap len
    u64 n_vaddr; // allocated by cann api
    u64 dev_len; // reg dev len
    unsigned long base_index;
    unsigned long dev_id; // npu id
    u64 *dev_page_addrs; // dev page io addr list
    unsigned long dev_page_num; // indicate num of dev_page_addrs
    unsigned long subpage_num; // (dev_page_size / PAGE_SIZE)
    struct page **ppages; // host page which are mapped to dev page
    unsigned long host_page_num; // the corresponding host page num to dev page num, equal to dev_page_num
    struct vm_area_struct *vma;
    struct phxfs_dev *dev;
    bool remap; // if vma remap_pfn_range set true, otherwise false
    struct p2p_vmap* map;
};
typedef struct phxfs_mmap_buffer* phxfs_mmap_buffer_t;


typedef vm_fault_t phxfs_vma_fault_t;


int phxfs_map_dev_addr_inner(phxfs_mmap_buffer_t pbuffer, u64 devaddr, u64 dev_len);
int phxfs_map_dev_addr(phxfs_ioctl_map_t *map_param, u64 devaddr, u64 dev_len, u64 cpuvaddr, u64 length);
int phxfs_mmap(struct file *filp, struct vm_area_struct *vma);
void phxfs_mbuffer_init(void);
void phxfs_mbuffer_put(phxfs_mmap_buffer_t pbuffer);
phxfs_mmap_buffer_t phxfs_mbuffer_get(unsigned long base_index);

struct vmnga_pci_dev_info {
    u8 bus_no;
    u8 device_no;
    u8 function_no;
};

struct vmnga_pcie_id_info {
    unsigned int venderid;
    unsigned int subvenderid;
    unsigned int deviceid;
    unsigned int subdeviceid;
    unsigned int bus;
    unsigned int device;
    unsigned int fn;
};

// addr info: alloc bar4 to external modules.

enum vmng_get_addr_type {
    VMNG_GET_ADDR_TYPE_TSDRV = 0,
    VMNG_GET_ADDR_TYPE_MAX
    // ... other types can be defined here
};

#endif