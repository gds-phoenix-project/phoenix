#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <linux/types.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <string>

#include "phoenix.h"

#define HUGE_PAGE_SIZE (64 * 1024)
#define SMALL_PAGE_SIZE (4 * 1024)
#define MMAP_LIMIT (1024 * 1024 * 1024)
#define PHXFS_MAX_DEVICES 8
#define QD 128

typedef struct phxfs_mmap_node_s {
    void **vaddrs;
    size_t mmap_count;
    uint64_t n_addr;
    size_t length;
    size_t length_left; // 最后一次mmap的长度
    bool has_reg;
    struct phxfs_mmap_node_s *next; // 指向下一个节点的指针
} phxfs_p2p_map_t;

typedef struct phxfs_mmap_buffer_s {
    int device_id;
    int bdev_fd;
    bool init_stat;
    struct phxfs_mmap_node_s *head;
    pthread_mutex_t lock;
} phxfs_mmap_buffer_t;

static int g_device_count = PHXFS_MAX_DEVICES;
static phxfs_mmap_buffer_t mbuffer[PHXFS_MAX_DEVICES];
static std::vector<std::string> phxfs_dev_path = {
    "/dev/phxfs_dev0", "/dev/phxfs_dev1",
    "/dev/phxfs_dev2", "/dev/phxfs_dev3",
    "/dev/phxfs_dev4", "/dev/phxfs_dev5",
    "/dev/phxfs_dev6", "/dev/phxfs_dev7"
};

static std::vector<bool> phxfs_initialized(PHXFS_MAX_DEVICES, false);

void free_phxfs_p2p_map(phxfs_mmap_buffer_t *buffer) {
    struct phxfs_mmap_node_s *current = buffer->head;
    struct phxfs_mmap_node_s *next;
    int count = 0;
    pthread_mutex_lock(&buffer->lock);

    while (current) {
        next = current->next;
        // For nodes that are not pinned, manual release is required
        free(current);
        current = next;
        count++;
    }

    buffer->head = NULL;
    pthread_mutex_unlock(&buffer->lock);

}

static int __phxfs_close(phxfs_mmap_buffer_t *mbuffer) {
    if (!mbuffer->init_stat)
        return -1;
    if (mbuffer->bdev_fd > 0)
        close(mbuffer->bdev_fd);

    free_phxfs_p2p_map(mbuffer);
    pthread_mutex_destroy(&mbuffer->lock);
    return 0;
}

static int phxfs_close_all() {
    for (int i = 0; i < g_device_count; i++) {
        if (phxfs_initialized[i]) {
            __phxfs_close(&mbuffer[i]);
            phxfs_initialized[i] = false;
        }
    }
    return 0;
}

int phxfs_close(int device_id) {
    if (device_id >= 0 && device_id < g_device_count) {
        if (phxfs_initialized[device_id]) {
            __phxfs_close(&mbuffer[device_id]);
            phxfs_initialized[device_id] = false;
        }
        return 0;
    }
    return -1;
}

static int __phxfs_open(const char *dev_path, phxfs_mmap_buffer_t *mbuffer) {
    mbuffer->bdev_fd = open(dev_path, O_RDWR);
    
    if (mbuffer->bdev_fd == -1) {
        printf("failed to open file %s\n", dev_path);
        return -1;
    }
    mbuffer->head = NULL;
    pthread_mutex_init(&mbuffer->lock, NULL);
    mbuffer->init_stat = true;
    return 0;
}



bool is_phxfs_initialized() {
    bool initialized = false;
    for (int i = 0; i < g_device_count; i++) {
        initialized = initialized | phxfs_initialized[i];
    }
    return initialized;
}

int phxfs_open(int deviceID) {
    int ret;
    if (!is_phxfs_initialized()) {
        if (deviceID == -1) {
            for (int id = 0; id < g_device_count; ++id) {
                if (!phxfs_initialized[id]) {
                    ret = __phxfs_open(phxfs_dev_path[id].c_str(), &mbuffer[id]);
                    if (ret < 0) {
                        phxfs_close_all();
                        return ret;
                    }
                    phxfs_initialized[id] = true;
                }
            }
        } else if (deviceID >= 0 && deviceID < g_device_count) {
            if (!phxfs_initialized[deviceID]) {
                ret = __phxfs_open(phxfs_dev_path[deviceID].c_str(), &mbuffer[deviceID]);
                phxfs_initialized[deviceID] = true;
                return ret;
            }
        } else { // only start device id = 0
            if (!phxfs_initialized[0]) {
                ret = __phxfs_open(phxfs_dev_path[deviceID].c_str(), &mbuffer[0]);
                phxfs_initialized[0] = true;
                return ret;
            }
        }
    }
    return 0;
}


int insert_phxfs_mmap_node(phxfs_mmap_buffer_t *mbuffer, phxfs_p2p_map_t *new_node) {
    if (!new_node) {
        fprintf(stderr, "%s: new_node is NULL\n", __func__);
        return -1;
    }

    pthread_mutex_lock(&mbuffer->lock);
    new_node->next = mbuffer->head;
    mbuffer->head = new_node;
    pthread_mutex_unlock(&mbuffer->lock);
    return 0;
}

phxfs_p2p_map_t *find_phxfs_mmap_node(phxfs_mmap_buffer_t *mbuffer, u64 n_addr, u64 len) {
    phxfs_p2p_map_t *current = mbuffer->head;
    while (current) {
        if (current->n_addr <= n_addr && ((current->n_addr + current->length) >= (n_addr + len))) {
            return current;
        }
        current = current->next;
    }
    return NULL; // 未找到节点
}

int delete_phxfs_mmap_node(phxfs_mmap_buffer_t *mbuffer, phxfs_p2p_map_t *p2p_map) {
    phxfs_p2p_map_t *current = mbuffer->head;
    phxfs_p2p_map_t *previous = NULL;
    pthread_mutex_lock(&mbuffer->lock);
    while (current) {
        if (current->n_addr == p2p_map->n_addr) {
            if (previous) {
                previous->next = current->next;
            } else {
                mbuffer->head = current->next;
            }
            pthread_mutex_unlock(&mbuffer->lock);
            return 0;
        }
        previous = current;
        current = current->next;
    }
    pthread_mutex_unlock(&mbuffer->lock);
    return -1;
}

static inline int __phxfs_regmem(phxfs_mmap_buffer_t *mbuffer, u64 n_addr, u64 c_addr, size_t len) {
    phxfs_ioctl_para_t para;
    int ret;

    if (!mbuffer->init_stat) {
        return -1;
    }

    para.map_param.n_vaddr = (u64)n_addr;
    para.map_param.c_vaddr = (u64)c_addr;
    para.map_param.c_size = len;
    para.map_param.n_size = len;
    para.map_param.dev.dev_id = mbuffer->device_id;
    ret = ioctl(mbuffer->bdev_fd, PHXFS_IOCTL_MAP, &para);

    return ret;
}

int phxfs_regmem(int device_id, const void *addr, size_t len, void **target_addr) {
    size_t i, ret = 0;
    unsigned long mmaped_len;
    
    phxfs_mmap_buffer_t *pb = &mbuffer[device_id];
    phxfs_p2p_map_t *p2p_map = (phxfs_p2p_map_t *)malloc(sizeof(phxfs_p2p_map_t));
    if (!p2p_map) {
        fprintf(stderr, "%s: new_node is NULL\n", __func__);
        return -1;
    }

    p2p_map->vaddrs = NULL;
    p2p_map->length = len;
    pb->device_id = device_id;

    if (p2p_map->length % HUGE_PAGE_SIZE != 0) {
        fprintf(stderr, "%s: p2p_map->length is not aligned\n", __func__);
        free(p2p_map);
        return -EFAULT;
    }

    p2p_map->mmap_count = p2p_map->length / MMAP_LIMIT;
    if (p2p_map->length % MMAP_LIMIT) {
        p2p_map->mmap_count++;
        p2p_map->length_left = p2p_map->length % MMAP_LIMIT;
    } else {
        p2p_map->length_left = 0;
    }

    p2p_map->vaddrs = (void **)malloc(p2p_map->mmap_count * sizeof(void *));
    if (p2p_map->vaddrs == NULL) {
        fprintf(stderr, "%s: p2p_map->vaddrs malloc fail\n", __func__);
        return -1;
    }

    p2p_map->n_addr = (u64)addr;
    mmaped_len = 0;
    
    for(i = 0; i < p2p_map->mmap_count; i++) {
        p2p_map->vaddrs[i] = NULL;
        if (i == (p2p_map->mmap_count - 1)) {
            // printf("phxfs_regmem 1\n ");
            if (p2p_map->length_left) {
                p2p_map->vaddrs[i] = mmap(p2p_map->vaddrs[i],
                                        p2p_map->length_left,
                                        PROT_READ|PROT_WRITE,
                                        MAP_SHARED,
                                        pb->bdev_fd,
                                        0);
                if ((u64)p2p_map->vaddrs[i] == 0xffffffffffffffff) {
                    fprintf(stderr, "%s: p2p_map->vaddrs mmap fail\n", __func__);
                    return -EFAULT;
                }
                // printf("phxfs_regmem 2\n ");
                ret = __phxfs_regmem(pb, (u64)p2p_map->n_addr + mmaped_len, (u64)p2p_map->vaddrs[i], p2p_map->length_left);
                if (ret) {
                    fprintf(stderr, "%s: p2p_map->vaddrs _phxfs_regmem fail\n", __func__);
                    return -EFAULT;
                }
                mmaped_len += p2p_map->length_left;
                // printf("phxfs_regmem 3\n ");
            } else {
                p2p_map->vaddrs[i] = mmap(p2p_map->vaddrs[i],
                                        MMAP_LIMIT,
                                        PROT_READ|PROT_WRITE,
                                        MAP_SHARED,
                                        pb->bdev_fd,
                                        0);
                if((u64)p2p_map->vaddrs[i] == 0xffffffffffffffff) {
                    fprintf(stderr, "%s: p2p_map->vaddrs mmap fail\n", __func__);
                    return -EFAULT;
                }
                ret = __phxfs_regmem(pb, (u64)p2p_map->n_addr+mmaped_len, (u64)p2p_map->vaddrs[i], MMAP_LIMIT);
                if(ret) {
                    fprintf(stderr, "%s: p2p_map->vaddrs _phxfs_regmem fail\n", __func__);
                    return -EFAULT;
                }
                mmaped_len += MMAP_LIMIT;
            }
        } else {
            p2p_map->vaddrs[i] = mmap(p2p_map->vaddrs[i],
                                    MMAP_LIMIT,
                                    PROT_READ|PROT_WRITE,
                                    MAP_SHARED,
                                    pb->bdev_fd,
                                    0);
            if ((u64)p2p_map->vaddrs[i] == 0xffffffffffffffff) {
                fprintf(stderr, "%s: p2p_map->vaddrs mmap fail\n", __func__);
                return -EFAULT;
            }
            ret = __phxfs_regmem(pb, (u64)p2p_map->n_addr+mmaped_len, (u64)p2p_map->vaddrs[i], MMAP_LIMIT);
            if (ret) {
                fprintf(stderr, "%s: p2p_map->vaddrs _phxfs_regmem fail\n", __func__);
                return -EFAULT;
            }
            mmaped_len += MMAP_LIMIT;
        }
    }
    p2p_map->has_reg = 1;
    insert_phxfs_mmap_node(pb, p2p_map);
    *target_addr = p2p_map->vaddrs[0];

    return 0;
}


int __phxfs_deregmem(phxfs_mmap_buffer_t *pb, u64 n_addr, u64 c_addr, size_t len) {
    phxfs_ioctl_para_t para;
    int ret;

    if(!pb->init_stat) {
        return -1;
    }

    para.map_param.n_vaddr = (u64)n_addr;
    para.map_param.c_vaddr = (u64)c_addr;
    para.map_param.c_size = len;
    para.map_param.n_size = len;
    para.map_param.dev.dev_id = pb->device_id;

    // print_mem(&para, sizeof(phxfs_ioctl_para_t));
    ret = ioctl(pb->bdev_fd, PHXFS_IOCTL_UNMAP, &para);

    return ret;
}

int phxfs_deregmem(int device_id, const void *addr, size_t len) {
    size_t i, ret = 0;
    unsigned long unmapped_len = 0;
    phxfs_p2p_map_t *p2p_map;
    phxfs_mmap_buffer_t *pb = &mbuffer[device_id];

    p2p_map = find_phxfs_mmap_node(pb, (u64)addr, len);
    if(p2p_map == NULL) {
        fprintf(stderr, "%s: p2p_map is not found fail\n", __func__);
        return -1;
    }

    if(p2p_map->length != len || !p2p_map->has_reg) {
        fprintf(stderr, "%s: p2p_map is not match or has not reg\n", __func__);
        return -1;
    }

    ret = delete_phxfs_mmap_node(pb, p2p_map);
    if (ret) {
        fprintf(stderr, "%s: delete_phxfs_mmap_node fail! not found the map\n", __func__);
        return -1;
    }

    unmapped_len = 0;
    for(i = 0; i < p2p_map->mmap_count; i++) {
        if(i == (p2p_map->mmap_count - 1)) {
            if(p2p_map->length_left) {
                ret = __phxfs_deregmem(pb, (u64)p2p_map->n_addr + unmapped_len, (u64)p2p_map->vaddrs[i], p2p_map->length_left);
                if(ret) {
                    fprintf(stderr, "%s: p2p_map->vaddrs _phxfs_unregmem fail\n", __func__);
                    return -1;
                }
                unmapped_len += p2p_map->length_left;
                ret = munmap(p2p_map->vaddrs[i], p2p_map->length_left);
                if(ret) {
                    fprintf(stderr, "%s: p2p_map->vaddrs munmap fail\n", __func__);
                    return -1;
                }
            } else {
                ret = __phxfs_deregmem(pb, (u64)p2p_map->n_addr + unmapped_len, (u64)p2p_map->vaddrs[i], MMAP_LIMIT);
                if(ret) {
                    fprintf(stderr, "%s: p2p_map->vaddrs _phxfs_unregmem fail\n", __func__);
                    return -1;
                }
                unmapped_len += MMAP_LIMIT;
                ret = munmap(p2p_map->vaddrs[i], MMAP_LIMIT);
                if(ret) {
                    fprintf(stderr, "%s: p2p_map->vaddrs munmap fail\n", __func__);
                    return -1;
                }
            }
        } else {
            ret = __phxfs_deregmem(pb, (u64)p2p_map->n_addr + unmapped_len, (u64)p2p_map->vaddrs[i], MMAP_LIMIT);
            if(ret) {
                fprintf(stderr, "%s: p2p_map->vaddrs _phxfs_unregmem fail\n", __func__);
                return -1;
            }
            unmapped_len += MMAP_LIMIT;
            ret = munmap(p2p_map->vaddrs[i], MMAP_LIMIT);
            if(ret) {
                fprintf(stderr, "%s: p2p_map->vaddrs munmap fail\n", __func__);
                return -1;
            }
        }
    }

    free(p2p_map);
    return ret;
}

int phxfs_close(phxfs_fileid_t fid) {
    return close(fid.fd);
}

ssize_t phxfs_read(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset){
    struct phxfs_xfer_addr *xfer_addr;
    ssize_t nbyte_per_iter;
    ssize_t nbyte_total = 0;
    ssize_t ret, i;
    void *target_addr;
    xfer_addr = phxfs_do_xfer_addr(fid.deviceID, buf, buf_offset, nbyte);

    if (xfer_addr == NULL) {
        fprintf(stderr, "%s: phxfs_read pread error\n", __func__);
        return -1;
    }

    for (i = 0;i < xfer_addr->nr_xfer_addrs; i++){
        target_addr = xfer_addr->x_addrs[i].target_addr;
        nbyte_per_iter = xfer_addr->x_addrs[i].nbyte;

        ret = pread(fid.fd, target_addr, nbyte_per_iter, f_offset + nbyte_total);
        if(ret != nbyte_per_iter && ret < 0){
            fprintf(stderr, "%s\n", strerror(errno));
            std::free(xfer_addr);
            return -1;
        }
        nbyte_total += ret;
    }
    std::free(xfer_addr);
    return nbyte_total;
}

// for read/write when registering buffer more than 1G
ssize_t phxfs_write(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset){
    struct phxfs_xfer_addr *xfer_addr;
    size_t nbyte_per_iter;
    size_t nbyte_total = 0;
    uint32_t i;
    void *target_addr;
    int ret;

    xfer_addr = phxfs_do_xfer_addr(fid.deviceID, buf, buf_offset, nbyte);
    if (xfer_addr == NULL) {
        fprintf(stderr, "%s: phxfs_read pread error\n", __func__);
        return -1;
    }

    for (i = 0;i < xfer_addr->nr_xfer_addrs; i++){
        target_addr = xfer_addr->x_addrs[i].target_addr;
        nbyte_per_iter = xfer_addr->x_addrs[i].nbyte;
        ret = pwrite(fid.fd, target_addr, nbyte_per_iter, f_offset + nbyte_total);

        if((size_t)ret != nbyte_per_iter){
            fprintf(stderr, "%s: phxfs_read pread error: ret is %d\n", __func__, ret);
            std::free(xfer_addr);
            return -1;
        }
        nbyte_total += nbyte_per_iter;
    }
    std::free(xfer_addr);
    return nbyte_total;
}

// for read/write when registering buffer more than 1G
struct phxfs_xfer_addr *phxfs_do_xfer_addr(int device_id, const void *buf, off_t buf_offset, size_t nbyte) {
    struct phxfs_xfer_addr *addr;
    phxfs_mmap_buffer_t *_local_mbuffer = &mbuffer[device_id];
    phxfs_p2p_map_t *p2p_map;
    uint32_t start, end, count;
    uint64_t offset_in_page;
    uint64_t nbyte_per_iter;
    uint64_t nbyte_left = nbyte;
    uint32_t i;

    p2p_map = find_phxfs_mmap_node(_local_mbuffer, (u64)buf, nbyte);

    if(!p2p_map) {
        fprintf(stderr, "%s: p2p_map is not found fail\n", __func__);
        return NULL;
    }

    if(!p2p_map->has_reg) {
        fprintf(stderr, "%s: p2p_map is not match or has not reg\n", __func__);
        return NULL;
    }
    
    if((nbyte + buf_offset) > p2p_map->length) {
        fprintf(stderr, "%s: Read/Write out of range 0, nbyte is %lu, buf_offset is %lu, length is %lu\n",
                __func__, nbyte, buf_offset, p2p_map->length);
        return NULL;
    }

    start = buf_offset / MMAP_LIMIT;
    end = (buf_offset + nbyte - 1) / MMAP_LIMIT;
    count = end - start + 1;
 
    addr = (struct phxfs_xfer_addr *)std::malloc(sizeof(struct phxfs_xfer_addr) + (count - 1) * sizeof(struct xfer_addr));
    if (!addr) {
        fprintf(stderr, "%s: addr malloc fail\n", __func__);
        return NULL;
    }

    addr->nr_xfer_addrs = 0;

    if ((start + count) > p2p_map->mmap_count || 
        end > p2p_map->mmap_count || count > p2p_map->mmap_count) {
        fprintf(stderr, "%s: Write out of range 1\n", __func__);
        return NULL;
    }

    for (i = start; i <= end; i++) {
        if(i == start){
            offset_in_page = buf_offset % MMAP_LIMIT;
            if(count > 1)
                nbyte_per_iter = MMAP_LIMIT - offset_in_page;
            else
                nbyte_per_iter = nbyte_left;
            addr->x_addrs[addr->nr_xfer_addrs++] = (struct xfer_addr){
                .target_addr = (void*)((u64)p2p_map->vaddrs[i] + offset_in_page),
                .nbyte = nbyte_per_iter
            };
        }
        else if(i == end){
            addr->x_addrs[addr->nr_xfer_addrs++] = (struct xfer_addr){
                .target_addr = p2p_map->vaddrs[i],
                .nbyte = (buf_offset + nbyte - 1) % MMAP_LIMIT + 1
            };
            nbyte_per_iter = addr->x_addrs[addr->nr_xfer_addrs - 1].nbyte;
        }
        else{
            addr->x_addrs[addr->nr_xfer_addrs++] = (struct xfer_addr){
                .target_addr = p2p_map->vaddrs[i],
                .nbyte = MMAP_LIMIT
            };
            nbyte_per_iter = addr->x_addrs[addr->nr_xfer_addrs - 1].nbyte;
        }
        nbyte_left -= nbyte_per_iter;
    }
    if (nbyte_left != 0 || addr->nr_xfer_addrs != count) {
        fprintf(stderr, "%s: phxfs_write error !\n", __func__);
        return NULL;
    }

    return addr;
}

