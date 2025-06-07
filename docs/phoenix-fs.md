# Phoenix-fs
The kernel module performs initialization and uninitialization operations when inserted and removed. We detail the initialization and uninitialization processes below, along with the character device interface provided by Phoenix-fs.
## 1. Kernel Module
### Initialization
```c
/** 
 * @file phxfs.c
 * @brief Phoenix-fs kernel module initialization. It will use the memory service provided by the ZONE_DEVICE to remap the GPU device's PCIe BAR memory to the kernel space, and create a character device for each GPU device.
 */
static int __init phxfs_init(void) {
    int ret, i;

    // get nvidia_p2p symbols
	if (nvfs_nvidia_p2p_init()) {
		printk("Could not load nvidia_p2p* symbols\n");
		ret = -EOPNOTSUPP;
		return -1;
	}

    // Initialize the GPU information table
	nvfs_fill_gpu2peer_distance_table_once();
	npu_num = 0;
	for (i = 0; i < MAX_DEV_NUM; i++) {
		if (gpu_info_table[i] != 0) {
			npu_num++;
		} else {
			break;
		}
	}

	if (npu_num <= 0 || npu_num > MAX_DEV_NUM) {
		printk("devdrv_get_devnum error:%u\n", npu_num);
		return -1;
	}
    // obtain the PCIe BAR information of each GPU device via the PCIe bus 
    // and remap the GPU device's BAR memory to the kernel space.
	ret = phxfs_ctrl_init(&ctrl, npu_num);
	if (ret != 0) {
		printk("npu_ctrl_init error:%d\n", ret);
		return -1;
	}

    // create a phoenix-fs character device for each GPU device
	ret = phxfs_cdev_init(&ctrl);
	if (ret) {
		printk("phxfs_init error!\n");
		return -1;
	}

    // initialize the hash table to store the registered GPU memory regions
	phxfs_mbuffer_init();
	return 0;
}

/**
 * @brief Initialize the Phoenix-fs control structure and remap the GPU device's BAR memory to the kernel space.
 * @param dev_ctrl: Pointer to the Phoenix-fs control structure.
 * @param dev_num: Number of GPU devices.
 * @return On success, 0 is returned.
 *         On failure, a negative error code is returned.
 */
*/
static int phxfs_ctrl_init(struct phxfs_ctrl *dev_ctrl, u32 dev_num) {
	
	// get the PCIe BAR information of each GPU device
    dev_ctrl->dev_num = dev_num;
    for (int i = 0; i < dev_num; i++) {
        // get the PCIe BAR information of each GPU device
        ...
		dev_ctrl->phx_dev[i].dev = pci_get_domain_bus_and_slot(0, bus, fn);
        // get the maximum BAR size for each GPU device, which is the size of the GPU memory
        dev_ctrl->phx_dev[i].paddr = pci_resource_start(dev_ctrl->phx_dev[i].dev, max_bar_idx);
        dev_ctrl->phx_dev[i].size = pci_resource_len(dev_ctrl->phx_dev[i].dev, max_bar_idx);
        
        // remap the GPU device's BAR memory to the kernel space
        ret = phxfs_devm_memremap(&dev_ctrl->phx_dev[i]);
		if (ret)
			return ret;
    }
	return 0;
}

```

### Uninitialization
```c
/**
 * @file phxfs.c
 * @brief Phoenix-fs kernel module uninitialization. It will delete the character devices created during initialization, and unmap the remapped GPU device's BAR memory from the kernel space.
 */
static void __exit phxfs_exit(void) {
	int i;
    // delete the character devices created during initialization
	for (i = 0; i < ctrl.dev_num; i++) {
		phxfs_cdev_del(&ctrl.phx_dev[i].cdev, &ctrl.phx_dev[i].device, &ctrl.phx_dev[i]);
	}

    // delete nvidia_p2p symbols
	nvfs_nvidia_p2p_exit();
    // destroy phxfs character device class
	class_destroy(phxfs_chr_class);
    // unregister the character device region
	unregister_chrdev_region(phxfs_chr_devt, PHXFS_MINORS);
}

/**
 * @brief Delete the Phoenix-fs character device and unmap the remapped GPU device's BAR memory from the kernel space.
 * @param cdev: Pointer to the character device structure.
 * @param cdev_device: Pointer to the device structure associated with the character device.
 * @param dev: Pointer to the Phoenix-fs device structure.
 */
void phxfs_cdev_del(struct cdev *cdev, struct device *cdev_device,
                    struct phxfs_dev *dev) {
	cdev_device_del(cdev, cdev_device);
	// unmap the remapped GPU device's BAR memory from the kernel space
    if (dev->remap) {
		devm_memunmap_pages(&dev->dev->dev, &dev->p2p_pgmap->pgmap);
		dev->pci_mem_va = NULL;
	}
	if (dev->p2p_pgmap != NULL) {
		devm_kfree(&dev->dev->dev, &dev->p2p_pgmap->pgmap);
	}
	dev->dev = NULL;
}
```

## 2. Character Device Interface
Phoenix-fs provides a character device interface that allows user-space applications to interact with the kernel module. The character device includes the following three basic operations:

### open
```c
/**
 * @file phxfs.c
 * @brief Phoenix-fs character device open operation. It will save the device metadata in the file structure.
 */
static int phxfs_open(struct inode *inode, struct file *filp) {
    // save the device metadata in the file structure
    filp->private_data = &ctrl.phx_dev[dev_idx];
	return 0；
}
```

### mmap
```c
/**
 * @file phxfs-mem.c
 * @brief Phoenix-fs character device mmap operation. It will set the vma flags and save the vma into the hash table.
 * @param filp: Pointer to the device file structure.
 * @param vma: Pointer to the virtual memory area structure.
 * @return On success, 0 is returned.
 *         On failure, a negative error code is returned.
 */
int phxfs_mmap(struct file *filp, struct vm_area_struct *vma) {
    int ret;
    struct mm_struct *mm = current->mm;

    // set the vma flags for the memory mapping
    vma->vm_flags &= ~VM_PFNMAP;    // this region is not a direct physical page frame mapping
    vma->vm_flags &= ~VM_IO;        // not used for device IO memory
    vma->vm_flags |= VM_MIXEDMAP;   // allow both anonymous and page-frame mappings
    vma->vm_flags |= mm->def_flags; 

    vma->vm_pgoff = 0;
    // set the page protection to non-cached
    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

    if (vma->vm_pgoff == 0) {
        // save the vma into the hash table
        ret = phxfs_add_phony_buffer(filp, vma);
        return ret;
    }

    return -EINVAL;
}
```
### ioctl

```c
/**
 * @file phxfs.c
 * @brief Phoenix-fs character device ioctl operation. It will handle the IOCTL commands for mapping and unmapping device addresses.
 * @param filp: Pointer to the device file structure.
 * @param cmd: The IOCTL command.
 * @param arg: The argument for the IOCTL command.
 * @return On success, 0 is returned.
 *         On failure, a negative error code is returned.
 */

static long phxfs_ioctl(struct file *filp, unsigned int cmd,
                        unsigned long arg) {
	void __user *argp = (void *)arg;
	switch (cmd) {
        //  map a device address to a user-space virtual address
		case PHXFS_IOCTL_MAP: {
			struct phxfs_ioctl_map_s map_param;
			if (copy_from_user(&map_param, argp, sizeof(struct phxfs_ioctl_map_s)))
				return -EFAULT;
			return phxfs_map_dev_addr(&map_param, map_param.n_vaddr, map_param.n_size,
									map_param.c_vaddr, map_param.c_size);
		}
        // unmap and clean up the device address mapping
		case PHXFS_IOCTL_UNMAP: {
			struct phxfs_ioctl_map_s map_param;
			if (copy_from_user(&map_param, argp, sizeof(struct phxfs_ioctl_map_s)))
				return -EFAULT;
			phxfs_map_dev_release(&map_param, map_param.n_vaddr, map_param.n_size,
								map_param.c_vaddr, map_param.c_size);
			return 0;
		}
		default:
			return -ENOTTY;
	}
}

/**
 * @brief Map a device address to a user-space virtual address.
 * @param map_param: Pointer to the mapping parameters structure.
 * @param devaddr: The device address to be mapped.
 * @param dev_len: The length of the device memory region.
 * @param cpuvaddr: The user-space virtual address to map to.
 * @param length: The length of the user-space virtual address region.
 * @return On success, 0 is returned.
 *         On failure, a negative error code is returned.
 */

int phxfs_map_dev_addr(phxfs_ioctl_map_t *map_param, u64 devaddr, u64 dev_len, u64 cpuvaddr, u64 length) {
    int ret = -EINVAL;
    phxfs_mmap_buffer_t mbuffer;
    
    // check and bind the mmap buffer
    mbuffer = phxfs_check_and_bind_mmap_buffer(cpuvaddr, length);
    if (mbuffer == NULL || mbuffer->vma == NULL || devaddr <= length) {
        return ret;
    } else {
        ret = 0;
        mbuffer->n_vaddr = devaddr;
        mbuffer->dev_len = dev_len;
        // map the GPU virtual address to the virtual address space created by mmap
        ret = phxfs_map_dev_addr_inner(mbuffer, devaddr, dev_len);
        return ret;
    }
}

/**
 * @brief Map a device address to a user-space virtual address. It will get the physical pages of the GPU memory region and create a mapping between the GPU memory region and the host memory region.
 * @param mbuffer: Pointer to the mmap buffer structure.
 * @param devaddr: The device address to be mapped.
 * @param dev_len: The length of the device memory region.
 * @return On success, 0 is returned.
 *         On failure, a negative error code is returned.
 */

int phxfs_map_dev_addr_inner(phxfs_mmap_buffer_t mbuffer, u64 devaddr, u64 dev_len) {
    ...    
    // calculate the number of device pages and host pages needed for the mapping
    nr_dev_pages = DIV_ROUND_UP(dev_len, page_size);

    mbuffer->subpage_num = page_size / PAGE_SIZE;
    mbuffer->dev_page_num = nr_dev_pages;
    mbuffer->host_page_num = nr_dev_pages * (mbuffer->subpage_num);

    ...

    mbuffer->ppages = (struct page **) kmalloc(mbuffer->host_page_num * sizeof(struct page *), GFP_KERNEL);
    mbuffer->map = kmalloc(sizeof(struct p2p_vmap) + (nr_dev_pages - 1) * sizeof(uint64_t), GFP_KERNEL);
    if (mbuffer->map == NULL) {
        printk("Failed to allocate mapping descriptor\n");
        return -ENOMEM;
    }

    // save the information of the mapping descriptor
    mbuffer->map->page_size = GPU_PAGE_SIZE;
    mbuffer->map->release = release_gpu_memory;
    mbuffer->map->size = dev_len;
    mbuffer->map->gpuvaddr = devaddr;
    mbuffer->map->n_addrs = mbuffer->dev_page_num; 

    gd = kmalloc(sizeof(struct gpu_region), GFP_KERNEL);

    gd->pages = NULL;
    mbuffer->map->data = (struct gpu_region*)gd;

    // get the physical pages of the GPU memory region, you can use replace it with your own function
    ret = nvfs_nvidia_p2p_get_pages(0, 0, mbuffer->map->gpuvaddr, 
                                    GPU_PAGE_SIZE * mbuffer->map->n_addrs, &gd->pages, 
                                    (void (*)(void*)) force_release_gpu_memory, mbuffer->map);   
    // save the physical addresses of the GPU memory region
    for(i = 0; i < mbuffer->map->n_addrs; i++)
    {
        if(gd->pages->pages[i]==NULL)
        {
            printk("mem allocation not success, i is %d!\n",i);
            goto out;
        }
        dev_page_addrs[i] = gd->pages->pages[i]->physical_address;
    }

    mbuffer->dev_page_addrs = dev_page_addrs;
    total_pages = mbuffer->host_page_num;
    
    // create the mapping between the GPU memory pages and the host memory pages
    for (i = 0; i < nr_dev_pages; i++) {
        pci_bar_off = dev_page_addrs[i] - mbuffer->dev->paddr;
        cpu_vaddr = (uint64_t)(mbuffer->dev->pci_mem_va + pci_bar_off);
        for (j = 0; j < mbuffer->subpage_num; j++) {
            mbuffer->ppages[i * mbuffer->subpage_num + j] = virt_to_page(cpu_vaddr + j * PAGE_SIZE);
        }
    }
    
    // establish the mapping between the GPU memory region and the host memory region via vm_insert_pages
    ret = vm_insert_pages(vma, mbuffer->c_vaddr, mbuffer->ppages, &total_pages);
    ...
}

/**
 * @brief Release the mapping descriptor and put the mmap buffer.
 * @param map_param: Pointer to the mapping parameters structure.
 * @param devaddr: The device address to be unmapped.
 * @param dev_len: The length of the device memory region.
 * @param cpuvaddr: The user-space virtual address to unmap from.
 * @param length: The length of the user-space virtual address region.
 */
void phxfs_map_dev_release(phxfs_ioctl_map_t *map_param, u64 devaddr, u64 dev_len, u64 cpuvaddr, u64 length) {
    phxfs_mmap_buffer_t mbuffer;
    // query the mmap buffer from the hash table using the user-space virtual address
    mbuffer = phxfs_check_and_bind_mmap_buffer(cpuvaddr, length);
    // put the mmap buffer and delete the mapping descriptor
    phxfs_mbuffer_put(mbuffer);
}

```

