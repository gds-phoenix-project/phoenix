#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <liburing.h>
#include <liburing/io_uring.h>
#include <sys/types.h>
#include <unistd.h>
#include "phxfs.h"
#include "phxfs_utils.h"


static void *async_thread(void *arg) {
    struct timespec io_start, io_end;
    struct io_uring_sqe *sqe;
    struct io_uring_cqe *cqe;
    phxfs_xfer_addr_list_t xfer_addr;
    ThreadData *data = (ThreadData *)arg;

    uint64_t io_time;
    size_t done_bytes = 0;
    size_t i, pending;
    size_t internal_bytes = 0;
    ssize_t io_size = (ssize_t)data->io_size;
    unsigned nr_completed = 0;

    unsigned long long chuck_size = data->size / data->depth;
    unsigned long long chuck_done_size = 0;

    
    int ret,j;
    int op = data->mode == 0 ? IORING_OP_READ : IORING_OP_WRITE;

    pr_info(__func__);
    
    while (done_bytes < data->size) {
        
        clock_gettime(CLOCK_MONOTONIC, &io_start);
        pending = 0;
        for (i = 0 ; i < data->depth; i++){
            if (chuck_done_size + (ssize_t)data->io_size  > chuck_size){
                pr_debug("out of range");
                break;
            }
            sqe = io_uring_get_sqe(data->ring);
            if (!sqe) {
                pr_error("io_uring_get_sqe failed");
                return NULL;
            }
            //
            phxfs_do_xfer_addr(data->device_id, (phxfs_io_para_s){
                .buf = data->gpu_buffer,
                .buf_offset = (off_t)(chuck_done_size + i * chuck_size),
                .nbyte = data->io_size
            }, &xfer_addr);
            internal_bytes = 0;
            if (xfer_addr.nr_xfer_addrs == 1){
                io_uring_prep_rw(op, sqe, data->fd,
                                 xfer_addr.x_addr.target_addr,
                                 xfer_addr.x_addr.nbyte,
                                 data->offset + done_bytes);
            }else{
                for (j = 0; j < xfer_addr.nr_xfer_addrs; j++){
                    io_uring_prep_rw(op, sqe, data->fd,
                                    xfer_addr.x_addrs[j].target_addr,
                                    xfer_addr.x_addrs[j].nbyte,
                                    data->offset + done_bytes + internal_bytes);
                    internal_bytes += xfer_addr.x_addrs[j].nbyte;
                }
            }
            pending += xfer_addr.nr_xfer_addrs;

            if (internal_bytes != data->io_size){
                pr_error("phxfs_xfer_addr faild");
            }
            io_uring_submit(data->ring);
        }
        nr_completed = 0;
        while (nr_completed != pending){
            ret = io_uring_wait_cqe(data->ring, &cqe);
            if (ret < 0 || cqe->res != io_size){ 
                pr_error("io_uring_wait_cqes failed");
                return NULL;
            }
            io_uring_cqe_seen(data->ring, cqe);
            nr_completed ++;
        }
        chuck_done_size += data->io_size;
        nr_completed = 0;
        clock_gettime(CLOCK_MONOTONIC, &io_end);
        done_bytes += data->io_size * pending;
        io_time = (io_end.tv_sec - io_start.tv_sec) * 1000000000LL + (io_end.tv_nsec - io_start.tv_nsec);
        data->io_operations ++;
        data->total_io_time += io_time;
        data->latency_vec.push_back(io_time);
    }
    return NULL;
}

static void *batch_thread(void *arg) {
    struct timespec io_start, io_end;
    struct io_uring_sqe *sqe;
    struct io_uring_cqe *cqe;
    phxfs_xfer_addr_list_t xfer_addr;
    ThreadData *data = (ThreadData *)arg;
    uint64_t io_time;
    size_t done_bytes = 0;
    size_t i, pending;
    size_t internal_bytes = 0;
    ssize_t io_size = (ssize_t)data->io_size;
    unsigned nr_completed = 0;

    unsigned long long chuck_size = data->size / data->depth;
    unsigned long long chuck_done_size = 0;

    
    int ret,j;
    int op = data->mode == 0 ? IORING_OP_READ : IORING_OP_WRITE;

    pr_info(__func__);
    
    while (done_bytes < data->size) {
        pending = 0;
        clock_gettime(CLOCK_MONOTONIC, &io_start);
        for (i = 0 ; i < data->depth; i++){
            if (chuck_done_size + (ssize_t)data->io_size  > chuck_size){
                pr_debug("out of range");
                break;
            }
            sqe = io_uring_get_sqe(data->ring);
            if (!sqe) {
                pr_error("io_uring_get_sqe failed");
                return NULL;
            }

            phxfs_do_xfer_addr(data->device_id, (phxfs_io_para_s){
                .buf = data->gpu_buffer,
                .buf_offset = (off_t)(chuck_done_size + i * chuck_size),
                .nbyte = data->io_size
            }, &xfer_addr);
            internal_bytes = 0;
            io_uring_prep_rw(op, sqe, data->fd,
                                xfer_addr.x_addr.target_addr,
                                xfer_addr.x_addr.nbyte,
                                data->offset + done_bytes);

            pending += xfer_addr.nr_xfer_addrs;
            if (internal_bytes != data->io_size){
                pr_error("phxfs_xfer_addr faild");
            }
        }
        pending = i;
        io_uring_submit(data->ring);
        nr_completed = 0;
        while (nr_completed != pending){
            ret = io_uring_wait_cqe(data->ring, &cqe);
            if (ret < 0 || cqe->res != io_size){ 
                pr_error("io_uring_wait_cqes failed");
                return NULL;
            }
            io_uring_cqe_seen(data->ring, cqe);
            nr_completed ++;
        }
        chuck_done_size += data->io_size;
        nr_completed = 0;
        clock_gettime(CLOCK_MONOTONIC, &io_end);
        done_bytes += data->io_size * pending;
        io_time = (io_end.tv_sec - io_start.tv_sec) * 1000000000LL + (io_end.tv_nsec - io_start.tv_nsec);
        data->io_operations ++;
        data->total_io_time += io_time;
        data->latency_vec.push_back(io_time);
    }
    return NULL;
}

static void *sync_write_thread(void *arg){
    struct timespec io_start, io_end;
    unsigned long io_time = 0;
    ThreadData *data = (ThreadData *)arg;
    ssize_t io_size = (ssize_t)data->io_size;
    size_t done_bytes = 0;
    phxfs_xfer_addr_list_t xfer_addr;

    pr_info(__func__);
    clock_gettime(CLOCK_MONOTONIC, &data->start_time);
    
    while (done_bytes < data->size) {
        clock_gettime(CLOCK_MONOTONIC, &io_start);
        phxfs_do_xfer_addr(data->device_id, (phxfs_io_para_s){
            .buf = data->gpu_buffer,
            .buf_offset = (off_t)(data->offset + done_bytes),
            .nbyte = data->io_size
        }, &xfer_addr);

        ssize_t result = pread(data->fd, xfer_addr.x_addr.target_addr, io_size, data->offset + done_bytes);
        if (result != io_size) {
            pr_error("read_thread error");
            return NULL;
        }
        if (result == 0) {
            // End of file reached
            break;
        }
        clock_gettime(CLOCK_MONOTONIC, &io_end);
        io_time = (io_end.tv_sec - io_start.tv_sec) * 1000000000LL + (io_end.tv_nsec - io_start.tv_nsec);
        data->total_io_time += io_time;
        data->latency_vec.push_back(io_time);
        data->io_operations++;
        done_bytes += result;
    }
    clock_gettime(CLOCK_MONOTONIC, &data->end_time);
    return NULL;
}


static void *sync_read_thread(void *arg){
    struct timespec io_start, io_end;
    ThreadData *data = (ThreadData *)arg;
    ssize_t io_size = (ssize_t)data->io_size;
    size_t done_bytes = 0;
    unsigned long io_time = 0;
    phxfs_xfer_addr_list_t xfer_addr;

    pr_info(__func__); 
    clock_gettime(CLOCK_MONOTONIC, &data->start_time);
    while (done_bytes < data->size) {
        clock_gettime(CLOCK_MONOTONIC, &io_start);
        phxfs_do_xfer_addr(data->device_id, (phxfs_io_para_s){
            .buf = data->gpu_buffer,
            .buf_offset = (off_t)(data->offset + done_bytes),
            .nbyte = data->io_size
        }, &xfer_addr);

        ssize_t result = pread(data->fd, xfer_addr.x_addr.target_addr, io_size, data->offset + done_bytes);

        if (result != io_size) {
            printf("read_thread error, result is %lu, size is %lu\n",result, data->io_size);
            return NULL;
        }
        if (result == 0) {
            // End of file reached
            break;
        }
        clock_gettime(CLOCK_MONOTONIC, &io_end);
        io_time = (io_end.tv_sec - io_start.tv_sec) * 1000000000LL + (io_end.tv_nsec - io_start.tv_nsec);;
        data->latency_vec.push_back(io_time);
        data->total_io_time += io_time;
        data->io_operations++;
        done_bytes += result;
    }
    clock_gettime(CLOCK_MONOTONIC, &data->end_time);
    return NULL;
}


int run_phoenix(GDSOpts opts){
    struct timespec prog_start, prog_end;
    GDSThread *threads;
    size_t chunk_size;
    std::vector<uint64_t> latency_vec;
    uint64_t total_io_operations = 0, total_io_time = 0;
    double average_io_latency, average_io_bandwidth;
    unsigned long long prog_time = 0;
    int file_fd, ret;

    static void *(*rw_funcs[3][2])(void *arg) = {
        {sync_read_thread, sync_write_thread}, 
        {async_thread, async_thread}, 
        {batch_thread, batch_thread}};

    threads = new GDSThread[opts.num_threads];
    thread_prep(threads, opts.num_threads);

    file_fd = open(opts.file_path,  O_CREAT | O_RDWR | O_DIRECT, 0644);

    if (file_fd < 0) {
        perror("Open file error");
        return 1;
    }

    check_cudaruntimecall(cudaSetDevice(opts.gpu_id));

    ret = phxfs_open(opts.gpu_id);

    if (ret != 0) {
        pr_error("phxfs init failed: " << ret);
        return 1;
    }

    chunk_size = opts.length / opts.num_threads;
    for (int i = 0; i < opts.num_threads; i++){
        ThreadData *data = &threads[i].data;
        data->thread_id = i;
        data->offset = i * chunk_size;
        data->size = chunk_size;
        data->buffer = NULL;
        data->total_io_time = 0;
        data->io_operations = 0;
        data->device_id = opts.gpu_id;
        data->io_size = opts.io_size;
        data->depth = opts.io_depth;
        data->fd = file_fd;
        data->mode = opts.mode;

        data->gpu_buffer = NULL;
        check_cudaruntimecall(cudaMalloc(&data->gpu_buffer, data->size));
        check_cudaruntimecall(cudaMemset(data->gpu_buffer, 0x00, data->size));
        check_cudaruntimecall(cudaStreamSynchronize(0));

        phxfs_regmem(data->device_id, data->gpu_buffer, data->size, &data->buffer);

        if (opts.async != 0){
            data->ring = new io_uring();
            struct io_uring_params params; 
            memset(&params, 0, sizeof(params));
            params.flags = 0 ;
            params.cq_entries = params.sq_entries = data->depth;

            ret = io_uring_queue_init_params(data->depth, data->ring, &params);
            
            if (ret) {
                pr_error("io_uring_queue_init failed: " << ret);
                return 1;
            }

            ret = io_uring_register_files(data->ring, &file_fd, 1);
            
            if (ret < 0 ){
                pr_error("io_uring_register_files fail , ret is" << ret);
                return 1;
            }
        }else{
            data->ring = NULL;
        }

        data->latency_vec.reserve(data->size / data->io_size + 10);

        if (ret){
            pr_error("phxfs regmem failed: " << ret);
            goto out;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &prog_start);
    for (int i = 0; i < opts.num_threads; i++) {
        if (pthread_create(&threads[i].thread, NULL, rw_funcs[opts.async][opts.mode], &threads[i].data) != 0) {
            perror("Pthread create error");
            goto out;
        }
    }

    for (int i = 0; i < opts.num_threads; i++) {
        pthread_join(threads[i].thread, NULL);
    }  
    clock_gettime(CLOCK_MONOTONIC, &prog_end);

    prog_time = (prog_end.tv_sec - prog_start.tv_sec) * 1000000000LL + (prog_end.tv_nsec - prog_start.tv_nsec);

    for (int i = 0; i < opts.num_threads; i++) {
        total_io_time += threads[i].data.total_io_time;
        total_io_operations += threads[i].data.io_operations;
        latency_vec.insert(latency_vec.end(), threads[i].data.latency_vec.begin(), threads[i].data.latency_vec.end());
    }
    pr_info("Total IO operations: " << total_io_operations);
    average_io_bandwidth = (((double)total_io_operations * opts.io_size * opts.io_depth)/(MB) ) / (1.0 * prog_time / 1000000000.0);
    pr_info("Average IO bandwidth: " << average_io_bandwidth << " MB/s");
    average_io_latency = (double)total_io_time / (total_io_operations * 1000.0);
    pr_info("Average IO latency: " << average_io_latency << " us");
    get_percentile(latency_vec);

    for (int i =0 ;i < opts.num_threads; i++){
        ret = phxfs_unregmem(threads[i].data.device_id, threads[i].data.gpu_buffer, threads[i].data.size);
        if (ret){
            pr_error("phxfs regmem failed: " << ret);
            return -1;
        }
        check_cudaruntimecall(cudaFree(threads[i].data.gpu_buffer));
    }

out:
    phxfs_close(opts.gpu_id);
    close(file_fd);

    return 0;
}