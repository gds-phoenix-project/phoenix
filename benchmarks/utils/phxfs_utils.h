#ifndef __GDS_TEST_H__
#define __GDS_TEST_H__

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <pthread.h>
#include <string>
#include <sys/types.h>
#include <getopt.h>
#include <iostream>
#include <cstring>
#include <cufile.h>
#include <cuda.h>
#include <array>
#include <memory>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm> 
#include <liburing.h>

// global var
#define DEFAULT_SIZE (1UL << 35) // 32 GB
#define KB (1UL << 10)
#define MB (1UL << 20)
#define GB (1UL << 30)
#define MAX_GPU_SIZE 44 * GB // no more than 45GB

#define IO_URING_MAX_DEPTH 64

// macro definition
#define point_offset(ptr, offset) reinterpret_cast<void*>(reinterpret_cast<uint64_t>(ptr) + offset)

#define check_cudaruntimecall(fn) \
	do { \
		cudaError_t res = fn; \
		if (res != cudaSuccess) { \
			const char *str = cudaGetErrorName(res); \
			std::cerr << "cuda runtime api call failed " << #fn \
				<<  __LINE__ << ":" << str << std::endl; \
			std::cerr << "EXITING program!!!" << std::endl; \
			exit(1); \
		} \
	} while(0)
#ifdef DEBUG
#define pr_debug(msg) \
    do { \
        std::cout << msg << std::endl; \
    } while(0)
#else
#define pr_debug(msg) 
#endif

#define pr_info(msg) \
    do { \
        std::cout << msg << std::endl; \
    } while(0)

#define pr_error(msg) \
    do { \
        std::cerr << msg << std::endl; \
    } while(0)

enum gds_op{
    OP_READ = 0,
    OP_WRITE = 1
};

enum xfer_mode{
    GPUD_WITHOUT_PHONY_BUFFER = 0,
    GPUD_WITH_PYONY_BUFFER = 1,
    GPUD_WITH_CPU_BUFFER = 2,
    END
};

typedef struct{
    bool mode;
    int async;
    char *file_path;
    size_t length;
    size_t io_size;
    int io_depth;
    int gpu_id;
    int num_threads;
    int xfer_mode;
}GDSOpts;

typedef struct {
    int thread_id;
    int device_id;
    int mode;
    int fd;
    off_t offset;
    size_t size;
    size_t io_size;
    size_t depth;
    size_t batch_size;
    void *buffer;
    void *gpu_buffer;
    void **gpu_buffers;
    struct timespec start_time;
    struct timespec end_time;
    long long total_io_time; // In nanoseconds
    unsigned long long io_operations; 
    std::vector<uint64_t> latency_vec;
    void *handler;
    struct io_uring *ring;
} ThreadData;

typedef struct{
    pthread_t thread;
    ThreadData data;
} GDSThread;

typedef struct io_args_s{
    void *devPtr;
    size_t io_size;
    off_t f_offset;
    off_t buf_off;
    ssize_t bytes_done;
    cudaEvent_t start;
    cudaEvent_t end;
} io_args_t;

static inline void thread_prep(GDSThread *threads, int num_threads){
    for (int i = 0;i < num_threads; i ++){
        ThreadData *data = &threads[i].data;
        data->thread_id = i;
        data->device_id = 0;
        data->offset = 0;
        data->size = 0;
        data->io_size = 0;
        data->depth = 0;
        data->buffer = NULL;
        data->gpu_buffer = NULL;
        data->fd = 0;
        data->start_time.tv_sec = 0;
        data->start_time.tv_nsec = 0;
        data->end_time.tv_sec = 0;
        data->end_time.tv_nsec = 0;
        data->total_io_time = 0;
        data->io_operations = 0;
    }
}

static inline void infoGDSOpts(const GDSOpts& opts) {
    std::cout << "GDSOpts:" << std::endl;
    std::cout << "  mode: " << opts.mode << std::endl;
    std::cout << "  async: " << opts.async << std::endl;
    std::cout << "  file_path: " << (opts.file_path ? opts.file_path : "null") << std::endl;
    std::cout << "  length: " << opts.length << std::endl;
    std::cout << "  io_size: " << opts.io_size << std::endl;
    std::cout << "  gpu_id: " << opts.gpu_id << std::endl;
    std::cout << "  num_threads: " << opts.num_threads << std::endl;
    std::cout << "  xfer_mode: " << opts.xfer_mode << std::endl;
    std::cout << "  io_depth: " << opts.io_depth << std::endl;
}

static inline void printThreadData(const ThreadData& data) {
    std::cout << "ThreadData:" << std::endl;
    std::cout << "  thread_id: " << data.thread_id << std::endl;
    std::cout << "  device_id: " << data.device_id << std::endl;
    std::cout << "  offset: " << data.offset << std::endl;
    std::cout << "  size: " << data.size << std::endl;
    std::cout << "  io_size: " << data.io_size << std::endl;
    std::cout << "  buffer: " << data.buffer << std::endl;
    std::cout << "  gpu_buffer: " << data.gpu_buffer << std::endl;
    std::cout << "  fd: " << data.fd << std::endl;
    std::cout << "  start_time: " << data.start_time.tv_sec << "s " << data.start_time.tv_nsec << "ns" << std::endl;
    std::cout << "  end_time: " << data.end_time.tv_sec << "s " << data.end_time.tv_nsec << "ns" << std::endl;
    std::cout << "  total_io_time: " << data.total_io_time << " ns" << std::endl;
    std::cout << "  io_operations: " << data.io_operations << std::endl;
}


static inline void printHelp(const char *progName) {
    std::cerr << "Usage: " << progName << " [options]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -m, --mode <write|read>      Set mode (write or read)" << std::endl;
    std::cerr << "  -a, --async <0|1>           Set async mode (0 for sync, 1 for async)" << std::endl;
    std::cerr << "  -f, --file <file_path>       Set the file path for the operation" << std::endl;
    std::cerr << "  -l, --length <length>        Set the length of the data (e.g., 1MB, 512KB)" << std::endl;
    std::cerr << "  -s, --size <size>            Set the IO size (e.g., 1KB, 64KB, etc.)" << std::endl;
    std::cerr << "  -d, --device <gpu_id>        Set the GPU device id" << std::endl;
    std::cerr << "  -t, --threads <num_threads>  Set the number of threads" << std::endl;
    std::cerr << "  -x, --xfer_mode <mode>       Set the transfer mode (integer value)" << std::endl;
    std::cerr << "  -i, --iodepth <depth>        Set the IO depth (number of simultaneous IOs)" << std::endl;
    std::cerr << "  -h, --help                   Show this help message" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example usage:" << std::endl;
    std::cerr << "  " << progName << " -m write -a 1 -f /path/to/file -l 1GB -s 64KB -d 0 -t 4 -x 1 -i 32" << std::endl;
}

static inline ssize_t get_size(std::string optarg){
    char unit = optarg.back();
    size_t multiplier = 1;
    if (unit == 'K' || unit == 'k')
        multiplier = KB;
    else if (unit == 'M' || unit == 'm')
        multiplier = MB;
    else if (unit == 'G' || unit == 'g')
        multiplier = GB;
    else if (unit >= '0' && unit <= '9')
        multiplier = 1;
    else
        return 4 * KB;
    if (multiplier != 1)
        optarg.pop_back();
    return std::stoull(optarg) * multiplier;
}

static inline bool parseOpts(int argc, char *argv[], GDSOpts &args) {
    int opt;
    static struct option long_options[] = {
        {"mode", required_argument, nullptr, 'm'},
        {"async", required_argument, nullptr, 'a'},
        {"file", required_argument, nullptr, 'f'},
        {"length", required_argument, nullptr, 'l'},
        {"size", required_argument, nullptr, 's'},
        {"device", required_argument, nullptr, 'd'},
        {"threads", required_argument, nullptr, 't'},
        {"xfer_mode", required_argument, nullptr, 'x'},
        {"iodepth", required_argument, nullptr, 'i'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    while ((opt = getopt_long(argc, argv, "m:a:f:l:s:d:t:x:i:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': // mode
                args.mode = (std::strcmp(optarg, "write") == 0);
                break;
            case 'a': // async
                args.async = std::stoi(optarg);
                break;
            case 'f': // file_path
                args.file_path = optarg;
                break;
            case 'l': // length
                args.length = get_size(std::string(optarg));
                break;
            case 's': { // io_size                
                args.io_size = get_size(std::string(optarg));
                break;
            }
            case 'd': // gpu_id
                args.gpu_id = std::stoi(optarg);
                break;
            case 't': // num_threads
                args.num_threads = std::stoi(optarg);
                break;
            case 'x': // xfer_mode
                args.xfer_mode = std::stoi(optarg);
                break;
            case 'i': // iodepth
                args.io_depth = std::stoi(optarg);
                break;
            case 'h': // help
                printHelp(argv[0]);
                return false;
            default:
                std::cerr << "Unspport option: " << opt << std::endl;
                return false;
        }
    }  
    if (!(args.file_path && args.length > 0 && args.io_size > 0 && args.num_threads > 0 && args.gpu_id >= 0)) {
        std::cerr << "Error: Missing or invalid arguments:" << std::endl;
        return false;
    }

    infoGDSOpts(args);

    return true;
}

static inline std::string exec_cmd(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result = "";
    
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("Failed to execute command: " + cmd);
    }
    
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result;
}


static inline void get_percentile(std::vector<uint64_t> &latency_vec){
    double p_flag[] = {0.95, 0.99, 0.999};
    std::sort(latency_vec.begin(), latency_vec.end());
    for (int i = 0; i < 3; i++){
        size_t index = (size_t)(p_flag[i] * latency_vec.size());
        pr_info(p_flag[i] * 100 << "th percentile latency: "<< 1.0 * latency_vec[index] / 1000.0 << " us");
    }
}

int run_gds(GDSOpts opts);
int run_phxfs(GDSOpts opts);
int run_posix(GDSOpts opts);

#endif