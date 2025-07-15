// unified_batch_reader.cpp


#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "phoenix.h"
#include "kvcache_reader.hh"
#include "cufile_sample_utils.h"

uint64_t block_size = 16 * 1024;  // default value
uint64_t MAX_BLOCKS_SIZE = 16 * 1024 * 1024; // 16MB
std::string block_file = "/mnt/phxfs/data.bin";


CuFileKVCacheReader::CuFileKVCacheReader(size_t max_batch_size_): max_batch_size(max_batch_size_) {
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("cuFile driver open failed: " + 
            std::string(cuFileGetErrorString(status)));
    }

    
    fd = open(block_file.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        throw std::runtime_error("File open failed: " + 
            std::string(strerror(errno)));
    }

    CUfileDescr_t cf_descr;
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        close(fd);
        throw std::runtime_error("File handle register failed");
    }

    devPtrs = new void*[max_batch_size];
    
    // 分配并注册每个buffer
    for (size_t i = 0; i < max_batch_size; i++) {
        
        cudaError_t cuda_status = cudaMalloc(&devPtrs[i], MAX_BLOCKS_SIZE);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("CUDA memory allocation failed");
        }
        
        status = cuFileBufRegister(devPtrs[i], MAX_BLOCKS_SIZE, 0);
        if (status.err != CU_FILE_SUCCESS) {
            throw std::runtime_error("Buffer register failed");
        }
    }

    status = cuFileBatchIOSetUp(&batch_id, max_batch_size);
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("Batch setup failed");
    }
}

void CuFileKVCacheReader::load_sequences(const std::string& trace_file) {
    std::ifstream file(trace_file);
    if (!file) {
        throw std::runtime_error("Cannot open trace file: " + trace_file);
    }

    std::string line;
    Sequence current_seq;
    
    while (std::getline(file, line)) {
        if (line.find("Seq") == 0) {  // 新序列开始
            if (!current_seq.id.empty()) {
                sequences.push_back(current_seq);
            }
            current_seq = Sequence();
            // 提取序列ID
            size_t pos = line.find("conversation id: ");
            if (pos != std::string::npos) {
                current_seq.id = line.substr(pos + 17);
                current_seq.id = current_seq.id.substr(0, current_seq.id.find(")"));
            }
        } else if (line.find("Round") == 0 && line.find("[") != std::string::npos) {  // 包含block IDs的行
            size_t start = line.find("[");
            size_t end = line.find("]");
            if (start != std::string::npos && end != std::string::npos) {
                std::string numbers = line.substr(start + 1, end - start - 1);
                std::stringstream ss(numbers);
                std::string number;
                while (std::getline(ss, number, ',')) {
                    // 去除前后空格
                    number.erase(0, number.find_first_not_of(" "));
                    number.erase(number.find_last_not_of(" ") + 1);
                    if (!number.empty()) {
                        current_seq.block_ids.push_back(std::stoul(number));
                    }
                }
            }
        }
    }
    
    if (!current_seq.id.empty()) {
        sequences.push_back(current_seq);
    }

    std::cout << "Loaded " << sequences.size() << " sequences" << std::endl;
}

void CuFileKVCacheReader::process_all_sequences() {
    // 开始记录端到端时间
    auto total_start = std::chrono::high_resolution_clock::now();
    
    size_t total_bytes = 0;
    double total_io_time = 0.0;  // 累计所有序列的IO时间

    CUfileIOParams_t *io_batch_params = new CUfileIOParams_t[max_batch_size];
    CUfileIOEvents_t *io_batch_events = new CUfileIOEvents_t[max_batch_size];
    
    for (const auto& seq : sequences) {
        size_t seq_bytes = seq.block_ids.size() * block_size;
        total_bytes += seq_bytes;
        
        // 记录序列的io端到端时间
        auto seq_start = std::chrono::high_resolution_clock::now();
        size_t current_batch_size = 0;
        
        // 处理这个序列的所有blocks
        for (size_t offset = 0; offset < seq.block_ids.size(); offset++) {
            
            // 准备IO参数
            if (current_batch_size != 0 && 
                io_batch_params[current_batch_size - 1].u.batch.file_offset + block_size == seq.block_ids[offset] * block_size
                && io_batch_params[current_batch_size - 1].u.batch.size + block_size <= MAX_BLOCKS_SIZE) {
                    io_batch_params[current_batch_size - 1].u.batch.size += block_size;
            } else if (current_batch_size < max_batch_size){
                io_batch_params[current_batch_size].mode = CUFILE_BATCH;
                io_batch_params[current_batch_size].fh = cf_handle;
                io_batch_params[current_batch_size].u.batch.devPtr_base = devPtrs[current_batch_size];
                io_batch_params[current_batch_size].u.batch.devPtr_offset = 0;
                io_batch_params[current_batch_size].u.batch.file_offset = (off_t)(seq.block_ids[offset] * block_size);
                io_batch_params[current_batch_size].u.batch.size = block_size;
                io_batch_params[current_batch_size].opcode = CUFILE_READ;
                current_batch_size++;
            } 

            if (current_batch_size == max_batch_size || offset == seq.block_ids.size() - 1) {

                CUfileError_t status = cuFileBatchIOSubmit(batch_id, current_batch_size, 
                                                           io_batch_params, 0);
                if (status.err != CU_FILE_SUCCESS) {
                    std::cout << "Batch submit failed: " << cuFileGetErrorString(status) << std::endl;
                    throw std::runtime_error("Batch submit failed");
                }

                // 等待完成
                unsigned int num_completed = 0;
                while (num_completed < current_batch_size) {
                    unsigned int nr = current_batch_size;
                    status = cuFileBatchIOGetStatus(batch_id, current_batch_size, &nr, 
                                                  io_batch_events, NULL);
                    if (status.err != CU_FILE_SUCCESS) {
                        throw std::runtime_error("Get status failed");
                    }
                    num_completed += nr;
                }
                // std::cout << "Batch completed: " << num_completed << std::endl;
                
                // 重置当前batch大小
                current_batch_size = 0;
            }
        }
        
        // 序列io端到端时间
        auto seq_end = std::chrono::high_resolution_clock::now();
        auto seq_duration = std::chrono::duration<double, std::milli>(seq_end - seq_start);
        
        total_io_time += seq_duration.count();  // 累加到总IO时间
    
    }
    
    // 计算端到端总时间
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration<double, std::milli>(total_end - total_start);
    
    // 计算带宽
    double io_bandwidth = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / 
                         (total_io_time / 1000.0);
    double total_bandwidth = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / 
                           (total_duration.count() / 1000.0);

    std::cout << "\nTotal Summary:"
              << "\n  Sequences: " << sequences.size()
              << "\n  Total Data: " << (total_bytes / (1024.0 * 1024.0)) << " MB"
              << "\n  Pure IO Time: " << total_io_time << " ms"
              << ", IO Bandwidth: " << io_bandwidth << " GB/s"
              << "\n  End-to-end Time: " << total_duration.count() << " ms"
              << ", Effective Bandwidth: " << total_bandwidth << " GB/s"
              << "\n  Overhead Time: " << (total_duration.count() - total_io_time) << " ms"
              << " (" << ((total_duration.count() - total_io_time) / total_duration.count() * 100) << "%)"
              << std::endl;
}

CuFileKVCacheReader::~CuFileKVCacheReader() {
    cuFileBatchIODestroy(batch_id);  
    for (size_t i = 0; i < max_batch_size; i++) {
        cuFileBufDeregister(devPtrs[i]);
        cudaFree(devPtrs[i]);
    }
    delete[] devPtrs;

    cuFileHandleDeregister(cf_handle);
    close(fd);
    cuFileDriverClose();
}

PhxfsKVCacheReader::PhxfsKVCacheReader(size_t max_batch_size_, int device_id_) 
                    : device_id(device_id_), max_batch_size(max_batch_size_) {
    int ret = phxfs_open(this->device_id);
    if (ret) {
        throw std::runtime_error("cuFile driver open failed: " + std::to_string(ret));
    }
    
    fd = open(block_file.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        throw std::runtime_error("File open failed: " + 
            std::string(strerror(errno)));
    }

    devPtrs = new void*[max_batch_size];
    host_ptrs = new void*[max_batch_size];

    
    // 分配并注册每个buffer
    for (size_t i = 0; i < max_batch_size; i++) {
        
        cudaError_t cuda_status = cudaMalloc(&devPtrs[i], MAX_BLOCKS_SIZE);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("CUDA memory allocation failed");
        }
        
        ret = phxfs_regmem(this->device_id, devPtrs[i], MAX_BLOCKS_SIZE, &host_ptrs[i]);
        if (ret) {
            throw std::runtime_error("Buffer register failed");
        }
    }

    // 初始化时设置batch
    struct io_uring_params params; 
    memset(&params, 0, sizeof(params));
    params.flags |= IORING_SETUP_SQPOLL ;
    // params.flags = 0 ;
    params.cq_entries = params.sq_entries = max_batch_size;
    params.sq_thread_idle = 0x2;

    ret = io_uring_queue_init_params(max_batch_size, &ring, &params);
    if (ret) {
        throw std::runtime_error("io_uring_queue_init failed: " + std::to_string(ret));
    }
    ret = io_uring_register_files(&ring, &fd, 1);
        
    if (ret < 0 ){
        throw std::runtime_error("io_uring_queue_init failed: " + std::to_string(ret));
    }

    ret = io_uring_enter(ring.ring_fd, 0, 0, IORING_ENTER_SQ_WAKEUP, NULL);

}

void PhxfsKVCacheReader::load_sequences(const std::string& trace_file) {
    std::ifstream file(trace_file);
    if (!file) {
        throw std::runtime_error("Cannot open trace file: " + trace_file);
    }

    std::string line;
    Sequence current_seq;
    
    while (std::getline(file, line)) {
        if (line.find("Seq") == 0) {
            if (!current_seq.id.empty()) {
                sequences.push_back(current_seq);
            }
            current_seq = Sequence();
            size_t pos = line.find("conversation id: ");
            if (pos != std::string::npos) {
                current_seq.id = line.substr(pos + 17);
                current_seq.id = current_seq.id.substr(0, current_seq.id.find(")"));
            }
        } else if (line.find("Round") == 0 && line.find("[") != std::string::npos) {
            size_t start = line.find("[");
            size_t end = line.find("]");
            if (start != std::string::npos && end != std::string::npos) {
                std::string numbers = line.substr(start + 1, end - start - 1);
                std::stringstream ss(numbers);
                std::string number;
                while (std::getline(ss, number, ',')) {
                    number.erase(0, number.find_first_not_of(" "));
                    number.erase(number.find_last_not_of(" ") + 1);
                    if (!number.empty()) {
                        current_seq.block_ids.push_back(std::stoul(number));
                    }
                }
            }
        }
    }
    
    if (!current_seq.id.empty()) {
        sequences.push_back(current_seq);
    }



    std::cout << "Loaded " << sequences.size() << " sequences" << std::endl;
}

void PhxfsKVCacheReader::process_all_sequences() {
    struct io_uring_sqe *sqe;
    struct io_uring_cqe *cqe;
    auto total_start = std::chrono::high_resolution_clock::now();
    size_t total_bytes = 0;
    double total_io_time = 0.0;  // 累计所有序列的IO时间


    for (const auto& seq : sequences) {
        size_t seq_bytes = seq.block_ids.size() * block_size;
        total_bytes += seq_bytes;
        
        CUfileIOParams_t *io_batch_params = new CUfileIOParams_t[this->max_batch_size];
        auto seq_start = std::chrono::high_resolution_clock::now();
        unsigned int current_batch_size = 0;
        
        for (size_t offset = 0; offset < seq.block_ids.size(); offset++) {
            if (current_batch_size != 0 && 
                io_batch_params[current_batch_size - 1].u.batch.file_offset + block_size == seq.block_ids[offset] * block_size
                && io_batch_params[current_batch_size - 1].u.batch.size + block_size <= MAX_BLOCKS_SIZE) {
                    io_batch_params[current_batch_size - 1].u.batch.size += block_size;
            } else if (current_batch_size < this->max_batch_size){
                io_batch_params[current_batch_size].mode = CUFILE_BATCH;
                io_batch_params[current_batch_size].u.batch.devPtr_base = host_ptrs[current_batch_size];
                io_batch_params[current_batch_size].u.batch.file_offset = (off_t)(seq.block_ids[offset] * block_size);
                io_batch_params[current_batch_size].u.batch.size = block_size;
                current_batch_size++;
            } 

            if (current_batch_size == this->max_batch_size || offset == seq.block_ids.size() - 1) {
                for (size_t i = 0; i < current_batch_size; i++) {
                    sqe = io_uring_get_sqe(&ring);
                    if (!sqe) {
                        throw std::runtime_error("Failed to get SQE");
                    }
                    io_uring_prep_read(sqe, fd, host_ptrs[i], io_batch_params[i].u.batch.size, 
                                      io_batch_params[i].u.batch.file_offset);
                }
                // 提交IO请求
                int ret = io_uring_submit(&ring);
                if (ret < 0) {
                    throw std::runtime_error("Failed to submit IO request: " + std::to_string(ret));
                }
                unsigned int num_completed = 0;
                while (num_completed < current_batch_size) {
                    ret = io_uring_wait_cqe(&ring, &cqe);
                    if (ret < 0) {
                        throw std::runtime_error("Failed to wait for CQE: " + std::to_string(ret));
                    }
                    if (cqe->res < 0) {
                        std::cout << "error" << errno << ":" << std::strerror(errno) << std::endl;
                        throw std::runtime_error("IO request failed: " + std::to_string(cqe->res));
                    }
                    io_uring_cqe_seen(&ring, cqe);
                    num_completed++;
                }
                
                
            }
        }
        auto seq_end = std::chrono::high_resolution_clock::now();
        auto seq_duration = std::chrono::duration<double, std::milli>(seq_end - seq_start);
        
        total_io_time += seq_duration.count();  // 累加到总IO时间

    }
    
    // 计算端到端总时间
    // 计算端到端总时间
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration<double, std::milli>(total_end - total_start);
    
    // 计算带宽
    double io_bandwidth = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / 
                         (total_io_time / 1000.0);
    double total_bandwidth = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / 
                           (total_duration.count() / 1000.0);

    std::cout << "\nTotal Summary:"
              << "\n  Sequences: " << sequences.size()
              << "\n  Total Data: " << (total_bytes / (1024.0 * 1024.0)) << " MB"
              << "\n  Pure IO Time: " << total_io_time << " ms"
              << ", IO Bandwidth: " << io_bandwidth << " GB/s"
              << "\n  End-to-end Time: " << total_duration.count() << " ms"
              << ", Effective Bandwidth: " << total_bandwidth << " GB/s"
              << "\n  Overhead Time: " << (total_duration.count() - total_io_time) << " ms"
              << " (" << ((total_duration.count() - total_io_time) / total_duration.count() * 100) << "%)"
              << std::endl;
}

PhxfsKVCacheReader::~PhxfsKVCacheReader() {
    io_uring_queue_exit(&ring);
    for (size_t i = 0; i < max_batch_size; i++) {
        phxfs_deregmem(this->device_id, devPtrs[i], MAX_BLOCKS_SIZE);
        cudaFree(devPtrs[i]);
    }
    delete[] devPtrs;
    close(fd);
    phxfs_close(this->device_id);
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <type: phxfs|gds> <gpu_id> <trace_file> <block_size>" << std::endl;
        return 1;
    }

    int type = 0;
    std::string type_str = std::string(argv[1]);
    int gpu_id = atoi(argv[2]);
    std::string trace_file = argv[3];
    block_size = atoll(argv[4]);
    block_file = argv[5];

    type = type_str == "phxfs" ? 0 : 1;

    std::cout << "KVCache Reader: " << std::endl
            << "  Type: " << type << std::endl
            << "  GPU ID: " << gpu_id << std::endl
            << "  Trace File: " << trace_file << std::endl
            << "  Block Size: " << block_size << std::endl
            << "  Block File: " << block_file << std::endl;

    try {
        cudaSetDevice(gpu_id);

        BaseKVCacheReader *reader = nullptr;
        if (type == 0) {
            reader = new PhxfsKVCacheReader();
        } else if (type == 1) {
            reader = new CuFileKVCacheReader();
        } else {
            throw std::invalid_argument("Invalid reader type");
        }
        reader->load_sequences(trace_file);
        reader->process_all_sequences();
        delete reader;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}