#include <cassert>
#include <emmintrin.h>
#include "smaug/core/datatypes.h"
#include "smaug/operators/common.h"
#include "smaug/utility/utils.h"

namespace smaug {

void* malloc_aligned(size_t size, bool zeroOut) {
    void* ptr = NULL;
    int err = posix_memalign(
            (void**)&ptr, CACHELINE_SIZE, next_multiple(size, CACHELINE_SIZE));
    assert(err == 0 && "Failed to allocate memory!");
    if (zeroOut)
        memset(ptr, 0, next_multiple(size, CACHELINE_SIZE));
    return ptr;
    
}

void flush_memory( void* addr, size_t size){
    /*unsigned long long num_lines = size / CACHELINE_SIZE;
    bool rem = (size % CACHELINE_SIZE == 0);
    void* cline = addr;
    unsigned long long iter =0;
    for ( iter = 0; iter < num_lines; iter ++){
        _mm_clflush(cline);
        cline += iter * CACHELINE_SIZE;
    }
    if ( rem)
        _mm_clflush(cline);
    _mm_mfence();
    */
    
}
std::string dataLayoutToStr(DataLayout layout) {
    switch (layout) {
        case DataLayout::NCHW:
            return "NCHW";
        case DataLayout::NHWC:
            return "NHWC";
        case DataLayout::NC:
            return "NC";
        case DataLayout::X:
            return "X";
        default:
            assert(false && "Unknown data layout!");
            return "";
    }
}

int calc_padding(int value, unsigned alignment) {
    if (alignment == 0 || value % alignment == 0)
        return 0;
    return (alignment - (value % alignment));
}

namespace gem5 {

#ifndef TRACE_MODE
void switchCpu() {
    if (runningInSimulation)
        m5_switch_cpu();
}

void dumpStats(const char* msg, int period) {
    if (runningInSimulation)
        m5_dump_stats(0, period, msg);
}

void dumpResetStats(const char* msg, int period) {
    if (runningInSimulation)
        m5_dump_reset_stats(0, period, msg);
}

void quiesce() {
    if (runningInSimulation) {
        m5_quiesce();
    }
}

void resetStats(int period){
    if ( runningInSimulation){
        m5_reset_stats(0,period);
    }
}

void wakeCpu(int id) {
    if (runningInSimulation) {
        m5_wake_cpu(id);
    }
}

uint64_t get_curr_Tick(){
    if (runningInSimulation)
        return m5_rpns();
    return 0;
}

int getCpuId() { return runningInSimulation ? m5_get_cpuid() : 0; }
#else
void switchCpu() {}

void dumpStats(const char* msg, int period) {}

void resetStats(int period) {}

uint64_t get_curr_Tick() { return 0; }

void dumpResetStats(const char* msg, int period) {}

void quiesce() {}

void wakeCpu(int id) {}

int getCpuId() { return 0; }
#endif

ScopedStats::ScopedStats(const char* _startLabel,
                         const char* _endLabel,
                         bool _resetStats)
        : startLabel(_startLabel), endLabel(_endLabel),
          resetStats(_resetStats) {
    if (resetStats)
        dumpResetStats(startLabel, 0);
    else
        dumpStats(startLabel, 0);
}

ScopedStats::~ScopedStats() {
    if (resetStats)
        dumpResetStats(endLabel, 0);
    else
        dumpStats(endLabel, 0);
}

}  // namespace gem5

}  // namespace smaug
