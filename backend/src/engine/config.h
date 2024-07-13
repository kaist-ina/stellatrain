#ifndef _ENGINE_CONFIG_H
#define _ENGINE_CONFIG_H
#include <string>
#include <cstdint>

/* Workers*/
const unsigned NUM_WORKER_THREADS_MASTER = 32;
const unsigned NUM_WORKER_THREADS = 16;

/* Shared Memory */
const unsigned MAX_SHM_CHUNKS = 128;
const unsigned MAX_SHARED_TENSORS = 512;
const unsigned MAX_TENSOR_ID_NAME_LEN = 63;

const unsigned MAX_NUM_BARRIERS = 1024;
const unsigned MAX_SHM_BARRIER_NAME_LEN = 63;

static_assert((MAX_TENSOR_ID_NAME_LEN + 1) % 8 == 0);
static_assert((MAX_SHM_BARRIER_NAME_LEN + 1) % 8 == 0);

static const std::string SHM_NAME_PREFIX("/fasterdp-shm-");
static const std::string SEM_NAME_PREFIX("/fasterdp-sem-");
static const off64_t META_BYTES_ALLOC = 64 * 1024 * 1024;
static const off64_t DATA_BYTES_ALLOC = 4LLU * 1024 * 1024 * 1024;

/* Task */
const size_t MAX_TASK_NAME_LEN = 63;
const size_t MAX_TENSOR_SHAPE_LEN = 16;
const size_t MAX_NUM_GPU_PER_NODE = 8;
const size_t MAX_NUM_BARRIER = 8;
const size_t NUM_MAX_PARAM_DIM = 8;
const size_t MAX_BARRIER_NAME_LEN = 31;


/* Debugging Flags */

#define USE_LOWER_PRIORITY_FOR_WORKERS 0

#define ENABLE_STAT 1

#define PROFILER_ENABLED 1

/** Implement Barrier Checker */
#define DEBUG_BARRIER 0

/** Print state changes. */
#define DEBUG_STATE_CHANGE 0

/** Debug accuracy. */
#define DEBUG_ACCURACY 0

/** Merge implementation*/
#define MERGE 1

/** Skip compressor */
#define SKIP_SOME_CRITICAL_PATHS 0

/** Flag to skip freeing `TrainTask` to mitigate crash. */
#define HACK_SKIP_FREE_TRAIN_TASK 0

#define PRIORITY_TX 1
#define PRIORITY_SCHED 1
#define IDX_COMPRESSION 1
#define FP16_COMPRESSION 0

#define IS_SPEED_AFFECTED ((DEBUG_BARRIER) || (DEBUG_ACCURACY) || (DEBUG_STATE_CHANGE))
#define IS_ACCURACY_AFFECTED ((DEBUG_BARRIER) || (SKIP_SOME_CRITICAL_PATHS))

#endif