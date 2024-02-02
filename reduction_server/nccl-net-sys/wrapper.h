/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2024, The Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 ************************************************************************/

#ifndef NCCL_NET_H_
#define NCCL_NET_H_

#include <stdint.h>
#include <stddef.h>

/* Error type */
typedef enum
{
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6,
  ncclInProgress = 7,
  ncclNumResults = 8
} ncclResult_t;

/* Reduction operation selector */
typedef enum
{
  ncclNumOps_dummy = 5
} ncclRedOp_dummy_t;
typedef enum
{
  ncclSum = 0,
  ncclProd = 1,
  ncclMax = 2,
  ncclMin = 3,
  ncclAvg = 4,
  /* ncclNumOps: The number of built-in ncclRedOp_t values. Also
   * serves as the least possible value for dynamic ncclRedOp_t's
   * as constructed by ncclRedOpCreate*** functions. */
  ncclNumOps = 5,
  /* ncclMaxRedOp: The largest valid value for ncclRedOp_t.
   * It is defined to be the largest signed value (since compilers
   * are permitted to use signed enums) that won't grow
   * sizeof(ncclRedOp_t) when compared to previous NCCL versions to
   * maintain ABI compatibility. */
  ncclMaxRedOp = 0x7fffffff >> (32 - 8 * sizeof(ncclRedOp_dummy_t))
} ncclRedOp_t;

/* Data types */
typedef enum
{
  ncclInt8 = 0,
  ncclChar = 0,
  ncclUint8 = 1,
  ncclInt32 = 2,
  ncclInt = 2,
  ncclUint32 = 3,
  ncclInt64 = 4,
  ncclUint64 = 5,
  ncclFloat16 = 6,
  ncclHalf = 6,
  ncclFloat32 = 7,
  ncclFloat = 7,
  ncclFloat64 = 8,
  ncclDouble = 8,
#if defined(__CUDA_BF16_TYPES_EXIST__)
  ncclBfloat16 = 9,
  ncclNumTypes = 10
#else
  ncclNumTypes = 9
#endif
} ncclDataType_t;

#define NCCL_NET_HANDLE_MAXSIZE 128

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

// Maximum number of requests per comm object
#define NCCL_NET_MAX_REQUESTS 8

typedef enum
{
  NCCL_LOG_NONE = 0,
  NCCL_LOG_VERSION = 1,
  NCCL_LOG_WARN = 2,
  NCCL_LOG_INFO = 3,
  NCCL_LOG_ABORT = 4,
  NCCL_LOG_TRACE = 5
} ncclDebugLogLevel;
typedef enum
{
  NCCL_INIT = 1,
  NCCL_COLL = 2,
  NCCL_P2P = 4,
  NCCL_SHM = 8,
  NCCL_NET = 16,
  NCCL_GRAPH = 32,
  NCCL_TUNING = 64,
  NCCL_ENV = 128,
  NCCL_ALLOC = 256,
  NCCL_CALL = 512,
  NCCL_PROXY = 1024,
  NCCL_NVLS = 2048,
  NCCL_ALL = ~0
} ncclDebugLogSubSys;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

typedef struct
{
  char *name;     // Used mostly for logging.
  char *pciPath;  // Path to the PCI device in /sys.
  uint64_t guid;  // Unique identifier for the NIC chip. Important for
                  // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport; // [NCCL_PTR_HOST|NCCL_PTR_CUDA|NCCL_PTR_DMABUF]
  int speed;      // Port speed in Mbps.
  int port;       // Port number.
  float latency;  // Network latency
  int maxComms;   // Maximum number of comms we can create
  int maxRecvs;   // Maximum number of grouped receives.
} ncclNetProperties_v6_t;

typedef ncclNetProperties_v6_t ncclNetProperties_t;

typedef struct
{
  // Name of the network (mainly for logs)
  const char *name;
  // Initialize the network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int *ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  ncclResult_t (*connect)(int dev, void *handle, void **sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  ncclResult_t (*accept)(void *listenComm, void **recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void *comm, void *data, int size, int type, void **mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void *comm, void *data, size_t size, int type, uint64_t offset, int fd, void **mhandle);
  ncclResult_t (*deregMr)(void *comm, void *mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void *sendComm, void *data, int size, int tag, void *mhandle, void **request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void *recvComm, int n, void **data, int *sizes, int *tags, void **mhandles, void **request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void *recvComm, int n, void **data, int *sizes, void **mhandles, void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void *request, int *done, int *sizes);
  // Close and free send/recv comm objects
  ncclResult_t (*closeSend)(void *sendComm);
  ncclResult_t (*closeRecv)(void *recvComm);
  ncclResult_t (*closeListen)(void *listenComm);
} ncclNet_v6_t;

typedef ncclNet_v6_t ncclNet_t;

#define NCCL_PLUGIN_SYMBOL ncclNetPlugin_v6

extern ncclNet_t NCCL_PLUGIN_SYMBOL;

typedef struct
{
  // Name of the collective network (mainly for logs)
  const char *name;
  // Initialize the collective network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  ncclResult_t (*devices)(int *ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void *handles[], int nranks, int rank, void *listenComm, void **collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  ncclResult_t (*reduceSupport)(ncclDataType_t dataType, ncclRedOp_t redOp, int *supported);
  // Register/Deregister memory. Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void *collComm, void *data, int size, int type, void **mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void *collComm, void *data, size_t size, int type, uint64_t offset, int fd, void **mhandle);
  ncclResult_t (*deregMr)(void *collComm, void *mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  ncclResult_t (*iallreduce)(void *collComm, void *sendData, void *recvData, int count,
                             ncclDataType_t dataType, ncclRedOp_t redOp, void *sendMhandle, void *recvMhandle, void **request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void *collComm, void *data, int size, void *mhandle, void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void *request, int *done, int *size);
  // Close and free collective comm objects
  ncclResult_t (*closeColl)(void *collComm);
  ncclResult_t (*closeListen)(void *listenComm);
} ncclCollNet_v6_t;

typedef ncclCollNet_v6_t ncclCollNet_t;

#define NCCL_COLLNET_PLUGIN_SYMBOL ncclCollNetPlugin_v6

// v5 struct for backwards compatibility
typedef struct
{
  // Name of the network (mainly for logs)
  const char *name;
  // Initialize the network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int *ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  ncclResult_t (*connect)(int dev, void *handle, void **sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  ncclResult_t (*accept)(void *listenComm, void **recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void *comm, void *data, int size, int type, void **mhandle);
  ncclResult_t (*deregMr)(void *comm, void *mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void *sendComm, void *data, int size, int tag, void *mhandle, void **request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void *recvComm, int n, void **data, int *sizes, int *tags, void **mhandles, void **request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void *recvComm, int n, void **data, int *sizes, void **mhandles, void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void *request, int *done, int *sizes);
  // Close and free send/recv comm objects
  ncclResult_t (*closeSend)(void *sendComm);
  ncclResult_t (*closeRecv)(void *recvComm);
  ncclResult_t (*closeListen)(void *listenComm);
} ncclNet_v5_t;

// v5 struct for backwards compatibility
typedef struct
{
  // Name of the collective network (mainly for logs)
  const char *name;
  // Initialize the collective network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  ncclResult_t (*devices)(int *ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void *handles[], int nranks, int rank, void *listenComm, void **collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  ncclResult_t (*reduceSupport)(ncclDataType_t dataType, ncclRedOp_t redOp, int *supported);
  // Register/Deregister memory. Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void *collComm, void *data, int size, int type, void **mhandle);
  ncclResult_t (*deregMr)(void *collComm, void *mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  ncclResult_t (*iallreduce)(void *collComm, void *sendData, void *recvData, int count,
                             ncclDataType_t dataType, ncclRedOp_t redOp, void *sendMhandle, void *recvMhandle, void **request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void *collComm, void *data, int size, void *mhandle, void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void *request, int *done, int *size);
  // Close and free collective comm objects
  ncclResult_t (*closeColl)(void *collComm);
  ncclResult_t (*closeListen)(void *listenComm);
} ncclCollNet_v5_t;

// v4 struct for backwards compatibility
typedef struct
{
  char *name;     // Used mostly for logging.
  char *pciPath;  // Path to the PCI device in /sys.
  uint64_t guid;  // Unique identifier for the NIC chip. Important for
                  // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport; // NCCL_PTR_HOST or NCCL_PTR_HOST|NCCL_PTR_CUDA
  int speed;      // Port speed in Mbps.
  int port;       // Port number.
  int maxComms;   // Maximum number of comms we can create
} ncclNetProperties_v4_t;

// v4 struct for backwards compatibility
typedef struct
{
  // Name of the network (mainly for logs)
  const char *name;
  // Initialize the network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int *ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v4_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  ncclResult_t (*connect)(int dev, void *handle, void **sendComm);
  // Finalize connection establishment after remote peer has called connectHandle
  ncclResult_t (*accept)(void *listenComm, void **recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void *comm, void *data, int size, int type, void **mhandle);
  ncclResult_t (*deregMr)(void *comm, void *mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void *sendComm, void *data, int size, void *mhandle, void **request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void *recvComm, void *data, int size, void *mhandle, void **request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void *recvComm, void *data, int size, void *mhandle, void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void *request, int *done, int *size);
  // Close and free send/recv comm objects
  ncclResult_t (*closeSend)(void *sendComm);
  ncclResult_t (*closeRecv)(void *recvComm);
  ncclResult_t (*closeListen)(void *listenComm);
} ncclNet_v4_t;

// v4 struct for backwards compatibility
typedef struct
{
  // Name of the collective network (mainly for logs)
  const char *name;
  // Initialize the collective network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  ncclResult_t (*devices)(int *ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v4_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void *handles[], int nranks, int rank, void *listenComm, void **collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  ncclResult_t (*reduceSupport)(ncclDataType_t dataType, ncclRedOp_t redOp, int *supported);
  // Register/Deregister memory. Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void *collComm, void *data, int size, int type, void **mhandle);
  ncclResult_t (*deregMr)(void *collComm, void *mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  ncclResult_t (*iallreduce)(void *collComm, void *sendData, void *recvData, int count,
                             ncclDataType_t dataType, ncclRedOp_t redOp, void *sendMhandle, void *recvMhandle, void **request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void *collComm, void *data, int size, void *mhandle, void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void *request, int *done, int *size);
  // Close and free collective comm objects
  ncclResult_t (*closeColl)(void *collComm);
  ncclResult_t (*closeListen)(void *listenComm);
} ncclCollNet_v4_t;

#endif // end include guard
