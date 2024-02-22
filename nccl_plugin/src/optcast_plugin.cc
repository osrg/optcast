/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2024, The Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 ************************************************************************/

#include <sys/time.h>

#include "core.h"
#include "utils.h"
#include "p2p_plugin.h"

#include <atomic>
#include <iostream>
#include <vector>

extern ncclNet_v6_t ncclNetPlugin_v6;
extern ncclNet_v5_t ncclNetPlugin_v5;

#define NCCL_PLUGIN_SYMBOL ncclNetPlugin_v6

int ncclNSharpDevs = -1;

struct optcastRequest
{
  int requestType;
  void *flushRequest;
  int size;
  int used;
  void *handler;
  void **srequests;
  void **rrequests;
  int nreqs;
  int idx;
};

struct serverHandler
{
  void *rcomm;
  void *scomm;
};

struct optcastComm
{
  bool bypass;
  std::atomic<uint64_t> cursor;
  std::vector<serverHandler> handlers;
  int nsplit;
};

struct optcastMr
{
  void *rMr;
  void *sMr;
};

enum optcastRequestType
{
  NCCL_OPTCAST_REQ_COLL,
  NCCL_OPTCAST_REQ_IFLUSH,
};

struct optcastListenComm
{
  int dev;
  void *listenCommP2P;
};

struct optcastCollComm
{
  int rank;
  int nranks;
  void *recvComm;
  void *sendComm;
  struct optcastComm *optcastComm;
  struct optcastRequest *reqs;
};

struct optcastMemHandle
{
  struct optcastMr *mr; // for optcast
  void *ncclIbMr;
  int type;
};

static ncclResult_t optcastConnect(int dev, const std::string &addr, int port, serverHandler *handler)
{
  // connected to addr:port
  int socket_fd;
  struct sockaddr_in serv_addr;

  // Create socket file descriptor
  if ((socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
  {
    WARN("Failed to create a socket\n");
    return ncclInternalError;
  }

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);

  // Convert IPv4 and IPv6 addresses from text to binary form
  if (inet_pton(AF_INET, addr.c_str(), &serv_addr.sin_addr) <= 0)
  {
    WARN("Invalid address\n");
    return ncclInternalError;
  }

  // Connect to the server
  if (connect(socket_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
  {
    // bufferstring
    char buffer[256];
    sprintf(buffer, "Failed to connect to server: %s:%d", addr.c_str(), port);
    WARN("%s", buffer);
    return ncclInternalError;
  }

  // Receive the size of the incoming message
  int msg_size;
  if (recv(socket_fd, &msg_size, sizeof(msg_size), 0) < 0)
  {
    return ncclInternalError;
  }

  // Receive the incoming message
  std::vector<char> connect_handle(msg_size);
  if (recv(socket_fd, connect_handle.data(), connect_handle.size(), 0) < 0)
  {
    return ncclInternalError;
  }

  std::vector<char> listen_handle(NCCL_NET_HANDLE_MAXSIZE);
  void *lcomm;

  NCCLCHECK(NCCL_PLUGIN_SYMBOL.listen(dev, listen_handle.data(), &lcomm));

  msg_size = listen_handle.size();
  send(socket_fd, &msg_size, sizeof(msg_size), 0);
  send(socket_fd, listen_handle.data(), msg_size, 0);

  void *scomm = nullptr;
  void *rcomm = nullptr;
  while (scomm == nullptr || rcomm == nullptr)
  {
    if (scomm == nullptr)
    {
      NCCLCHECK(NCCL_PLUGIN_SYMBOL.connect(dev, connect_handle.data(), &scomm));
    }
    if (rcomm == nullptr)
    {
      NCCLCHECK(NCCL_PLUGIN_SYMBOL.accept(lcomm, &rcomm));
    }
  }
  INFO(NCCL_ALL, "connected to the reduction server: %s:%d", addr.c_str(), port);

  handler->rcomm = rcomm;
  handler->scomm = scomm;

  NCCLCHECK(NCCL_PLUGIN_SYMBOL.closeListen(lcomm));

  return ncclSuccess;
}

static ncclResult_t optcastInit(int dev, int nranks, int rank, optcastComm **comm, optcastRequest *reqs, int nreqs)
{
  char *s = getenv("OPTCAST_REDUCTION_SERVERS");
  if (s == nullptr)
  {
    return ncclInternalError;
  }

  auto oComm = new optcastComm();

  char *c = getenv("OPTCAST_BYPASS");
  oComm->bypass = c == nullptr ? false : true;

  c = getenv("OPTCAST_SPLIT");
  oComm->nsplit = c == nullptr ? 1 : std::stoi(c);

  if (oComm->bypass)
  {
    *comm = oComm;
    INFO(NCCL_ALL, "optcast_init done (bypass mode)");
    return ncclSuccess;
  }

  // split addr by ,
  std::string ss(s);
  std::vector<std::string> servers;
  std::string delimiter = ",";
  size_t pos = 0;
  while ((pos = ss.find(delimiter)) != std::string::npos)
  {
    auto token = ss.substr(0, pos);
    servers.push_back(token);
    ss.erase(0, pos + delimiter.length());
  }
  servers.push_back(ss);

  ncclResult_t ret;
  for (auto &server : servers)
  {
    // split server by : to get addr and port
    std::string delimiter = ":";
    size_t pos = 0;
    pos = server.find(delimiter);
    if (pos == std::string::npos)
    {
      goto end;
    }
    auto addr = server.substr(0, pos);
    auto port = std::stoi(server.substr(pos + delimiter.length()));

    serverHandler handler;
    NCCLCHECKGOTO(optcastConnect(dev, addr, port, &handler), ret, end);
    oComm->handlers.push_back(handler);
  }

  for (int i = 0; i < nreqs; i++)
  {
    auto req = (optcastRequest *)reqs + i;
    req->srequests = (void **)malloc(sizeof(void *) * oComm->handlers.size());
    req->rrequests = (void **)malloc(sizeof(void *) * oComm->handlers.size());
  }

  *comm = oComm;
  INFO(NCCL_ALL, "optcast_init done");

  return ncclSuccess;
end:
  delete oComm;
  return ncclInternalError;
}

static ncclResult_t optcastAllreduce(optcastComm *oComm, optcastRequest *req, bool isHalf, void *sendData, void *recvData, void *sendMhandle, void *recvMhandle, int count)
{
  if (oComm->bypass)
  {
    req->nreqs = 0;
    return ncclSuccess;
  }
  int tag = 0x69;
  int size = isHalf ? count * 2 : count * 4;
  int nsplit = oComm->nsplit;
  int nhandlers = oComm->handlers.size();
  auto idx = oComm->cursor.fetch_add(nsplit) % nhandlers;

  if (size % nsplit != 0)
  {
    WARN("size(%d) is not divisible by nsplit(%d)", size, nsplit);
    return ncclInvalidUsage;
  }
  int csize = size / nsplit;
  auto sMr = (optcastMr *)sendMhandle;
  auto rMr = (optcastMr *)recvMhandle;

  TRACE(NCCL_ALL, "req(%p)/idx(%d) allreduce start", req, idx);

  for (int i = 0; i < nsplit; i++)
  {
    auto &h = oComm->handlers[(idx + i) % nhandlers];
    void *srequest = nullptr, *rrequest = nullptr;
    while (srequest == nullptr || rrequest == nullptr)
    {
      if (srequest == nullptr)
      {
        NCCLCHECK(NCCL_PLUGIN_SYMBOL.isend(h.scomm, (char *)sendData + i * csize, csize, tag, sMr->sMr, &srequest));
      }
      if (rrequest == nullptr)
      {
        void *r = (char *)recvData + i * csize;
        NCCLCHECK(NCCL_PLUGIN_SYMBOL.irecv(h.rcomm, 1, &r, &csize, &tag, &rMr->rMr, &rrequest));
      }
    }
    req->srequests[i] = srequest;
    req->rrequests[i] = rrequest;
  }

  TRACE(NCCL_ALL, "req(%p)/idx(%d) allreduce requested size: %d, csize: %d, nsplit: %d", req, idx, size, csize, nsplit);

  req->nreqs = nsplit;
  req->idx = idx;
  return ncclSuccess;
}

static int optcastTest(optcastRequest *req)
{
  for (int i = 0; i < req->nreqs; i++)
  {
    int done = 0;
    if (req->srequests[i] == nullptr)
    {
      continue;
    }
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.test(req->srequests[i], &done, nullptr));
    if (done == 0)
    {
      return -1;
    }
    req->srequests[i] = nullptr;
    if (i == req->nreqs - 1)
    {
      TRACE(NCCL_ALL, "req(%p)/idx(%d) send done", req, req->idx);
    }
  }
  for (int i = 0; i < req->nreqs; i++)
  {
    int done = 0;
    if (req->rrequests[i] == nullptr)
    {
      continue;
    }
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.test(req->rrequests[i], &done, nullptr));
    if (done == 0)
    {
      return -1;
    }
    req->rrequests[i] = nullptr;
  }
  TRACE(NCCL_ALL, "req(%p)/idx(%d) recv done", req, req->idx);
  return 0;
}

static ncclResult_t optcastClose(void *comm)
{
  auto oComm = (optcastComm *)comm;
  for (auto &handler : oComm->handlers)
  {
    NCCL_PLUGIN_SYMBOL.closeSend(handler.scomm);
    NCCL_PLUGIN_SYMBOL.closeRecv(handler.rcomm);
  }
  delete oComm;
  return ncclSuccess;
}

static ncclResult_t optcastRegMr(void *comm, void *data, int size, int type, optcastMr **mhandle)
{
  auto oComm = (optcastComm *)comm;
  auto mr = new optcastMr();
  // FIXME: all rcomm and scomm must use the same ib device
  // the first call of regMr will register the memory, the rest will just return the same mr from cache
  for (auto &handler : oComm->handlers)
  {
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.regMr(handler.rcomm, data, size, type, &mr->rMr));
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.regMr(handler.scomm, data, size, type, &mr->sMr));
  }
  *mhandle = mr;
  return ncclSuccess;
}

static ncclResult_t optcastDeregMr(void *comm, optcastMr *mr)
{
  auto oComm = (optcastComm *)comm;
  for (auto &handler : oComm->handlers)
  {
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.deregMr(handler.rcomm, mr->rMr));
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.deregMr(handler.scomm, mr->sMr));
  }
  delete mr;
  return ncclSuccess;
}

static ncclResult_t ncclOptcastInit(ncclDebugLogger_t logFunction)
{
  struct timeval tval;
  gettimeofday(&tval, nullptr);
  srand((int)tval.tv_usec);

  ncclResult_t v = NCCL_PLUGIN_SYMBOL.init(logFunction);
  INFO(NCCL_INIT, "ncclOptcastInit error: %d", v);

  return v;
}

static ncclResult_t ncclOptcastDevices(int *ndev)
{
  *ndev = ncclNSharpDevs;
  return ncclSuccess;
}

static ncclResult_t ncclOptcastGetProperties_v6(int dev, ncclNetProperties_v6_t *props)
{
  return NCCL_PLUGIN_SYMBOL.getProperties(dev, props);
}

static ncclResult_t ncclOptcastGetProperties_v5(int dev, ncclNetProperties_v5_t *props)
{
  return ncclNetPlugin_v5.getProperties(dev, props);
}

static ncclResult_t ncclOptcastListen(int dev, void *opaqueHandle, void **listenComm)
{
  struct optcastListenComm *lComm;
  ncclResult_t status;

  NCCLCHECK(ncclIbMalloc((void **)&lComm, sizeof(struct optcastListenComm)));
  status = NCCL_PLUGIN_SYMBOL.listen(dev, opaqueHandle, &lComm->listenCommP2P);
  lComm->dev = dev;
  *listenComm = lComm;
  return status;
}

static ncclResult_t ncclOptcastConnect(void *handles[], int nranks, int rank, void *listenComm, void **collComm)
{
  struct optcastListenComm *lComm = (struct optcastListenComm *)listenComm;
  struct optcastCollComm *cComm;

  NCCLCHECK(ncclIbMalloc((void **)&cComm, sizeof(struct optcastCollComm)));
  NCCLCHECK(ncclIbMalloc((void **)&cComm->reqs, sizeof(struct optcastRequest) * MAX_REQUESTS));
  NCCLCHECK(optcastInit(lComm->dev, nranks, rank, &cComm->optcastComm, cComm->reqs, MAX_REQUESTS));

  cComm->nranks = nranks;
  cComm->rank = rank;
  if (cComm->rank == -1)
  {
    WARN("Could not determine my rank\n");
    return ncclInternalError;
  }
  int next = (cComm->rank + 1) % nranks;
  do
  {
    if (cComm->sendComm == nullptr)
      NCCLCHECK(NCCL_PLUGIN_SYMBOL.connect(lComm->dev, handles[next], &cComm->sendComm));
    if (cComm->recvComm == nullptr)
      NCCLCHECK(NCCL_PLUGIN_SYMBOL.accept(lComm->listenCommP2P, &cComm->recvComm)); // From prev
  } while (cComm->sendComm == nullptr || cComm->recvComm == nullptr);

  char devName[MAXNAMESIZE];
  ncclNetProperties_v6_t prop;
  ncclOptcastGetProperties_v6(lComm->dev, &prop);
  snprintf(devName, MAXNAMESIZE, "%s:%d", prop.name, prop.port);
  INFO(NCCL_ALL, "Optcast rank %d/%d initialized on %s(%d)", cComm->rank, nranks, devName, lComm->dev);

  *collComm = cComm;
  return ncclSuccess;
}

static ncclResult_t ncclOptcastReduceSupport(ncclDataType_t dataType, ncclRedOp_t redOp, int *supported)
{
  if (dataType != ncclFloat32 && dataType != ncclFloat16)
  {
    *supported = 0;
    return ncclSuccess;
  }
  if (redOp != ncclSum)
  {
    *supported = 0;
    return ncclSuccess;
  }
  *supported = 1;
  return ncclSuccess;
}

static ncclResult_t ncclOptcastRegMrDmaBuf(void *collComm, void *data, size_t size, int type, uint64_t offset, int fd, void **mhandle)
{
  return ncclInternalError;
}

static ncclResult_t ncclOptcastRegMr(void *collComm, void *data, int size, int type, void **mhandle)
{
  struct optcastCollComm *cComm = (struct optcastCollComm *)collComm;

  struct optcastMemHandle *mh;
  NCCLCHECK(ncclIbMalloc((void **)&mh, sizeof(struct optcastMemHandle)));

  mh->type = type;

  NCCLCHECK((ncclResult_t)optcastRegMr(cComm->optcastComm, data, size, type, &mh->mr));
  NCCLCHECK(NCCL_PLUGIN_SYMBOL.regMr(cComm->recvComm, data, size, type, &mh->ncclIbMr));

  *mhandle = mh;
  return ncclSuccess;
}

static ncclResult_t ncclOptcastDeregMr(void *collComm, void *mhandle)
{
  struct optcastCollComm *cComm = (struct optcastCollComm *)collComm;
  struct optcastMemHandle *mh = (struct optcastMemHandle *)mhandle;

  NCCLCHECK((ncclResult_t)optcastDeregMr(cComm->optcastComm, mh->mr));
  NCCLCHECK(NCCL_PLUGIN_SYMBOL.deregMr(cComm->recvComm, mh->ncclIbMr));

  free(mh);
  return ncclSuccess;
}

static ncclResult_t ncclOptcastGetRequest(struct optcastRequest *reqs, struct optcastRequest **req)
{
  for (int i = 0; i < MAX_REQUESTS; i++)
  {
    struct optcastRequest *r = reqs + i;
    if (r->used == 0)
    {
      r->used = 1;
      r->flushRequest = nullptr;
      r->size = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("Optcast: unable to allocate request");
  *req = nullptr;
  return ncclInternalError;
}

static ncclResult_t ncclOptcastIallreduce(void *collComm, void *sendData, void *recvData, int count,
                                          ncclDataType_t dataType, ncclRedOp_t redOp, void *sendMhandle, void *recvMhandle, void **request)
{
  struct optcastCollComm *cComm = (struct optcastCollComm *)collComm;
  struct optcastMemHandle *sMh = (struct optcastMemHandle *)sendMhandle;
  struct optcastMemHandle *rMh = (struct optcastMemHandle *)recvMhandle;

  if (dataType != ncclFloat32 && dataType != ncclFloat16)
  {
    WARN("Optcast: unsupported data type\n");
    return ncclInternalError;
  }

  if (redOp != ncclSum)
  {
    WARN("Optcast: unsupported reduce operation\n");
    return ncclInternalError;
  }

  struct optcastRequest *req;
  NCCLCHECK(ncclOptcastGetRequest(cComm->reqs, &req));
  NCCLCHECK(optcastAllreduce(cComm->optcastComm, req, dataType == ncclFloat16, sendData, recvData, sMh->mr, rMh->mr, count));

  req->requestType = NCCL_OPTCAST_REQ_COLL;
  *request = req;
  return ncclSuccess;
}

static ncclResult_t ncclOptcastIflush(void *collComm, void *data, int size, void *mhandle, void **request)
{
  struct optcastCollComm *cComm = (struct optcastCollComm *)collComm;
  struct optcastMemHandle *mh = (struct optcastMemHandle *)mhandle;
  struct optcastRequest *req;

  NCCLCHECK(ncclOptcastGetRequest(cComm->reqs, &req));
  req->requestType = NCCL_OPTCAST_REQ_IFLUSH;
  NCCL_PLUGIN_SYMBOL.iflush(cComm->recvComm, 1, &data, &size, &mh->ncclIbMr, &req->flushRequest);
  if (!req->flushRequest)
  {
    *request = nullptr;
    req->used = 0;
    return ncclSuccess;
  }

  *request = req;
  return ncclSuccess;
}

static ncclResult_t ncclOptcastTest(void *request, int *done, int *size)
{
  struct optcastRequest *req = (struct optcastRequest *)request;

  if (req->requestType == NCCL_OPTCAST_REQ_IFLUSH)
  {
    NCCL_PLUGIN_SYMBOL.test(req->flushRequest, done, size);
    if (*done == 1)
    {
      req->used = 0;
    }
    return ncclSuccess;
  }

  if (optcastTest(req) == 0)
  {
    *done = 1;
    *size = req->size;
    req->used = 0;
  }
  else
  {
    *done = 0;
  }

  return ncclSuccess;
}

static ncclResult_t ncclOptcastCloseColl(void *collComm)
{
  struct optcastCollComm *cComm = (struct optcastCollComm *)collComm;

  NCCLCHECK(NCCL_PLUGIN_SYMBOL.closeRecv(cComm->recvComm));
  NCCLCHECK(NCCL_PLUGIN_SYMBOL.closeSend(cComm->sendComm));
  NCCLCHECK(optcastClose(cComm->optcastComm));

  free(cComm);
  return ncclSuccess;
}

static ncclResult_t ncclOptcastCloseListen(void *listenComm)
{
  struct optcastListenComm *lComm = (struct optcastListenComm *)listenComm;
  ncclResult_t status;

  status = NCCL_PLUGIN_SYMBOL.closeListen(lComm->listenCommP2P);
  free(listenComm);
  return status;
}

ncclCollNet_v6_t ncclCollNetPlugin_v6 = {
    "Optcast",
    ncclOptcastInit,
    ncclOptcastDevices,
    ncclOptcastGetProperties_v6,
    ncclOptcastListen,
    ncclOptcastConnect,
    ncclOptcastReduceSupport,
    ncclOptcastRegMr,
    ncclOptcastRegMrDmaBuf,
    ncclOptcastDeregMr,
    ncclOptcastIallreduce,
    ncclOptcastIflush,
    ncclOptcastTest,
    ncclOptcastCloseColl,
    ncclOptcastCloseListen};

ncclCollNet_v5_t ncclCollNetPlugin_v5 = {
    "Optcast",
    ncclOptcastInit,
    ncclOptcastDevices,
    ncclOptcastGetProperties_v5,
    ncclOptcastListen,
    ncclOptcastConnect,
    ncclOptcastReduceSupport,
    ncclOptcastRegMr,
    ncclOptcastDeregMr,
    ncclOptcastIallreduce,
    ncclOptcastIflush,
    ncclOptcastTest,
    ncclOptcastCloseColl,
    ncclOptcastCloseListen};
