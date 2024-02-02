lib_LTLIBRARIES     = libnccl-net.la

libnccl_net_la_CPPFLAGS = -I$(top_srcdir)/include
libnccl_net_la_CFLAGS   = $(CFLAGS) -DENABLE_TRACE
libnccl_net_la_CXXFLAGS = $(CFLAGS) -DENABLE_TRACE -std=c++17
libnccl_net_la_LIBADD   = -lcudart_static
libnccl_net_la_LDFLAGS  = $(LDFLAGS)

libnccl_net_la_SOURCES = \
	ibvwrap.c \
	utils.c	\
	param.c \
	socket.c \
	socket_plugin.c \
	p2p_plugin.c \
	ib_plugin.c \
	optcast_plugin.cc
