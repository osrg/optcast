AC_PREREQ([2.69])

AC_INIT([optcast-nccl-plugin], [0.1], [], [],[http://github.com/osrg/optcast])

AM_INIT_AUTOMAKE([1.10 foreign tar-ustar subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
AC_CONFIG_MACRO_DIR([m4])

AC_USE_SYSTEM_EXTENSIONS
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AC_PROG_SED
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_HEADER_STDC
LT_LIB_M

AC_ARG_ENABLE([debug],AS_HELP_STRING([--enable-debug], [Enable extra debugging code (default is NO).]),
              [], [enable_debug=no])

if test $enable_debug = yes; then
	AC_DEFINE([ENABLE_DEBUG], [1], [Enable debugging code])
	CFLAGS="$CFLAGS -O0 -g3 -Werror"
else
	CFLAGS="$CFLAGS -O3 -DNDEBUG"
fi

#check for cuda
AC_ARG_WITH([cuda],
            [AC_HELP_STRING([--with-cuda=PATH],
            [Path to non-standard CUDA installation])],
            [AS_IF([test -d $withval/lib64], [cuda_libdir="lib64"], [cuda_libdir="lib"])
             CFLAGS="-I$withval/include $CFLAGS"
             LDFLAGS="-L$withval/$cuda_libdir $LDFLAGS"],
            [CFLAGS="-I/usr/local/cuda/include $CFLAGS"
             LDFLAGS="-L/usr/local/cuda/$cuda_libdir $LDFLAGS"])

AC_CHECK_HEADER(       [cuda_runtime.h], [], [AC_MSG_FAILURE([CUDA runtime header files not found])])
AC_CHECK_LIB([cudart], [cudaMalloc],     [], [AC_MSG_FAILURE([CUDA runtime libs not found])])

#check for verbs
AC_ARG_WITH([verbs],
            [AC_HELP_STRING([--with-verbs(=DIR)],
            [Build Infiniband  support, adding DIR/include, DIR/lib, and DIR/lib64 to the search path for headers and libraries])],
            [CFLAGS="-I$with_verbs/include $CFLAGS"
             LDFLAGS="-L$with_verbs/lib64 -L$with_verbs/lib -libverbs $LDFLAGS"],
            [CFLAGS="-I/usr/include $CFLAGS"
             LDFLAGS="-L/usr/lib64 -L/usr/lib -libverbs $LDFLAGS"])

AC_CHECK_HEADER(        [infiniband/verbs.h],  [],[AC_MSG_FAILURE([ibverbs header files not found])])
AC_CHECK_LIB([ibverbs], [ibv_get_device_list], [],[AC_MSG_FAILURE([libibverbs not found]);])

AC_CHECK_DECLS([IBV_ACCESS_RELAXED_ORDERING, IBV_QPF_GRH_REQUIRED, ibv_reg_dmabuf_mr], [], [],
                    [[#include <infiniband/verbs.h>]])

#check for required headers
AC_CHECK_HEADERS([limits.h stdlib.h string.h unistd.h], [],
                 [AC_MSG_FAILURE([unable to find required headers])])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                ])
AC_OUTPUT

echo "Optcast NCCL Plugin has been configured."
