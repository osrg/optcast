PLUGIN_DIR ?= ../nccl_plugin

NVCC_GENCODE ?= "-gencode=arch=compute_70,code=sm_70" # speed up NCCL build

build: build-nccl build-nccl-plugin build-nccl-tests

install: install-nccl install-nccl-plugin

clean: clean-nccl clean-nccl-plugin clean-nccl-tests

build-nccl:
	$(MAKE) -C nccl -j src.build NVCC_GENCODE=$(NVCC_GENCODE)

clean-nccl:
	$(MAKE) -C nccl clean

install-nccl: build-nccl
	$(MAKE) -C nccl install

build-nccl-plugin: $(PLUGIN_DIR)/Makefile
	$(MAKE) -C $(PLUGIN_DIR) -j

install-nccl-plugin: build-nccl-plugin
	$(MAKE) -C $(PLUGIN_DIR) install

clean-nccl-plugin:
	$(MAKE) -C $(PLUGIN_DIR) clean

build-nccl-tests:
	$(MAKE) -C nccl-tests -j MPI=1 MPI_HOME=$(MPI_HOME)

clean-nccl-tests:
	$(MAKE) -C nccl-tests clean

$(PLUGIN_DIR)/Makefile:
	cd $(PLUGIN_DIR) && ./autogen.sh && ./configure --with-cuda=/usr/local/cuda
