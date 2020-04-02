APP := sampleOnnx

CC := g++

CUDA_INSTALL_PATH ?= /usr/local/cuda-10.2
CUDNN_INSTALL_PATH ?= /root/cudnn-7.6
TRT_INSTALL_PATH ?= /root/TensorRT-6.0.1

SRCS := \
	sampleOnnx.cpp

OBJS := $(SRCS:.cpp=.o)

CPPFLAGS := \
	-std=c++11 -g\
	-I"$(TRT_INSTALL_PATH)/include" \
	-I"$(CUDA_INSTALL_PATH)/include" \
	-I"$(CUDNN_INSTALL_PATH)/include"

LDFLAGS := \
        -lnvonnxparser \
	-lnvparsers \
	-lnvinfer \
	-lcudart \
	-lcudnn \
	-L"$(TRT_INSTALL_PATH)/lib" \
	-L"$(CUDA_INSTALL_PATH)/lib64" \
	-L"$(CUDNN_INSTALL_PATH)/lib64"

all: $(APP)

%.o: %.cpp
	@echo "Compiling: $<"
	@ $(CC) $(CPPFLAGS) -c $<

$(APP): $(OBJS)
	@echo "Linking: $@"
	@ $(CC) -o $@ $(OBJS) $(CPPFLAGS) $(LDFLAGS)

clean:
	rm -rf $(APP) $(OBJS) *.ppm *.txt
