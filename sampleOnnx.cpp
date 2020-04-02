#include <memory>
#include <chrono>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <string.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvOnnxConfig.h"
using namespace nvinfer1;
using namespace nvonnxparser;

#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

const int INFERENCE_BATCH = 5;
const int INPUT_C = 3;
const int INPUT_H = 224;
const int INPUT_W = 224;
const char* INPUT_NAME="data";

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
    public:
        void log(nvinfer1::ILogger::Severity severity, const char* msg) override
        {
            // suppress info-level messages
            if (severity == Severity::kVERBOSE) return;

            switch (severity)
            {
                case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
                case Severity::kERROR: std::cerr << "ERROR: "; break;
                case Severity::kWARNING: std::cerr << "WARNING: "; break;
                case Severity::kINFO: std::cerr << "INFO: "; break;
                default: std::cerr << "UNKNOWN: "; break;
            }
            std::cerr << msg << std::endl;
        }
} gLogger;

int main(int argc, char** argv)
{

    IBuilder* builder = createInferBuilder(gLogger);
    // For the explicit batch model which was onlu supported by onnxParser
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    auto parser = nvonnxparser::createParser(*network, gLogger);

    auto parsed = parser->parseFromFile("mobilenetv2.onnx", static_cast<int>(Logger::Severity::kINFO));
    if (!parsed)
    {
        return false;
    }

    // Modife the batch dimention to -1 which is dynamic, if you're not willing ro regenerate the model
    // This might not works for some onnx model generated with static tensor.
    auto input = network->getInput(0);
    input->setDimensions(Dims4{-1, INPUT_C, INPUT_H, INPUT_W});

    builder->setMaxBatchSize(10);
    // Create an optimization profile and set the dimension as below
    IBuilderConfig* config = builder->createBuilderConfig();
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(INPUT_NAME, OptProfileSelector::kMIN, Dims4(1, INPUT_C, INPUT_H, INPUT_W));
    profile->setDimensions(INPUT_NAME, OptProfileSelector::kOPT, Dims4(5, INPUT_C, INPUT_H, INPUT_W));
    profile->setDimensions(INPUT_NAME, OptProfileSelector::kMAX, Dims4(10, INPUT_C, INPUT_H, INPUT_W));

    config->addOptimizationProfile(profile);
    config->setMaxWorkspaceSize(1 << 30);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    auto context = engine->createExecutionContext();
    if (!context)
    {
        return false;
    }    
    
    int nbBindings = engine->getNbBindings();
    assert(nbBindings == 2);

    // Malloc Device Memory based on the inference batch 
    void* buffers[nbBindings];
    CHECK(cudaMalloc(&buffers[0], INFERENCE_BATCH*INPUT_C*INPUT_H*INPUT_W*sizeof(float)));
    CHECK(cudaMalloc(&buffers[1], INFERENCE_BATCH*1000*sizeof(float)));

    int N = INFERENCE_BATCH*INPUT_C*INPUT_H*INPUT_W;
    std::vector<float> data(N);
    for (int i = 0; i < N; ++i)
    {
        data[i] = rand() % 256;
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaMemcpyAsync(buffers[0], &data[0], N* sizeof(float), cudaMemcpyHostToDevice, stream));

    // Specific the shape before inference
    context->setBindingDimensions(0, Dims4(INFERENCE_BATCH, INPUT_C, INPUT_H, INPUT_W));

    auto tStart = std::chrono::high_resolution_clock::now();
    // Use execcuteV2 for dynamic shaoe mode
    bool status = context->executeV2(buffers);
    auto tEnd = std::chrono::high_resolution_clock::now();
    float totalHost = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
    std::cerr << "Inferencing Cost: " << totalHost << std::endl;
    if (!status)
    {
        return false;
    }

    engine->destroy();
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));

    return 0;
}
