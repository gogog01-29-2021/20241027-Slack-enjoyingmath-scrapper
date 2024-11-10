#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <deque>
#include <tuple>
#include <algorithm>
#include <random>
// 1) 타일 크기 정의:

// #define TILE_WIDTH 16: CUDA 커널에서 사용하는 타일의 크기를 정의합니다.
// 2) CUDA 커널 정의:

// conv2d: 2D 컨볼루션을 수행하는 CUDA 커널.
// computeQValues: Q 값을 계산하는 CUDA 커널.
// computeLoss: 손실 값을 계산하는 CUDA 커널.
// updateWeights: 원자적 연산을 사용하여 가중치를 업데이트하는 CUDA 커널.
// computeGradients: 가중치의 그래디언트를 계산하는 CUDA 커널.
// 3) CPU에서 2D 컨볼루션 레이어 실행:

// convolutionLayer: 입력 데이터를 사용하여 CUDA 커널을 실행하고 결과를 출력합니다.
// 4) 경험 저장을 위한 구조체와 큐:

// Experience: 상태, 행동, 보상, 다음 상태 및 완료 여부를 포함하는 구조체.
// replayBuffer: 경험을 저장하는 큐.
// 5) 경험을 리플레이 버퍼에 저장:

// storeExperience: 상태, 행동, 보상, 다음 상태 및 완료 여부를 리플레이 버퍼에 저장합니다.
// 6)학습 단계 수행:

// trainStep: 경험을 샘플링하고, Q 값을 계산하고, 손실을 계산하며, 그래디언트를 계산하고, 가중치를 업데이트하는 함수.
// 7) 타깃 네트워크 업데이트:

// updateTargetNetwork: 주기적으로 타깃 네트워크를 업데이트합니다.
// 8) 메인 학습 루프:

// main: 환경과 상호작용하며 주기적으로 모델을 학습시키고, 타깃 네트워크를 업데이트합니다.
// 요약
// 이 코드는 DQN(Deep Q-Network) 알고리즘을 C++와 CUDA를 사용하여 구현한 것입니다.
// 경험을 리플레이 버퍼에 저장하고, CUDA 커널을 사용하여 Q 값을 계산하고 손실을 계산하며, 원자적 연산을 통해 가중치를 업데이트합니다.
// 학습 루프에서 주기적으로 모델을 학습시키고 타깃 네트워크를 업데이트합니다.

// 타일 크기 정의
#define TILE_WIDTH 16

// 2D 컨볼루션 CUDA 커널
__global__ void conv2d(const float *input, const float *kernel, float *output, int inputWidth, int inputHeight, int kernelWidth, int kernelHeight, int outputWidth, int outputHeight)
{
    // 각 스레드의 위치 계산
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // 유효한 출력 위치인지 확인
    if (row < outputHeight && col < outputWidth)
    {
        float value = 0.0f;
        for (int m = 0; m < kernelHeight; ++m)
        {
            for (int n = 0; n < kernelWidth; ++n)
            {
                int inputRow = row + m;
                int inputCol = col + n;
                value += input[inputRow * inputWidth + inputCol] * kernel[m * kernelWidth + n];
            }
        }
        output[row * outputWidth + col] = value;
    }
}

// Q 값 계산을 위한 CUDA 커널
__global__ void computeQValues(const float *states, const float *weights, float *qValues, int numStates, int stateSize, int numActions)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numStates)
    {
        for (int action = 0; action < numActions; ++action)
        {
            float qValue = 0.0f;
            for (int i = 0; i < stateSize; ++i)
            {
                qValue += states[idx * stateSize + i] * weights[action * stateSize + i];
            }
            qValues[idx * numActions + action] = qValue;
        }
    }
}

// 손실 계산을 위한 CUDA 커널
__global__ void computeLoss(const float *qValues, const int *actions, const float *rewards, const float *nextQValues, const bool *dones, float *losses, int batchSize, int numActions, float gamma)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize)
    {
        int action = actions[idx];
        float maxNextQ = 0.0f;
        for (int a = 0; a < numActions; ++a)
        {
            if (nextQValues[idx * numActions + a] > maxNextQ)
            {
                maxNextQ = nextQValues[idx * numActions + a];
            }
        }
        float target = rewards[idx] + (dones[idx] ? 0.0f : gamma * maxNextQ);
        losses[idx] = (qValues[idx * numActions + action] - target) * (qValues[idx * numActions + action] - target);
    }
}

// CPU에서 2D 컨볼루션 레이어 실행
void convolutionLayer(const std::vector<float> &input, const std::vector<float> &kernel, std::vector<float> &output, int inputWidth, int inputHeight, int kernelWidth, int kernelHeight, int outputWidth, int outputHeight)
{
    float *d_input, *d_kernel, *d_output;
    size_t inputSize = inputWidth * inputHeight * sizeof(float);
    size_t kernelSize = kernelWidth * kernelHeight * sizeof(float);
    size_t outputSize = outputWidth * outputHeight * sizeof(float);

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernel, kernelSize);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_input, input.data(), inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), kernelSize, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((outputWidth + TILE_WIDTH - 1) / TILE_WIDTH, (outputHeight + TILE_WIDTH - 1) / TILE_WIDTH);

    conv2d<<<dimGrid, dimBlock>>>(d_input, d_kernel, d_output, inputWidth, inputHeight, kernelWidth, kernelHeight, outputWidth, outputHeight);

    cudaMemcpy(output.data(), d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

// 경험 저장을 위한 구조체와 큐
struct Experience
{
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> nextState;
    bool done;
};

std::deque<Experience> replayBuffer;

// 경험을 리플레이 버퍼에 저장
void storeExperience(const std::vector<float> &state, int action, float reward, const std::vector<float> &nextState, bool done)
{
    replayBuffer.push_back({state, action, reward, nextState, done});
    if (replayBuffer.size() > 10000)
    {
        replayBuffer.pop_front();
    }
}

// 원자적 연산을 통해 가중치를 업데이트하는 CUDA 커널
__global__ void updateWeights(float *weights, const float *gradients, float learningRate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        atomicAdd(&weights[idx], -learningRate * gradients[idx]);
    }
}

// 가중치의 그래디언트를 계산하는 CUDA 커널
__global__ void computeGradients(const float *qValues, float *gradients, const float *states, int batchSize, int stateSize, int numActions)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * stateSize)
    {
        int stateIdx = idx / stateSize;
        int featureIdx = idx % stateSize;
        for (int action = 0; action < numActions; ++action)
        {
            atomicAdd(&gradients[action * stateSize + featureIdx], qValues[stateIdx * numActions + action] * states[stateIdx * stateSize + featureIdx]);
        }
    }
}

// 학습 단계 수행
void trainStep(float *d_states, float *d_nextStates, int *d_actions, float *d_rewards, bool *d_dones, float *d_weights, float *d_nextWeights, float *d_gradients, int batchSize, int stateSize, int numActions, float gamma, float learningRate)
{
    // 경험을 랜덤으로 샘플링하여 배치 생성
    std::vector<Experience> batch(batchSize);
    std::sample(replayBuffer.begin(), replayBuffer.end(), batch.begin(), batchSize, std::mt19937{std::random_device{}()});

    // 배치를 벡터로 변환
    std::vector<float> states(batchSize * stateSize);
    std::vector<float> nextStates(batchSize * stateSize);
    std::vector<int> actions(batchSize);
    std::vector<float> rewards(batchSize);
    std::vector<bool> dones(batchSize);

    for (int i = 0; i < batchSize; ++i)
    {
        states.insert(states.end(), batch[i].state.begin(), batch[i].state.end());
        nextStates.insert(nextStates.end(), batch[i].nextState.begin(), batch[i].nextState.end());
        actions[i] = batch[i].action;
        rewards[i] = batch[i].reward;
        dones[i] = batch[i].done;
    }

    // CUDA 메모리 할당
    float *d_qValues, *d_nextQValues, *d_losses;
    cudaMalloc(&d_qValues, batchSize * numActions * sizeof(float));
    cudaMalloc(&d_nextQValues, batchSize * numActions * sizeof(float));
    cudaMalloc(&d_losses, batchSize * sizeof(float));
    cudaMalloc(&d_gradients, stateSize * numActions * sizeof(float)); // 그래디언트 메모리 할당

    dim3 dimBlock(256);
    dim3 dimGrid((batchSize + 255) / 256);

    // 호스트 데이터를 디바이스 메모리로 복사
    cudaMemcpy(d_states, states.data(), states.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nextStates, nextStates.data(), nextStates.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_actions, actions.data(), actions.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rewards, rewards.data(), rewards.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dones, dones.data(), dones.size() * sizeof(bool), cudaMemcpyHostToDevice);

    // Q 값 계산
    computeQValues<<<dimGrid, dimBlock>>>(d_states, d_weights, d_qValues, batchSize, stateSize, numActions);
    computeQValues<<<dimGrid, dimBlock>>>(d_nextStates, d_nextWeights, d_nextQValues, batchSize, stateSize, numActions);

    // 손실 계산
    computeLoss<<<dimGrid, dimBlock>>>(d_qValues, d_actions, d_rewards, d_nextQValues, d_dones, d_losses, batchSize, numActions, gamma);

    // 손실 값 가져오기
    float *h_losses = new float[batchSize];
    cudaMemcpy(h_losses, d_losses, batchSize * sizeof(float), cudaMemcpyDeviceToHost);

    float loss = std::accumulate(h_losses, h_losses + batchSize, 0.0f) / batchSize;
    std::cout << "Loss: " << loss << std::endl;

    // 그래디언트 계산
    computeGradients<<<dimGrid, dimBlock>>>(d_qValues, d_gradients, d_states, batchSize, stateSize, numActions);

    // 가중치 업데이트
    updateWeights<<<dimGrid, dimBlock>>>(d_weights, d_gradients, learningRate, stateSize * numActions);

    // 메모리 해제
    delete[] h_losses;
    cudaFree(d_qValues);
    cudaFree(d_nextQValues);
    cudaFree(d_losses);
    cudaFree(d_gradients); // 그래디언트 메모리 해제
}

// 타깃 네트워크 업데이트 (GPU)
// 타깃 네트워크를 주기적으로 업데이트합니다.
void updateTargetNetwork(float *d_weights, float *d_nextWeights, int weightSize)
{
    cudaMemcpy(d_nextWeights, d_weights, weightSize * sizeof(float), cudaMemcpyDeviceToDevice);
}

// 메인 학습 루프 (CPU와 GPU)
// 환경과 상호작용하면서 주기적으로 모델을 학습시키고, 타깃 네트워크를 업데이트합니다.
int main()
{
    // 필요한 변수 및 메모리 할당
    int numEpisodes = 1000;
    int updateTargetEvery = 10;
    int batchSize = 32;
    int stateSize = 84 * 84; // 예: 84x84 이미지
    int numActions = 6;      // 예: 가능한 행동 수
    float gamma = 0.99;
    float learningRate = 0.001;

    // Mario 환경 초기화

    // 모델 및 타깃 모델 가중치 초기화
    float *d_weights, *d_nextWeights;
    cudaMalloc(&d_weights, stateSize * numActions * sizeof(float));
    cudaMalloc(&d_nextWeights, stateSize * numActions * sizeof(float));

    float *d_states, *d_nextStates, *d_rewards, *d_gradients;
    int *d_actions;
    bool *d_dones;

    cudaMalloc(&d_states, batchSize * stateSize * sizeof(float));
    cudaMalloc(&d_nextStates, batchSize * stateSize * sizeof(float));
    cudaMalloc(&d_rewards, batchSize * sizeof(float));
    cudaMalloc(&d_actions, batchSize * sizeof(int));
    cudaMalloc(&d_dones, batchSize * sizeof(bool));
    cudaMalloc(&d_gradients, stateSize * numActions * sizeof(float));

    for (int episode = 0; episode < numEpisodes; ++episode)
    {
        // 상태 초기화 및 변환
        bool done = false;
        while (!done)
        {
            // 에이전트 행동 선택 및 환경과 상호작용

            // 경험 저장

            // 학습 단계 수행
            trainStep(d_states, d_nextStates, d_actions, d_rewards, d_dones, d_weights, d_nextWeights, d_gradients, batchSize, stateSize, numActions, gamma, learningRate);

            // 타깃 네트워크 주기적 업데이트
            if (episode % updateTargetEvery == 0)
            {
                updateTargetNetwork(d_weights, d_nextWeights, stateSize * numActions);
            }
        }

        std::cout << "Episode " << episode << " complete" << std::endl;
    }

    // 메모리 해제 및 종료
    cudaFree(d_weights);
    cudaFree(d_nextWeights);
    cudaFree(d_states);
    cudaFree(d_nextStates);
    cudaFree(d_rewards);
    cudaFree(d_actions);
    cudaFree(d_dones);
    cudaFree(d_gradients);

    return 0;
}
