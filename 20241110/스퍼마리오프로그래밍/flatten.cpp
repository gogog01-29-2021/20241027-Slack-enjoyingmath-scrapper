#include <vector>
#include <iostream>
// 요약
// 이미지 상태를 1차원 벡터로 평탄화하여 저장하는 방법을 사용했습니다.
// 경험 구조체에는 평탄화된 상태와 다음 상태를 저장합니다.
// 평탄화된 상태를 복원하여 3차원 이미지로 변환하는 함수도 제공합니다.
// 이 방식은 CUDA에서 데이터를 처리할 때 메모리 접근 패턴을 단순화하고, 데이터 전송을 쉽게 관리할 수 있도록 도와줍니다.

// 경험 저장을 위한 구조체와 큐
struct Experience
{
    std::vector<float> state; // 평탄화된 이미지 상태
    int action;
    float reward;
    std::vector<float> nextState; // 평탄화된 다음 상태
    bool done;
};

std::deque<Experience> replayBuffer;

void storeExperience(const std::vector<std::vector<std::vector<float>>> &state, int action, float reward, const std::vector<std::vector<std::vector<float>>> &nextState, bool done)
{
    // 상태와 다음 상태를 평탄화하여 저장
    replayBuffer.push_back({flattenImage(state), action, reward, flattenImage(nextState), done});
    if (replayBuffer.size() > 10000)
    {
        replayBuffer.pop_front();
    }
}

// 이미지를 1차원 벡터로 평탄화하는 함수
std::vector<float> flattenImage(const std::vector<std::vector<std::vector<float>>> &image)
{
    std::vector<float> flatImage;
    for (const auto &channel : image)
    {
        for (const auto &row : channel)
        {
            flatImage.insert(flatImage.end(), row.begin(), row.end());
        }
    }
    return flatImage;
}

// 1차원 벡터를 3차원 이미지로 복원하는 함수
std::vector<std::vector<std::vector<float>>> restoreImage(const std::vector<float> &flatImage, int channels, int height, int width)
{
    std::vector<std::vector<std::vector<float>>> image(channels, std::vector<std::vector<float>>(height, std::vector<float>(width)));
    int idx = 0;
    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                image[c][h][w] = flatImage[idx++];
            }
        }
    }
    return image;
}

// 예제 사용
int main()
{
    // 3x3 크기의 예제 이미지 (채널 수는 2)
    std::vector<std::vector<std::vector<float>>> image = {
        {{1, 2, 3},
         {4, 5, 6},
         {7, 8, 9}},
        {{10, 11, 12},
         {13, 14, 15},
         {16, 17, 18}}};

    // 이미지 평탄화
    std::vector<float> flatImage = flattenImage(image);

    // 평탄화된 이미지 출력
    std::cout << "Flattened Image: ";
    for (float val : flatImage)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 이미지 복원
    std::vector<std::vector<std::vector<float>>> restoredImage = restoreImage(flatImage, 2, 3, 3);

    // 복원된 이미지 출력
    std::cout << "Restored Image:" << std::endl;
    for (const auto &channel : restoredImage)
    {
        for (const auto &row : channel)
        {
            for (float val : row)
            {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}
int main()
{
    // 예제 이미지 상태 (3채널, 84x84 크기)
    std::vector<std::vector<std::vector<float>>> state(3, std::vector<std::vector<float>>(84, std::vector<float>(84, 1.0f)));
    std::vector<std::vector<std::vector<float>>> nextState(3, std::vector<std::vector<float>>(84, std::vector<float>(84, 0.5f)));
    int action = 1;
    float reward = 1.0f;
    bool done = false;

    // 경험 저장
    storeExperience(state, action, reward, nextState, done);

    // 리플레이 버퍼의 첫 번째 경험 출력
    const Experience &exp = replayBuffer.front();
    std::cout << "Action: " << exp.action << ", Reward: " << exp.reward << ", Done: " << exp.done << std::endl;

    // 복원된 상태 출력
    std::vector<std::vector<std::vector<float>>> restoredState = restoreImage(exp.state, 3, 84, 84);
    std::cout << "Restored State[0][0][0]: " << restoredState[0][0][0] << std::endl;

    return 0;
}
