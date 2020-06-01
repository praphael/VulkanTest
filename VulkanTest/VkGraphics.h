#pragma once

#include "VkSetup.h"

#include <array>
#include <glm/glm.hpp>

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription();

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions();
};


void createRenderPass(_In_ VkDevice device,
    _In_ VkFormat swapChainImageFormat,
    _Out_ VkRenderPass* renderPass);

VkShaderModule createShaderModule(_In_ VkDevice device,
    _In_ const std::vector<char>& code);

void createDescriptorSetLayout(_In_ VkDevice device,
    _Out_ VkDescriptorSetLayout* descriptorSetLayout);

void createGraphicsPipeline(_In_ VkDevice device,
    _In_ VkExtent2D swapChainExtent,
    _In_ VkRenderPass renderPass,
    _In_ VkDescriptorSetLayout descriptorSetLayout,
    _Out_ VkPipelineLayout* pipelineLayout,
    _Out_ VkPipeline* graphicsPipeline);

void createFramebuffers(_In_ VkDevice device,
    _In_ VkRenderPass renderPass,
    _In_ VkExtent2D swapChainExtent,
    _In_ std::vector<VkImageView> &swapChainImageViews,
    _Out_ std::vector<VkFramebuffer> &swapChainFramebuffers);

void createCommandPool(_In_ VkDevice device,
    _In_ VkPhysicalDevice physicalDevice,
    _In_ VkSurfaceKHR surface,
    _Out_ VkCommandPool* commandPool);

void createCommandBuffers(_In_ VkDevice device,
    _In_ VkRenderPass renderPass,
    _In_ VkPipeline graphicsPipeline,
    _In_ VkPipelineLayout pipelineLayout,
    _In_ std::vector<VkFramebuffer> &swapChainFramebuffers,
    _In_ VkExtent2D swapChainExtent,
    _In_ VkCommandPool commandPool,
    _In_ VkBuffer vertexBuffer,
    _In_ uint32_t numVertices,
    _In_ VkBuffer indexBuffer,
    _In_ uint32_t numIndices,
    _Out_ std::vector<VkDescriptorSet> &m_descriptorSets,
    _Out_ std::vector<VkCommandBuffer> &commandBuffers);
