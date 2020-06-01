#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>
#include <optional>

#ifndef _In_  
#define _In_ 
#endif

#ifndef _Out_ 
#define _Out_ 
#endif

struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() {
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

GLFWwindow* initWindow(_In_ int width, 
					   _In_ int height, 
					   _In_ const char* title);

void createInstance(_In_ const std::vector<const char*> validationLayers,
					_Out_ VkInstance *pInstance);

void setupDebugMessenger(_In_ VkInstance instance, 
						 _Out_ VkDebugUtilsMessengerEXT* pDebugMessenger);

QueueFamilyIndices findQueueFamilies(_In_ VkPhysicalDevice device,
									 _In_ VkSurfaceKHR surface);

void createSurface(_In_ VkInstance instance, 
				   _In_ GLFWwindow *pWindow, 
				   _Out_ VkSurfaceKHR* surface);

void pickPhysicalDevice(_In_ VkInstance instance,
						_In_ VkSurfaceKHR surface, 
						_In_ const std::vector<const char*> extensions, 
						_Out_ VkPhysicalDevice* pPhysicalDevice);

void createLogicalDevice_AndSetupQueues(_In_ VkPhysicalDevice device,
				_In_ VkSurfaceKHR surface,
				_In_ const std::vector<const char*> deviceExtensions,
				_In_ const std::vector<const char*> validationLayers,
				_Out_ VkDevice* logicalDevice,
				_Out_ VkQueue* graphicsQueue,
				_Out_ VkQueue* presentQueue);

void DestroyDebugUtilsMessengerEXT(_In_ VkInstance instance,
								   _In_ VkDebugUtilsMessengerEXT debugMessenger,
								   _In_ const VkAllocationCallbacks* pAllocator);

void createSwapChain(_In_ VkPhysicalDevice physicalDevice,
	_In_ VkDevice logicalDevice,
	_In_ VkSurfaceKHR surface,
	_In_ int screenWidth,
	_In_ int screenHeight,
	_Out_ VkSwapchainKHR* swapChain,
	_Out_ std::vector<VkImage>& swapChainImages,
	_Out_ VkFormat* swapChainImageFormat,
	_Out_ VkExtent2D* swapChainExtent);

VkImageView createImageView(_In_ VkDevice device,
	_In_ VkImage image,
	_In_ VkFormat format);

void createImageViews(_In_ VkDevice logicalDevice,
	_In_ std::vector<VkImage>& swapChainImages,
	_In_ VkFormat swapChainImageFormat,
	_Out_ std::vector<VkImageView>& swapChainImageViews);


