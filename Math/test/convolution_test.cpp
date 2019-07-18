#include "convolution_test.h"

#include "..\math\convolution.h"
#include "..\utils\print_tensor.h"

void test_convolution()
{
	const int rank = 3;

	const int input_channels = 2;
	const int output_channels = 3;

	// Input
	const int input_depth = 3;
	const int input_height = 4;
	const int input_width = 5;
	const int input_size = input_channels * input_depth * input_height * input_width;

	int input_dimensions[rank + 1] = { input_channels, input_depth ,input_height, input_width };
	int input_strides[rank + 1] = { 0 };

	calculate_strides(input_dimensions, input_strides, rank + 1);

	// Initialize Input
	float inputs[input_size] = { 0 };
	for (size_t i = 0; i < input_size ; i++)
	{
		inputs[i] = i + 1.0f;
	}

	// Kernel
	const int kernel_depth = 3;
	const int kernel_height = 2;
	const int kernel_width = 3;
	const int kernel_size = output_channels * input_channels * kernel_depth * kernel_height * kernel_width;

	int kernel_dimensions[rank + 2] = { output_channels, input_channels, kernel_depth ,kernel_height, kernel_width };
	int kernel_strides[rank + 2] = { 0 };

	calculate_strides(kernel_dimensions, kernel_strides, rank + 2);

	// Initialize Kernel
	float kernels[kernel_size] = { 0 };
	for (size_t i = 0; i < kernel_size; i++)
	{
		kernels[i] = i + 1.0f;
	}

	// Strides
    const int stride_depth = 1;
	const int stride_height = 1;
	const int stride_width = 1;

	int stride_dimensions[rank] = { stride_depth, stride_height, stride_width };

	// Padding
	const int padding_depth = 1;
	const int padding_height = 0;
	const int padding_width = 0;

	int padding_dimensions[rank] = { padding_depth, padding_height, padding_width };

	// Dilation
	const int dilation_depth = 1;
	const int dilation_height = 1;
	const int dilation_width = 1;

	int dilation_dimensions[rank] = { dilation_depth, dilation_height,dilation_width };

	// Outputs

	const int output_depth  = int((input_depth  + 2 * padding_depth  - dilation_depth  * (kernel_depth - 1) - 1) / stride_depth) + 1;
	const int output_height = int((input_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) / stride_height) + 1;
	const int output_width  = int((input_width  + 2 * padding_width  - dilation_width  * (kernel_width - 1) - 1) / stride_width) + 1;
	const int output_size = output_channels * output_depth * output_height * output_width;

	int output_dimensions[rank + 1] = { output_channels, output_depth, output_height, output_width };
	int output_strides[rank + 1] = { 0 };

	calculate_strides(output_dimensions, output_strides, rank+1);

	std::cout << "==================================================================" << "\n";
	std::cout << "Input Tensor" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs, rank + 1, input_dimensions, input_strides);

	std::cout << "==================================================================" << "\n";
	std::cout << "Filter Tensor" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(kernels, rank + 2, kernel_dimensions, kernel_strides);

	float outputs[output_channels * output_depth * output_height * output_width] = { 0 };

	convolution_nd<float, int>(
		inputs, input_dimensions, input_strides,
		kernels, kernel_dimensions, kernel_strides,
		outputs, output_dimensions, output_strides,
		stride_dimensions,
		padding_dimensions,
		dilation_dimensions,
		rank
		);

	std::cout << "==================================================================" << "\n";
	std::cout << "Output Tensor" << "\n";
	std::cout << "==================================================================" << "\n\n";

	print_tensor(outputs, rank + 1, output_dimensions, output_strides);

}
