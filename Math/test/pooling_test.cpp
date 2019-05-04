#include "pooling_test.h"

#include "..\math\pooling.h"
#include "..\utils\print_tensor.h"

using aicore::math::max_pooling_nd;
using aicore::math::avg_pooling_nd;

void test_pooling()
{
	const int rank = 3;

	// Input
	const int input_channels = 2;
	const int input_depth = 3;
	const int input_height = 4;
	const int input_width = 5;

	int input_dimensions[rank + 1] = { input_channels, input_depth ,input_height, input_width };
	int input_strides[rank + 1] = { 0 };

	calculate_strides(input_dimensions, input_strides, rank + 1);

	// Initialize Input
	float inputs[input_channels * input_depth * input_height * input_width] = { 0 };
	for (size_t i = 0; i < input_channels * input_depth * input_height * input_width; i++)
	{
		inputs[i] = i + 1.0f;
	}

	// Kernel
	const int kernel_depth = 2;
	const int kernel_height = 2;
	const int kernel_width = 1;

	int kernel_dimensions[rank] = { kernel_depth, kernel_height, kernel_width };

	// Strides
	const int stride_depth = 3;
	const int stride_height = 1;
	const int stride_width = 2;

	int stride_dimensions[rank] = { stride_depth, stride_height, stride_width };

	// Padding
	const int padding_depth = 1;
	const int padding_height = 1;
	const int padding_width = 0;

	int padding_dimensions[rank] = { padding_depth, padding_height, padding_width };

	// Dilation
	const int dilation_depth = 1;
	const int dilation_height = 1;
	const int dilation_width = 1;

	int dilation_dimensions[rank] = { dilation_depth, dilation_height,dilation_width };

	// Outputs

	const int output_channels = input_channels;
	const int output_depth =  int((input_depth  + 2 * padding_depth  - dilation_depth  * (kernel_depth  - 1) - 1) / stride_depth) + 1;
	const int output_height = int((input_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) / stride_height) + 1;
	const int output_width =  int((input_width  + 2 * padding_width  - dilation_width  * (kernel_width  - 1) - 1) / stride_width) + 1;

	int output_dimensions[rank + 1] = { output_channels, output_depth, output_height, output_width };
	int output_strides[rank + 1] = { 0 };

	calculate_strides(output_dimensions, output_strides, rank + 1);

	std::cout << "==================================================================" << "\n";
	std::cout << "Input Tensor" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs, rank + 1, input_dimensions, input_strides);

	float outputs[ output_channels * output_depth * output_height * output_width] = { 0 };

	max_pooling_nd<float, int>(
		inputs, input_dimensions, input_strides,
		outputs, output_dimensions, output_strides,
		kernel_dimensions,
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
