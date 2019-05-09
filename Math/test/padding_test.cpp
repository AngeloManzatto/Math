#include "padding_test.h"


#include "..\math\padding.h"
#include "..\utils\print_tensor.h"


void test_padding()
{
	const int rank = 3;

	// Input
	const int input_channels = 3;
	const int input_depth = 3;
	const int input_height = 4;
	const int input_width = 5;

	int input_dimensions[rank + 1] = { input_channels,input_depth,input_height, input_width };
	int input_strides[rank + 1] = { 0 };

	calculate_strides(input_dimensions, input_strides, rank + 1);

	// Initialize Input
	float inputs[input_channels * input_depth * input_height * input_width] = { 0 };
	for (size_t i = 0; i < input_channels  * input_depth * input_height * input_width; i++)
	{
		inputs[i] = i + 1.0f;
	}

	// Padding
	const int padding_depth = 1;
	const int padding_height = 2;
	const int padding_width = 3;

	int padding_dimensions[rank] = { padding_depth, padding_height, padding_width };

	// Outputs
	const int output_channels = input_channels;
	const int output_depth  = int(input_depth  + 2 * padding_depth);
	const int output_height = int(input_height + 2 * padding_height);
	const int output_width  = int(input_width  + 2 * padding_width);

	int output_dimensions[rank + 1] = { output_channels, output_depth, output_height, output_width };
	int output_strides[rank + 1] = { 0 };

	calculate_strides(output_dimensions, output_strides, rank + 1);

	std::cout << "==================================================================" << "\n";
	std::cout << "Input Tensor" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs, rank + 1, input_dimensions, input_strides);

	float outputs[output_channels * output_depth * output_height * output_width] = { 0 };

	zero_padding_nd<float, int>(
		inputs, input_dimensions, input_strides,
		outputs, output_dimensions, output_strides,
		padding_dimensions,
		rank
		);

	std::cout << "==================================================================" << "\n";
	std::cout << "Output Tensor" << "\n";
	std::cout << "==================================================================" << "\n\n";

	print_tensor(outputs, rank + 1, output_dimensions, output_strides);
}
