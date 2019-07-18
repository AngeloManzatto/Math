#include "algebra_test.h"

#include <cassert>

#include "..\math\algebra.h"
#include "..\utils\print_tensor.h"
#include "..\utils\tensor_utils.h"

void matrix_test()
{

	const int rank_a = 5;

	// Input
	const int input_channels_a = 5;
	const int input_batch_a = 1;
	const int input_depth_a = 3;
	const int input_height_a = 2;
	const int input_width_a = 6;

	int input_dimensions_a[rank_a] = { input_channels_a,input_batch_a, input_depth_a ,input_height_a, input_width_a };
	int input_strides_a[rank_a] = { 0 };

	calculate_strides(input_dimensions_a, input_strides_a, rank_a);

	// Initialize Input
	float inputs_a[input_channels_a * input_batch_a * input_depth_a  *input_height_a * input_width_a] = { 0 };
	for (size_t i = 0; i < input_channels_a *input_batch_a* input_depth_a * input_height_a * input_width_a; i++)
	{
		inputs_a[i] = i + 1.0f;
	}

	const int rank_b = 5;

	// Input b
	const int input_channels_b = 5;
	const int input_batch_b = 4;
	const int input_depth_b = 1;
	const int input_height_b = 6;
	const int input_width_b = 2;

	int input_dimensions_b[rank_b] = { input_channels_b,input_batch_b, input_depth_b ,input_height_b, input_width_b};
	int input_strides_b[rank_b] = { 0 };

	calculate_strides(input_dimensions_b, input_strides_b, rank_b);

	// Initialize Input
	float inputs_b[input_channels_b * input_batch_b*input_depth_b * input_height_b * input_width_b] = { 0 };
	for (size_t i = 0; i < input_channels_b * input_batch_b*input_depth_b *input_height_b * input_width_b; i++)
	{
		inputs_b[i] = i + 1.0f;
	}

	// Outputs
	const int rank_o = 5;

	const int output_channels = 5;
	const int output_batch = 4;
	const int output_depth = 3;
	const int output_height = 2;
	const int output_width = 2;

	int output_dimensions[rank_o] = { input_channels_a,output_batch, input_depth_a,  input_height_a,input_width_b };
	int output_strides[rank_o] = { 0 };

	calculate_strides(output_dimensions, output_strides, rank_o);

	//assert();

	std::cout << "==================================================================" << "\n";
	std::cout << "Input Tensor A" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs_a, rank_a, input_dimensions_a, input_strides_a);

	std::cout << "==================================================================" << "\n";
	std::cout << "Input Tensor B" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs_b, rank_b, input_dimensions_b, input_strides_b);

	float outputs[output_channels * output_batch * output_depth * output_height * output_width] = { 0 };

	matrix_mult<float, int>(
		inputs_a,  input_dimensions_a, input_strides_a,rank_a,
		inputs_b, input_dimensions_b, input_strides_b,rank_b,
		outputs, output_dimensions, output_strides, rank_o

		);

	std::cout << "==================================================================" << "\n";
	std::cout << "Output Tensor" << "\n";
	std::cout << "==================================================================" << "\n\n";

	print_tensor(outputs, rank_b, output_dimensions, output_strides);
}

void matrix_test_case_1()
{
	// Tensor A

	// Rank
	const int rank_a = 1;

	// Dimension
	const int input_width_a = 6;
	int input_dimensions_a[rank_a] = { input_width_a };

	// Stride
	int input_strides_a[rank_a] = { 1 };

	// Initialize Input
	float inputs_a[input_width_a] = { 0 };
	generate_sequence(inputs_a, input_width_a);

	// Tensor B
	const int rank_b = 1;

	// Rank
	const int input_width_b = 1;

	// Dimension
	int input_dimensions_b[rank_b] = {  input_width_b };
	int input_strides_b[rank_b] = { 1 };

	// Initialize Input
	float inputs_b[input_width_b] = { 6 };
	//generate_sequence(inputs_b, input_width_b);

	// Tensor Output

	// Rank
	const int rank_o = 1;

	// Dimension
	const int input_width_o = 1;
	int input_dimensions_o[rank_o] = { input_width_o};

	// Stride
	int input_strides_o[rank_o] = { 1 };

	// Initialize Tensor
	float inputs_o[input_width_o] = { 0 };

	// Check constraints
	assert(check_broadcast(input_dimensions_a, input_dimensions_b, rank_a));

	std::cout << "==================================================================" << "\n";
	std::cout << "Input Tensor A" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs_a, rank_a, input_dimensions_a, input_strides_a);

	std::cout << "==================================================================" << "\n";
	std::cout << "Input Tensor B" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs_b, rank_b, input_dimensions_b, input_strides_b);

	std::cout << "==================================================================" << "\n";
	std::cout << "Dot Product Tensor A * Tensor B" << "\n";
	std::cout << "==================================================================" << "\n\n";

	auto a = dot_product(inputs_a, input_width_a, input_strides_a[0], inputs_b, input_width_b, input_strides_b[0]);

	std::cout << a << "\n";

	matrix_mult(inputs_a, input_dimensions_a, input_strides_a, rank_a,
				inputs_b, input_dimensions_b, input_strides_b, rank_b,
		        inputs_o, input_dimensions_o, input_strides_o, rank_o);

	std::cout << "==================================================================" << "\n";
	std::cout << "Tensor Output" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs_o, rank_o, input_dimensions_o, input_strides_o);
	
}

void matrix_test_case_2()
{
	// Tensor A

	// Rank
	const int rank_a = 2;

	// Dimension
	const int input_height_a = 2;
	const int input_width_a = 6;
	const int input_size_a = input_height_a * input_width_a;

	int input_dimensions_a[rank_a] = { input_height_a, input_width_a };
	int input_strides_a[rank_a] = { 0 };

	float inputs_a[input_size_a] = { 0 };

	calculate_strides(input_dimensions_a, input_strides_a, rank_a);
	generate_sequence(inputs_a, input_size_a);

	// Tensor B

	// Rank
	const int rank_b = 2;

	// Dimension
	const int input_height_b = 6;
	const int input_width_b = 3;
	const int input_size_b = input_height_b * input_width_b;

	int input_dimensions_b[rank_b] = { input_height_b, input_width_b };
	int input_strides_b[rank_b] = { 0 };

	float inputs_b[input_size_b] = { 0 };

	calculate_strides(input_dimensions_b, input_strides_b, rank_b);
	generate_sequence(inputs_b, input_size_b);

	// Tensor Output

	// Rank
	const int rank_o = 2;

	// Dimension
	const int input_height_o = input_height_a;
	const int input_width_o = input_width_b;
	const int input_size_o = input_height_o * input_width_o;

	int input_dimensions_o[rank_o] = { input_height_o, input_width_o };
	int input_strides_o[rank_o] = { 0 };

	float inputs_o[input_size_o] = { 0 };

	calculate_strides(input_dimensions_o, input_strides_o, rank_o);

	std::cout << "==================================================================" << "\n";
	std::cout << "Input Tensor A" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs_a, rank_a, input_dimensions_a, input_strides_a);

	std::cout << "==================================================================" << "\n";
	std::cout << "Input Tensor B" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs_b, rank_b, input_dimensions_b, input_strides_b);

	std::cout << "==================================================================" << "\n";
	std::cout << "Inner Product Matrix A * Matrix B" << "\n";
	std::cout << "==================================================================" << "\n\n";

	inner_product(inputs_a, input_dimensions_a, input_strides_a,
			      inputs_b, input_dimensions_b, input_strides_b,
		          inputs_o, input_dimensions_o, input_strides_o);

	print_tensor(inputs_o, rank_o, input_dimensions_o, input_strides_o);
}

void matrix_test_case_3()
{
	// Tensor A

	// Rank
	const int rank_a = 1;

	// Dimension
	const int input_width_a = 6;
	int input_dimensions_a[rank_a] = { input_width_a };

	// Stride
	int input_strides_a[rank_a] = { 1 };

	// Initialize Input
	float inputs_a[input_width_a] = { 0 };
	generate_sequence(inputs_a, input_width_a);

	// Tensor B

	// Rank
	const int rank_b = 2;

	// Dimension
	const int input_height_b = 6;
	const int input_width_b = 3;
	const int input_size_b = input_height_b * input_width_b;

	int input_dimensions_b[rank_b] = { input_height_b, input_width_b };
	int input_strides_b[rank_b] = { 0 };

	float inputs_b[input_size_b] = { 0 };

	calculate_strides(input_dimensions_b, input_strides_b, rank_b);
	generate_sequence(inputs_b, input_size_b);

	// Tensor Output

	// Rank
	const int rank_o = 2;

	// Dimension
	const int input_height_o = 1;
	const int input_width_o = input_width_b;
	const int input_size_o = input_height_o * input_width_o;

	int input_dimensions_o[rank_o] = { input_height_o, input_width_o };
	int input_strides_o[rank_o] = { 0 };

	float inputs_o[input_size_o] = { 0 };

	calculate_strides(input_dimensions_o, input_strides_o, rank_o);

	std::cout << "==================================================================" << "\n";
	std::cout << "Input Tensor A" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs_a, rank_a, input_dimensions_a, input_strides_a);

	std::cout << "==================================================================" << "\n";
	std::cout << "Input Tensor B" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs_b, rank_b, input_dimensions_b, input_strides_b);

	std::cout << "==================================================================" << "\n";
	std::cout << "Vector A * Matrix B" << "\n";
	std::cout << "==================================================================" << "\n\n";

	matrix_mult(inputs_a, input_dimensions_a, input_strides_a, rank_a,
		        inputs_b, input_dimensions_b, input_strides_b, rank_b,
		        inputs_o, input_dimensions_o, input_strides_o, rank_o);

	print_tensor(inputs_o, rank_o, input_dimensions_o, input_strides_o);
}

void matrix_test_case_4()
{
	// Tensor A

	// Rank
	const int rank_a = 2;

	// Dimension
	const int input_height_a = 6;
	const int input_width_a = 3;
	const int input_size_a = input_height_a * input_width_a;

	int input_dimensions_a[rank_a] = { input_height_a, input_width_a };
	int input_strides_a[rank_a] = { 0 };

	float inputs_a[input_size_a] = { 0 };

	calculate_strides(input_dimensions_a, input_strides_a, rank_a);
	generate_sequence(inputs_a, input_size_a);

	// Tensor B

	// Rank
	const int rank_b = 1;

	// Dimension
	const int input_width_b = 3;
	int input_dimensions_b[rank_b] = { input_width_b };

	// Stride
	int input_strides_b[rank_b] = { 1 };

	// Initialize Input
	float inputs_b[input_width_b] = { 0 };
	generate_sequence(inputs_b, input_width_b);


	// Tensor Output

	// Rank
	const int rank_o = 2;

	// Dimension
	const int input_height_o = input_height_a;
	const int input_width_o = 1;
	const int input_size_o = input_height_o * input_width_o;

	int input_dimensions_o[rank_o] = { input_height_o, input_width_o };
	int input_strides_o[rank_o] = { 0 };

	float inputs_o[input_size_o] = { 0 };

	calculate_strides(input_dimensions_o, input_strides_o, rank_o);

	std::cout << "==================================================================" << "\n";
	std::cout << "Input Tensor A" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs_a, rank_a, input_dimensions_a, input_strides_a);

	std::cout << "==================================================================" << "\n";
	std::cout << "Input Tensor B" << "\n";
	std::cout << "==================================================================" << "\n\n";
	print_tensor(inputs_b, rank_b, input_dimensions_b, input_strides_b);

	std::cout << "==================================================================" << "\n";
	std::cout << "Matrix A * Vector B" << "\n";
	std::cout << "==================================================================" << "\n\n";

	matrix_mult(inputs_a, input_dimensions_a, input_strides_a, rank_a,
				inputs_b, input_dimensions_b, input_strides_b, rank_b,
			    inputs_o, input_dimensions_o, input_strides_o, rank_o);

	print_tensor(inputs_o, rank_o, input_dimensions_o, input_strides_o);
}
