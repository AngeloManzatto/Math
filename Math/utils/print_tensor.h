#pragma once

#include <iostream>

// Print tensor
template<typename value_type, typename size_type>
void print_tensor(
	const value_type * inputs,
	const size_type rank,
	const size_type * input_dimensions,
	const size_type * input_strides, 
	size_type dimension_step = 0
)
{

	if (dimension_step < rank - 1)
	{
		for (int i = 0; i < input_dimensions[dimension_step]; i++, inputs += input_strides[dimension_step])
		{
			print_tensor(inputs, rank, input_dimensions, input_strides, dimension_step + 1);
		}
		std::cout << "\n";
	}
	else
	{
		for (int i = 0; i < input_dimensions[dimension_step]; i++, inputs += input_strides[dimension_step])
		{
			if (i != 0)
			{
				std::cout << ", " << *inputs;
			}
			else {
				std::cout << *inputs;
			}
		}
		std::cout << "\n";
	}

}

// Calculate stride
template<typename size_type = int>
void calculate_strides(size_type * dimensions, size_type * strides, size_type rank)
{
	strides[rank - 1] = 1;
	for (size_t i = rank - 1; i != 0; --i)
	{
		strides[i - 1] = strides[i] * dimensions[i];
	}
}