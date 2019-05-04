#pragma once

#include <iostream>

template<typename value_type, typename size_type>
value_type convolution_single_filter
(
	const value_type * inputs,  const size_type * input_dimensions,  const size_type * input_strides,
	const value_type * kernels, const size_type * kernel_dimensions, const size_type * kernel_strides,
	const size_type  * dilation_dimensions,
	size_type * padding_offset,
	size_type rank,
	size_type dimension_step = 0,
	size_type input_offset = 0,
	size_type kernel_offset = 0
)
{
	value_type sum = value_type();

	if (dimension_step < rank - 1)
	{
		for (int k_i = 0; k_i < kernel_dimensions[dimension_step]; k_i++, inputs += input_strides[dimension_step] * dilation_dimensions[dimension_step],kernels += kernel_strides[dimension_step])
		{
			if (dilation_dimensions[dimension_step] * k_i + padding_offset[dimension_step] >= 0 &&
				dilation_dimensions[dimension_step] * k_i + padding_offset[dimension_step] < input_dimensions[dimension_step])
			{
				sum += convolution_single_filter<value_type, size_type>(
					inputs, input_dimensions, input_strides,
					kernels, kernel_dimensions, kernel_strides,
					dilation_dimensions,
					padding_offset,
					rank,
					dimension_step + 1,
					input_offset,
					kernel_offset
					);
			}
		}

	}
	else
	{
		for (int k_i = 0; k_i < kernel_dimensions[dimension_step]; k_i++, inputs += input_strides[dimension_step] * dilation_dimensions[dimension_step], kernels += kernel_strides[dimension_step])
		{
			if (dilation_dimensions[dimension_step] * k_i + padding_offset[dimension_step] >= 0 &&
				dilation_dimensions[dimension_step] * k_i + padding_offset[dimension_step] < input_dimensions[dimension_step])
			{
				sum += *(inputs + input_offset) * *kernels;
			}
		}
	}

	return sum;
}

template<typename value_type, typename size_type>
void convolution_recursive
(
	const value_type * inputs,  const size_type * input_dimensions,  const size_type * input_strides,
	const value_type * kernels, const size_type * kernel_dimensions, const size_type * kernel_strides,
		  value_type * outputs, const size_type * output_dimensions, const size_type * output_strides,
	const size_type * stride_dimensions,
	const size_type * padding_dimensions,
	const size_type * dilation_dimensions,
		  size_type * padding_offset,
		  size_type rank,
		  size_type dimension_step = 0,
		  size_type input_offset = 0
)
{
	if (dimension_step < rank - 1)
	{
		for (int o_i = 0; o_i < output_dimensions[dimension_step]; o_i++, outputs += output_strides[dimension_step])
		{
			padding_offset[dimension_step] = (o_i * stride_dimensions[dimension_step] - padding_dimensions[dimension_step]);

			convolution_recursive<value_type, size_type>(
				inputs,  input_dimensions,  input_strides,
				kernels, kernel_dimensions, kernel_strides,
				outputs, output_dimensions, output_strides,
				stride_dimensions,
				padding_dimensions,
				dilation_dimensions,
				padding_offset,
				rank,
				dimension_step + 1,
				(o_i * stride_dimensions[dimension_step] - padding_dimensions[dimension_step] + input_offset) * input_dimensions[dimension_step + 1]
				);
		}
		//std::cout << "\n";
	}
	else
	{
		for (int o_i = 0; o_i < output_dimensions[dimension_step]; o_i++, outputs += output_strides[dimension_step])
		{
			padding_offset[dimension_step] = (o_i * stride_dimensions[dimension_step] - padding_dimensions[dimension_step]);
			//*outputs += 1;
			*outputs += convolution_single_filter<value_type, size_type>(
				inputs, input_dimensions, input_strides,
				kernels, kernel_dimensions, kernel_strides,
				dilation_dimensions,
				padding_offset,
				rank,
				0,
				(o_i * stride_dimensions[dimension_step] - padding_dimensions[dimension_step] + input_offset),
				0
				);
			
			//std::cout << *inputs << ",";
		}
		//std::cout  << "\n";
	}
}

// BCHW
template<typename value_type, typename size_type>
void convolution_nd
(
	const value_type * inputs,  const size_type * input_dimensions,  const size_type * input_strides,
	const value_type * kernels, const size_type * kernel_dimensions, const size_type * kernel_strides,
	      value_type * outputs, const size_type * output_dimensions, const size_type * output_strides,
	const size_type * stride_dimensions,
	const size_type * padding_dimensions,
	const size_type * dilation_dimensions,
	const size_type  rank,
	const size_type * bias = nullptr
)
{

	const size_type& output_channels = kernel_dimensions[0];
	const size_type& input_channels = kernel_dimensions[1];
	
	const size_type& input_channel_strides = input_strides[0];
	const size_type& kernel_channel_stride = kernel_strides[1];
	const size_type& output_channel_stride = output_strides[0];

	// Padding offset stores the padded values for each process step so we can ignore them 
	size_type * padding_offset = new size_type[rank]();

	size_type input_size = 1;
	for (size_type i = 0; i < rank + 1; i++)
	{
		input_size *= input_dimensions[i];
	}

	for (size_type o_c = 0; o_c < output_channels; o_c++, outputs += output_channel_stride)
	{
		for (size_type i_c = 0; i_c < input_channels; i_c++, kernels += kernel_channel_stride, inputs += input_channel_strides)
		{
			convolution_recursive<value_type, size_type>(
				inputs,  input_dimensions  + 1, input_strides  + 1,
				kernels, kernel_dimensions + 2, kernel_strides + 2,
				outputs, output_dimensions + 1, output_strides + 1,
				stride_dimensions,
				padding_dimensions,
				dilation_dimensions,
				padding_offset,
				rank,
				0,
				0
				);
		}

		//Reset pointer position
		inputs -= input_size;
	}

	


	// Free padding cache
	delete[] padding_offset;
}