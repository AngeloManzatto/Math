#pragma once

template<typename value_type, typename size_type>
void zero_padding_recursive
(
	const value_type * inputs,  const size_type * input_dimensions, const size_type * input_strides,
		  value_type * outputs, const size_type * output_dimensions, const size_type * output_strides,
	const size_type * padding_dimensions,
	size_type rank,
	size_type dimension_step = 0,
	size_type output_offset = 0
)
{
	if (dimension_step < rank - 1)
	{

		for (int i_i = 0; i_i < input_dimensions[dimension_step]; i_i++, inputs += input_strides[dimension_step], outputs += output_strides[dimension_step])
		{
			zero_padding_recursive<value_type, size_type>(
				inputs, input_dimensions, input_strides,
				outputs , output_dimensions, output_strides,
				padding_dimensions,
				rank,
				dimension_step + 1,
				(padding_dimensions[dimension_step] + output_offset) * output_dimensions[dimension_step+1]
				);
		}
	}
	else
	{
		for (int i_i = 0; i_i < input_dimensions[dimension_step]; i_i++, inputs += input_strides[dimension_step], outputs += output_strides[dimension_step])
		{
			*(outputs + output_offset+ padding_dimensions[dimension_step])= *inputs;
		}
	}
}

template<typename value_type, typename size_type>
void zero_padding_nd(
	const value_type * inputs,  const size_type * input_dimensions,  const size_type * input_strides,
		  value_type * outputs, const size_type * output_dimensions, const size_type * output_strides,
	const size_type * padding_dimensions,
	const size_type rank
)
{
	const size_type& channels = input_dimensions[0];

	const size_type& input_channel_strides = input_strides[0];
	const size_type& output_channel_stride = output_strides[0];


	for (size_type c = 0; c < channels; c++, outputs += output_channel_stride, inputs += input_channel_strides)
	{
		zero_padding_recursive<value_type, size_type>(
			inputs,  input_dimensions  + 1, input_strides  + 1,
			outputs, output_dimensions + 1, output_strides + 1,
			padding_dimensions,
			rank
			);
	}

}
