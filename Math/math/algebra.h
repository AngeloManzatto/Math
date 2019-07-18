/*
*
* Artificial Intelligence Library
*
* Copyright (C) 2019 by Angelo Antonio Manzatto
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*
**/

template<typename value_type, typename size_type>
value_type dot_product
(
	const value_type * inputs_a, const size_type  input_dimensions_a, const size_type  input_strides_a,
	const value_type * inputs_b, const size_type  input_dimensions_b, const size_type  input_strides_b
)
{
	size_type size = (input_dimensions_a >= input_dimensions_b) ? input_dimensions_a : input_dimensions_b;

	value_type sum = value_type();
	for (size_type i = 0; i < size; i++, inputs_a += input_strides_a * (input_dimensions_a > 1) ,inputs_b += input_strides_b * (input_dimensions_b > 1))
	{
		sum += (*inputs_a) * (*inputs_b);
	}
	
	return sum;
}

template<typename value_type, typename size_type>
void inner_product
(
	const value_type * inputs_a, const size_type * input_dimensions_a, const size_type * input_strides_a,
	const value_type * inputs_b, const size_type * input_dimensions_b, const size_type * input_strides_b,
		  value_type * outputs,  const size_type * output_dimensions,  const size_type * output_strides
)
{
	for (size_type i = 0; i < input_dimensions_a[0]; i++, inputs_a += input_strides_a[0] )
	{
		for (size_type j = 0; j < input_dimensions_b[1]; j++, inputs_b += input_strides_b[1])
		{
			*outputs++ = dot_product(inputs_a, input_dimensions_a[1], input_strides_a[1],
				                     inputs_b, input_dimensions_b[0], input_strides_b[0]);
		}

		// Reset pointer position
		inputs_b -= input_dimensions_b[1];

	}
}

template<typename value_type, typename size_type>
void matrix_mult_recursive
(
	const value_type * inputs_a, const size_type * input_dimensions_a, const size_type * input_strides_a,
	const value_type * inputs_b, const size_type * input_dimensions_b, const size_type * input_strides_b,
	      value_type * outputs, const size_type  * output_dimensions, const size_type * output_strides, 
	size_type rank,
	size_type dimension_step = 0
)
{
	if (dimension_step < rank - 3)
	{
		for (int o_i = 0; o_i < output_dimensions[dimension_step]; o_i++, 
			outputs += output_strides[dimension_step],
			inputs_a += input_strides_a[dimension_step] * (input_dimensions_a[dimension_step] > 1),
			inputs_b += input_strides_b[dimension_step] * (input_dimensions_b[dimension_step] > 1))
		{
			matrix_mult_recursive< value_type, size_type>(
				inputs_a, input_dimensions_a, input_strides_a,
				inputs_b, input_dimensions_b, input_strides_b,
				outputs, output_dimensions, output_strides,
				rank,
				dimension_step + 1
				);
		}
	}
	else
	{
		for (int o_i = 0; o_i < output_dimensions[dimension_step]; o_i++, 
			outputs += output_strides[dimension_step],
			inputs_a += input_strides_a[dimension_step] * (input_dimensions_a[dimension_step] > 1),
			inputs_b += input_strides_b[dimension_step] * (input_dimensions_b[dimension_step] > 1))
		{
			int offset = rank - 2;
			inner_product< value_type, size_type>(
				inputs_a, input_dimensions_a + offset, input_strides_a + offset,
				inputs_b, input_dimensions_b + offset, input_strides_b + offset,
				outputs,  output_dimensions  + offset, output_strides  + offset
				);
		}
	}
}

template<typename value_type, typename size_type>
void matrix_mult
(
	const value_type * inputs_a,  const size_type * input_dimensions_a,  const size_type * input_strides_a, const size_type rank_a,
	const value_type * inputs_b,  const size_type * input_dimensions_b,  const size_type * input_strides_b, const size_type rank_b,
	      value_type * outputs,   const size_type * output_dimensions,   const size_type * output_strides,  const size_type rank_o
)
{
	if (rank_a==1 && rank_b==1)
	{
		*outputs = dot_product<value_type, size_type>(inputs_a, input_dimensions_a[0], input_strides_a[0],
													  inputs_b, input_dimensions_b[0], input_strides_b[0]);
	}
	else if (rank_a == 1 && rank_b == 2)
	{
		
		size_type input_dimension_t[2] = { 1, input_dimensions_a[0] };
		size_type input_strides_t[2] = { input_strides_a[0], 1 };

		inner_product<value_type, size_type>(inputs_a, input_dimension_t, input_strides_t,
										     inputs_b, input_dimensions_b, input_strides_b,
											 outputs,  output_dimensions, output_strides);

	}
	else if (rank_a == 2 && rank_b == 1)
	{
		size_type input_dimension_t[2] = { input_dimensions_b[0], 1 };
		size_type input_strides_t[2] = { 1, 1 };

		inner_product<value_type, size_type>(inputs_a, input_dimensions_a, input_strides_a,
											 inputs_b, input_dimension_t, input_strides_t,
										     outputs,  output_dimensions, output_strides);
	}
	else
	{
	
		matrix_mult_recursive< value_type, size_type>(
			inputs_a, input_dimensions_a, input_strides_a,
			inputs_b, input_dimensions_b, input_strides_b,
			outputs, output_dimensions, output_strides,
			rank_o
			);
	}
}