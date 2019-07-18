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

		zero_padding_recursive<value_type, size_type>(
			inputs,  input_dimensions, input_strides,
			outputs, output_dimensions, output_strides,
			padding_dimensions,
			rank
			);
	

}
