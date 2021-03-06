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

#include <iostream>

namespace aicore {

	namespace math
	{
		//----------------------------------------------------------------------------------------
		// N-Dimensional Max Pooling
		//----------------------------------------------------------------------------------------

		template<typename value_type, typename size_type>
		value_type max_pooling_single_filter
		(
			const value_type * inputs, const size_type * input_dimensions, const size_type * input_strides,
			const size_type  * kernel_dimensions,
			const size_type  * dilation_dimensions,
			size_type * padding_offset,
			size_type rank,
			size_type dimension_step = 0,
			size_type input_offset = 0
		)
		{
			// Default minimum value for comparison
			value_type maximum = -std::numeric_limits<const value_type>::max();

			if (dimension_step < rank - 1)
			{

				for (int k_i = 0; k_i < kernel_dimensions[dimension_step]; k_i++, inputs += input_strides[dimension_step] * dilation_dimensions[dimension_step])
				{
					// Check if offset is within padding range
					if (dilation_dimensions[dimension_step] * k_i + padding_offset[dimension_step] >= 0 &&
						dilation_dimensions[dimension_step] * k_i + padding_offset[dimension_step] < input_dimensions[dimension_step])
					{
						value_type result = max_pooling_single_filter<value_type, size_type>(
							inputs, input_dimensions, input_strides,
							kernel_dimensions,
							dilation_dimensions,
							padding_offset,
							rank,
							dimension_step + 1,
							input_offset
							);

						maximum = result > maximum ? result : maximum;
					}

				}
			}
			else
			{

				for (int k_i = 0; k_i < kernel_dimensions[dimension_step]; k_i++, inputs += input_strides[dimension_step] * dilation_dimensions[dimension_step])
				{
					// Check if offset is within padding range
					if (dilation_dimensions[dimension_step] * k_i + padding_offset[dimension_step] >= 0 &&
						dilation_dimensions[dimension_step] * k_i + padding_offset[dimension_step] < input_dimensions[dimension_step])
					{
						value_type result = *(inputs + input_offset);

						maximum = result > maximum ? result : maximum;
					}

				}
			}

			return maximum;
		}

		template<typename value_type, typename size_type>
		void max_pooling_recursive
		(
			const value_type * inputs,  const size_type * input_dimensions,  const size_type * input_strides,
			      value_type * outputs, const size_type * output_dimensions, const size_type * output_strides,
			const size_type * kernel_dimensions,
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

					max_pooling_recursive<value_type, size_type>(
						inputs, input_dimensions, input_strides,
						outputs, output_dimensions, output_strides,
						kernel_dimensions,
						stride_dimensions,
						padding_dimensions,
						dilation_dimensions,
						padding_offset,
						rank,
						dimension_step + 1,
						(o_i * stride_dimensions[dimension_step] - padding_dimensions[dimension_step] + input_offset) * input_dimensions[dimension_step + 1]
						);

				}
			}
			else
			{
				for (int o_i = 0; o_i < output_dimensions[dimension_step]; o_i++, outputs += output_strides[dimension_step])
				{
					padding_offset[dimension_step] = (o_i * stride_dimensions[dimension_step] - padding_dimensions[dimension_step]);

					*outputs = max_pooling_single_filter<value_type, size_type>(
						inputs, input_dimensions, input_strides,
						kernel_dimensions,
						dilation_dimensions,
						padding_offset,
						rank,
						0,
						(o_i * stride_dimensions[dimension_step] - padding_dimensions[dimension_step] + input_offset)
						);

				}
			}

		}

		// BCHW
		template<typename value_type, typename size_type>
		void max_pooling_nd
		(
			const value_type * inputs,  const size_type * input_dimensions,  const size_type * input_strides,
			      value_type * outputs, const size_type * output_dimensions, const size_type * output_strides,
			const size_type * kernel_dimensions,
			const size_type * stride_dimensions,
			const size_type * padding_dimensions,
			const size_type * dilation_dimensions,
			const size_type rank
		)
		{

			int * padding_offset = new int[rank]();

			max_pooling_recursive<value_type, size_type>(
				inputs,  input_dimensions, input_strides,
				outputs, output_dimensions, output_strides,
				kernel_dimensions,
				stride_dimensions,
				padding_dimensions,
				dilation_dimensions,
				padding_offset,
				rank
				);

			// Free padding cache
			delete[] padding_offset;
		}

		//----------------------------------------------------------------------------------------
		// N-Dimensional Average Pooling
		//----------------------------------------------------------------------------------------
	
		template<typename value_type, typename size_type>
		value_type avg_pooling_single_filter
		(
			const value_type * inputs, const size_type * input_dimensions, const size_type * input_strides,
			const size_type  * kernel_dimensions,
			const size_type  * dilation_dimensions,
			size_type * padding_offset,
			size_type rank,
			size_type dimension_step = 0,
			size_type input_offset = 0,
			size_type filter_size = 0,
			bool include_pad_count = false
		)
		{
			// Default minimum value for comparison
			value_type average = value_type();

			if (dimension_step < rank - 1)
			{

				for (int k_i = 0; k_i < kernel_dimensions[dimension_step]; k_i++, inputs += input_strides[dimension_step] * dilation_dimensions[dimension_step])
				{
					// Check if offset is within padding range
					if (dilation_dimensions[dimension_step] * k_i + padding_offset[dimension_step] >= 0 &&
						dilation_dimensions[dimension_step] * k_i + padding_offset[dimension_step] < input_dimensions[dimension_step])
					{
						average += avg_pooling_single_filter<value_type, size_type>(
							inputs, input_dimensions, input_strides,
							kernel_dimensions,
							dilation_dimensions,
							padding_offset,
							rank,
							dimension_step + 1,
							input_offset,
							0,
							include_pad_count
							);
						filter_size += (include_pad_count == false);
					}
					filter_size += (include_pad_count == true);
				}
			}
			else
			{

				for (int k_i = 0; k_i < kernel_dimensions[dimension_step]; k_i++, inputs += input_strides[dimension_step] * dilation_dimensions[dimension_step])
				{
					// Check if offset is within padding range
					if (dilation_dimensions[dimension_step] * k_i + padding_offset[dimension_step] >= 0 &&
						dilation_dimensions[dimension_step] * k_i + padding_offset[dimension_step] < input_dimensions[dimension_step])
					{
						average += *(inputs + input_offset);
						filter_size += (include_pad_count == false);
					}
					filter_size += (include_pad_count == true);
				}
			}

			return average / filter_size;
		}

		template<typename value_type, typename size_type>
		void avg_pooling_recursive
		(
			const value_type * inputs,  const size_type * input_dimensions, const size_type * input_strides,
			      value_type * outputs, const size_type * output_dimensions, const size_type * output_strides,
			const size_type * kernel_dimensions,
			const size_type * stride_dimensions,
			const size_type * padding_dimensions,
			const size_type * dilation_dimensions,
			size_type * padding_offset,
			size_type rank,
			size_type dimension_step = 0,
			size_type input_offset = 0,
			bool include_pad_count = false
		)
		{
			if (dimension_step < rank - 1)
			{

				for (int o_i = 0; o_i < output_dimensions[dimension_step]; o_i++, outputs += output_strides[dimension_step])
				{
					padding_offset[dimension_step] = (o_i * stride_dimensions[dimension_step] - padding_dimensions[dimension_step]);

					avg_pooling_recursive<value_type, size_type>(
						inputs, input_dimensions, input_strides,
						outputs, output_dimensions, output_strides,
						kernel_dimensions,
						stride_dimensions,
						padding_dimensions,
						dilation_dimensions,
						padding_offset,
						rank,
						dimension_step + 1,
						(o_i * stride_dimensions[dimension_step] - padding_dimensions[dimension_step] + input_offset) * input_dimensions[dimension_step + 1],
						include_pad_count
						);

				}
			}
			else
			{
				for (int o_i = 0; o_i < output_dimensions[dimension_step]; o_i++, outputs += output_strides[dimension_step])
				{
					padding_offset[dimension_step] = (o_i * stride_dimensions[dimension_step] - padding_dimensions[dimension_step]);

					*outputs = avg_pooling_single_filter<value_type, size_type>(
						inputs, input_dimensions, input_strides,
						kernel_dimensions,
						dilation_dimensions,
						padding_offset,
						rank,
						0,
						(o_i * stride_dimensions[dimension_step] - padding_dimensions[dimension_step] + input_offset),
						0,
						include_pad_count
						);

				}
			}

		}

		// BCHW
		template<typename value_type, typename size_type>
		void avg_pooling_nd
		(
			const value_type * inputs,  const size_type * input_dimensions,  const size_type * input_strides,
			      value_type * outputs, const size_type * output_dimensions, const size_type * output_strides,
			const size_type * kernel_dimensions,
			const size_type * stride_dimensions,
			const size_type * padding_dimensions,
			const size_type * dilation_dimensions,
			      size_type rank,
			bool include_pad_count = false
		)
		{

			// Padding offset stores the padded values for each process step so we can ignore them 
			int * padding_offset = new int[rank]();

			avg_pooling_recursive<value_type, size_type>(
				inputs, input_dimensions, input_strides,
				outputs, output_dimensions, output_strides,
				kernel_dimensions,
				stride_dimensions,
				padding_dimensions,
				dilation_dimensions,
				padding_offset,
				rank,
				0,
				0, 
				include_pad_count
				);

			// Free padding cache
			delete[] padding_offset;
		}
}}





