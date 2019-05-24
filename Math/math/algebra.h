template<typename value_type, typename size_type>
value_type dot_product
(
	const value_type * inputs_a, const size_type  input_strides_a,
	const value_type * inputs_b, const size_type  input_strides_b,
	const size_type size
)
{
	value_type sum = value_type();
	for (size_type i = 0; i < size; i++, inputs_a += input_strides_a,inputs_b += input_strides_b)
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
			*outputs++ = dot_product(inputs_a, input_strides_a[1], inputs_b, input_strides_b[0], input_dimensions_b[0]);
		}

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
	/*if (rank_a==1 and rank_b=1)
	{
		outputs[0] = dot_product<value_type, size_type>(inputs_a, input_strides_a[0], inputs_b, input_strides_b[0], output_dimensions[0]);
	}
	else 
	{
	*/
		matrix_mult_recursive< value_type, size_type>(
			inputs_a, input_dimensions_a, input_strides_a,
			inputs_b, input_dimensions_b, input_strides_b,
			outputs, output_dimensions, output_strides,
			rank_o
			);
	//}
}