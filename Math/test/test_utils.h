#pragma once

template<typename value_type, typename size_type>
void generate_sequence(value_type * inputs, size_type size)
{
	for (size_type i = 0; i < size; i++)
	{
		inputs[i] = i + 1;
	}
}