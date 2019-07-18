#pragma once

// Calculate stride
template<typename size_type>
bool check_broadcast(const size_type * dimensions_a, const size_type * dimensions_b, const size_type rank)
{

	for (size_t i = 0; i < rank; i++)
	{
		if (dimensions_a[i] != dimensions_b[i] && dimensions_a[i] > 1 && dimensions_b[i] > 1)
		{
			return false;
		};
	}

	return true;
}


// Generate sequence of numbers
template<typename value_type, typename size_type>
void generate_sequence(value_type * inputs, size_type size)
{
	for (size_type i = 0; i < size; i++)
	{
		inputs[i] = i + 1;
	}
}