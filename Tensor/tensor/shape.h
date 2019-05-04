#pragma once

#include <array>

template<size_t N, typename T = int>
class Shape
{
public:

	Shape() {};

	Shape(const T& value)
	{
		values_.fill(value);
	}

	Shape(std::initializer_list<T> list)
	{

		std::copy(list.begin(), list.end(), values_.begin());
	}

	      T& operator[](int index)       { return values_[index]; }
	const T& operator[](int index) const { return values_[index]; }

	const size_t size() const noexcept { return  N; }

	std::array<T, N>& operator*() {
		return values_;
	}

	const std::array<T, N>& operator*() const {
		return values_;
	}

	std::array<T, N>* operator->() {
		return &values_;
	}

	const std::array<T, N>* operator->() const {
		return &values_;
	}


private:
	std::array<int, N> values_{ 0 };
};
