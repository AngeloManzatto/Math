#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

template<typename T> // Forward
class Tensor;

template<bool...> struct bool_pack;

template<bool... bs>
using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

template<class R, class... Ts>
using are_all_convertible = all_true<std::is_convertible<Ts, R>::value...>;

//-----------------------------------------------------------------
// Tensor 
//-----------------------------------------------------------------
template<typename T>
class Tensor
{
public:

	using value_type = T;
	using array_type = std::vector<value_type>;
	using iterator = typename std::vector<T>::iterator;
	using const_iterator = typename std::vector<T>::const_iterator;
	using reverse_iterator = typename std::vector<T>::reverse_iterator;
	using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;

	// ----------------------------------------------------------
	// Constructors


	Tensor(const std::vector<int>& shape);
	virtual ~Tensor();

	// Getters
	const int rank() const { return shapes_.size(); }
	const int size() const { return data_.size(); }

	      T* data() { return data_.data(); }
	const T* data()  const { return data_.data(); }

	const T& front() const { return data_[0]; }
	const T& back()  const { return data_[data_.size() - 1]; }

	const bool empty() { return data_.empty(); }

	const std::vector<int>& shapes() const { return shapes_; }
	const std::vector<int>& strides() const { return strides_; }

	const int shape(int index)  const { return index < 0 ? shapes_[shapes_.size() + index] : shapes_[index]; }
	const int stride(int index) const { return index < 0 ? strides_[strides_.size() + index] : strides_[index]; }

	iterator begin() { return data_.begin(); }
	iterator end() { return data_.end(); }

	const_iterator begin() const { return data_.begin(); }
	const_iterator end() const { return data_.end(); }

	iterator rbegin() { return data_.rbegin(); }
	iterator rend() { return data_.rend(); }

	const_iterator rbegin() const { return data_.rbegin(); }
	const_iterator rend() const { return data_.rend(); }

public:

	// Subscription

	template <typename... Dims,
		typename = typename std::enable_if<are_all_convertible<std::size_t, Dims...>::value>::type>
		T&  operator()(Dims&&... dims) { return *(data() + offset(std::vector<int>({ int(dims)... }))); }

	template <typename... Dims,
		typename = typename std::enable_if<are_all_convertible<std::size_t, Dims...>::value>::type>
		const T& operator()(Dims&&... dims) const { return *(data() + offset(std::vector<int>({ int(dims)... }))); }

	      T& operator()(const std::vector<size_t>& indices) { return *(data() + offset(indices)); }
	const T& operator()(const std::vector<size_t>& indices) const { return *(data() + offset(indices)); }

	      T& operator[](int index) { return data_[index]; }
	const T& operator[](int index) const { return data_[index]; }

	      T& at(int index)       { return data_[index]; }
	const T& at(int index) const { return data_[index]; }

	Tensor<T>& reshape(const std::vector<int>& shapes);

private:

	// Disable copy and assignment
	Tensor(const Tensor&) {};
	Tensor& operator=(const Tensor&) {};

private:

	int calculate_size(const std::vector<int>& shp);

	void calculate_stride();

	int offset(const std::vector<int>& indices);

private:

	std::size_t size_{ 0 };
	std::size_t offset_{ 0 };
	std::vector<int> shapes_;
	std::vector<int> strides_;
	std::vector<T> data_;

};

template<typename T>
inline Tensor<T>::Tensor(const std::vector<int>& shapes) :
	shapes_(std::move(shapes))
{
	calculate_stride();

	size_ = calculate_size(shapes_);

	data_.resize(size_);
}

template<typename T>
inline Tensor<T>::~Tensor()
{
}

template<typename T>
inline Tensor<T>& Tensor<T>::reshape(const std::vector<int>& shapes)
{
	shapes_.assign(shapes.begin(), shapes.end());

	calculate_stride();

	size_ = calculate_size(shapes_);

	if (data_.capacity() < size_)
	{
		data_.resize(size_);
	}

	return *this;
}

template<typename T>
inline int Tensor<T>::calculate_size(const std::vector<int>& shp)
{
	return std::accumulate(shp.begin(), shp.end(), 1, std::multiplies<int>());
}

template<typename T>
inline void Tensor<T>::calculate_stride()
{
	if (strides_.size() != shapes_.size())
		strides_.resize(shapes_.size());

	strides_[shapes_.size() - 1] = 1;
	for (size_t i = shapes_.size() - 1; i != 0; --i)
	{
		strides_[i - 1] = strides_[i] * shapes_[i];
	}
}

template<typename T>
inline int Tensor<T>::offset(const std::vector<int>& indices)
{
	assert(indices.size() == shapes_.size());

	constexpr std::size_t zero = 0;

	return std::inner_product(strides_.begin(), strides_.end(), indices.begin(), zero);
}
