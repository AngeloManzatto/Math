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
// CRTP
//-----------------------------------------------------------------
template<typename Derived>
struct base_expression
{
	operator const Derived&() const { return static_cast<const Derived&>(*this); }
};

//-----------------------------------------------------------------
// Tensor - Tensor Expression
//-----------------------------------------------------------------
template<typename Left, typename Right, typename Operator>
class tensor_tensor_operator : public base_expression<tensor_tensor_operator<Left, Right, Operator>>
{
public:

	tensor_tensor_operator(const base_expression<Left>& l, const base_expression<Right>& r) :left_(l), right_(r) {};

	auto operator[](int index) const { return Operator::eval(left_[index], right_[index]); }

	int rank() const { return left_.rank() >= right_.rank() ? left_.rank() : right_.rank(); }

private:
	const Left& left_;
	const Right& right_;
};

//-----------------------------------------------------------------
// Tensor - Scalar Expression
//-----------------------------------------------------------------
template<typename Left, typename Right, typename Operator>
class tensor_scalar_operator : public base_expression<tensor_scalar_operator<Left, Right, Operator>>
{
public:

	tensor_scalar_operator(const base_expression<Left>& l, const Right& r) :left_(l), right_(r) {};

	auto operator[](int index) const { return Operator::eval(left_[index], right_); }

private:
	const Left& left_;
	const Right right_;
};

//-----------------------------------------------------------------
// Scalar - Tensor Expression
//-----------------------------------------------------------------
template<typename Left, typename Right, typename Operator>
class scalar_tensor_operator : public base_expression<scalar_tensor_operator<Left, Right, Operator>>
{
public:

	scalar_tensor_operator(const Left& l, const base_expression<Right>& r) :left_(l), right_(r) {};

	auto operator[](int index) const { return Operator::eval(left_, right_[index]); }

private:

	const Left   left_;
	const Right& right_;
};

//-----------------------------------------------------------------
// Addition Operator
//-----------------------------------------------------------------
struct add_operator
{
	template<typename T, typename U> static auto eval(const T& l, const U& r) { return l + r; }
};

//-----------------------------------------------------------------
// Tensor + Tensor
//-----------------------------------------------------------------
template<typename Left, typename Right>
tensor_tensor_operator<Left, Right, add_operator> operator+(const base_expression<Left>& l, const base_expression<Right>& r)
{
	return tensor_tensor_operator<Left, Right, add_operator>(l, r);
}

//-----------------------------------------------------------------
// Tensor + Scalar Operator
//-----------------------------------------------------------------
template<typename Left, typename Right>
	std::enable_if_t<std::is_arithmetic<Right>::value, tensor_scalar_operator<Left, Right, add_operator>> // Enable only if Right is arithmetic type
operator+(const base_expression<Left>& l, const Right& r)
{
	return tensor_scalar_operator<Left, Right, add_operator>(l, r);
}

//-----------------------------------------------------------------
//  Scalar + Tensor Operator
//-----------------------------------------------------------------	
template<typename Left, typename Right>
	std::enable_if_t<std::is_arithmetic<Left>::value, scalar_tensor_operator<Left, Right, add_operator>> // Enable only if Left is arithmetic type
operator+(const Left& r, const base_expression<Right>& l)
{
	return scalar_tensor_operator<Left, Right, add_operator>(l, r);
}

//-----------------------------------------------------------------
// Tensor Unary Expression
//-----------------------------------------------------------------
template<typename Left, typename Operator>
class tensor_unary_operator : public base_expression<tensor_unary_operator<Left, Operator>>
{
public:

	tensor_unary_operator(const base_expression<Left>& l) :left_(l) {};

	auto operator[](int index) const { return Operator::eval(left_[index]); }

	std::size_t size() const { return left_.size(); }

	std::vector<int>& shapes() const { return left_.shapess(); }

private:
	const Left& left_;

};

struct neg_operator
{
	template<typename T> static auto eval(const T& l) { return -l; }
};

//-----------------------------------------------------------------
// Tensor Unary Negation Operator
//-----------------------------------------------------------------
template<typename Left>
tensor_unary_operator<Left, neg_operator> operator-(const base_expression<Left>& l)
{
	return tensor_unary_operator<Left, neg_operator>(l);
}

//-----------------------------------------------------------------
// Tensor 
//-----------------------------------------------------------------
template<typename T>
class Tensor : public base_expression<Tensor<T>>
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

	template<typename Expression>
	Tensor& operator=(const base_expression<Expression>& expression)
	{
		const Expression& expression_(expression);
		for (int i = 0; i < size(); i++)
		{
			data_[i] = expression_[i];
		}

		return *this;
	}

	// Getters
	const int rank() const { return shapes_.size(); }
	const int size() const { return data_.size(); }

	      T* data()        { return data_.data(); }
	const T* data()  const { return data_.data(); }

	const T& front() const { return data_[0]; }
	const T& back()  const { return data_[data_.size() - 1]; }

	const bool empty() { return data_.empty(); }

	const std::vector<int>& shapes() const { return shapes_; }
	const std::vector<int>& strides() const { return strides_; }

	const int shape(int index) const { return index < 0 ? shapes_[shapes_.size() + index] : shapes_[index]; }
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

	      T& operator()(const std::vector<size_t>& indices)       { return *(data() + offset(indices)); }
	const T& operator()(const std::vector<size_t>& indices) const { return *(data() + offset(indices)); }

		  T& operator[](int index)       { return data_[index]; }
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
inline Tensor<T>::Tensor(const std::vector<int>& shapes):
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

// Print tensor
template<typename value_type, typename size_type>
void print_tensor(std::ostream& os, value_type * ptr, size_type rank, const size_type * shapes, const size_type * strides, size_type dimension_step = 0)
{

	if (dimension_step < rank - 1)
	{
		for (int i = 0; i < shapes[dimension_step]; i++, ptr += strides[dimension_step])
		{
			print_tensor(os, ptr, rank, shapes, strides, dimension_step + 1);
		}
		os << "\n";
	}
	else
	{
		for (int i = 0; i < shapes[dimension_step]; i++, ptr += strides[dimension_step])
		{
			if (i != 0)
			{
				os << ", " << *ptr;
			}
			else {
				os << *ptr;
			}
		}
		os << "\n";
	}

}

// Print tensor
template<typename T>
std::ostream& operator << (std::ostream& os, const Tensor<T>& tensor)
{

	print_tensor(os, tensor.data(), tensor.rank(), tensor.shapes().data(), tensor.strides().data());

	return os;
}

