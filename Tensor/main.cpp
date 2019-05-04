#include <iostream>


#include "tensor\tensor.h"


int main()
{

	Tensor<float> a({ 4, 4});
	Tensor<float> b({ 4, 4});
	Tensor<float> c({ 4, 4 });

	a(0 ,0) = 1.4f;
	b(3, 3) = 1.8f;

	c = -a + b;

	std::cout << c << std::endl;
	

	return 0;
}