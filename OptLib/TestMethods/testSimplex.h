#pragma once
#include "stdafx.h"

namespace OptLib
{
	namespace UnitTests
	{
		class testSimplex
		{
		public:
			static void testSimplexVal()
			{
				std::cout << "******Simplex test start*****\n";

				Point<3> x{ {{1.1},{1.2},{0.3} } };
				auto y = RawPoint<3>{ std::move(x) };
				std::cout << "The RawPoint is " << y << '\n';

				auto z = PointVal<3>{ std::move(y), 1.0 };
				std::cout << "Point with value is " << z << '\n';

				Point<2> x1{ {{1.1},{1.2} } }, x2{ {{2.1},{2.2} } }, x3{ {{3.1},{3.2} } };

				SetOfPoints<3, Point<2>> simplex{ x1,x2,x3 };
				auto simplex2 = simplex;
				RawSetOfPoints<3, Point<2>> ss{ std::move(simplex) };

				std::array<double, 3> FuncVal{ 5.4, 6.7, 1.2 };
				SimplexValSort<2> simplex_val{ std::move(simplex2), std::move(FuncVal) };
				std::cout << "SimplexVal is " << simplex_val << '\n';

				std::cout << "******Simplex test end*******\n\n";
			}
		};
	} // UnitTests
} // OptLib