#include "stdafx.h"

#ifdef DEBUG_LIB
	#include "OptLib/TestMethods/testSimplex.h"
	#include "OptLib/TestMethods/testFunction.h"
	#include "OptLib/TestMethods/testState.h"
	#include "OptLib/TestMethods/testOptimizer.h"
#endif // DEBUG_LIB

int main()
{
#ifdef DEBUG_LIB
	// test Simplex
	OptLib::UnitTests::testSimplex::testSimplexVal();
	
	// test functions
	OptLib::UnitTests::testFunction::testGradient();
	OptLib::UnitTests::testFunction::testHessian();
	OptLib::UnitTests::testFunction::test2DHessian();
	OptLib::UnitTests::testFunction::test2DWithHessNoGrad();
	OptLib::UnitTests::testFunction::testParaboloid();
	OptLib::UnitTests::testFunction::testParaboloidAlongDirection();

	// test states
	OptLib::UnitTests::testState::testStateDirect();
	OptLib::UnitTests::testState::testStateSegment();
	//OptLib::UnitTests::testState::testStatePoint(); // TODO
	//OptLib::UnitTests::testState::testStateStochastic(); // TODO

	// test optimizers
	OptLib::UnitTests::testOptimizer::testBicection();
	OptLib::UnitTests::testOptimizer::testOverallOptimizer_WithBicection(); 
	OptLib::UnitTests::testOptimizer::test_Grid();
	OptLib::UnitTests::testOptimizer::testOverallOptimizer_WithGrid();
	//testOptimizer::testOptimizer_();

#endif // DEBUG_LIB
}
