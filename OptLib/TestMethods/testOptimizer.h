#pragma once
#include "stdafx.h"

namespace OptLib
{
	namespace UnitTests
	{
		class testOptimizer
		{
		public:
			static void testBicection()
			{
				std::cout << "******Bicection test start*****\n";
				ConcreteOptimizer::Bisection Algo{ new ConcreteFunc::FunctionWithHess{}, {{{-2}, {5}}} };
				std::cout << "Current simplex is:\n" << "  " << Algo.GuessDomain() << "\n";
				for (int i = 0; i < 10; i++)
				{
					Algo.Proceed();
					std::cout << "Current simplex is:\n" << "  " << Algo.GuessDomain() << "\n";
				}
				std::cout << "******Bicection test end*******\n\n";
			}

			static void testOverallOptimizer_WithBicection()
			{
				std::cout << "******OverallOptimizer With Bicection test start*****\n";

				OptimizerParams prm{ 0.001, 0.001, 101 };
				ConcreteFunc::Function f{};
				ConcreteOptimizer::Bisection Algo{ &f, {{{-2}, {5}}} };
				Optimizer<1, FuncInterface::IFunc<1>, ConcreteState::StateSegment> opt{ &Algo, prm };

				std::cout << "Optimization with Bicection started...\n";
				opt.Optimize();
				std::cout << "Optimization with Bicection finalized.\n";

				std::cout << "Total number of iterations is s = " << opt.CurIterCount() << '\n';
				std::cout << "Final guess is x = " << opt.CurrentGuess() << '\n';

				std::cout << "******OverallOptimizer With Bicection test end*******\n\n";
			}

			/*static void testDirect1DFuncAlongGrad()
			{
				std::cout << "******Optimization test start*****\n";
				FuncInterface::IFunc<1>* f = new ConcreteFunc::FunctionWithHess();
				ConcreteState::StateSimplex<1>* state = new ConcreteState::StateSimplex(Simplex<2, 1>{ { {-2}, { 5 }}});
				OptimizerInterface::IDirectAlgo<1>* algo = new ConcreteOptimizer::Bicection(f, state);

				Optimizer<1> Optimizer{ algo, OptimizerParams{0.001, 0.001, 100} };

				std::cout << "Initial simplex is:\n" << "  " << algo->CurSimplex() << "\n";
				Optimizer.Optimize();
				std::cout << "Final simplex is:\n" << "  " << algo->CurSimplex() << "\n";
				std::cout << "Guess value is:\n" << "  " << Optimizer.CurrentGuess() << "\n";
				std::cout << "Iter count is:\n" << "  " << Optimizer.cur_iter_count() << "\n";
				std::cout << "****** Optimization test end*******\n\n";
			}*/
		};

	} // UnitTests
} // OptLib