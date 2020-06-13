using System;

namespace NNBasicsUtilities.ActivationFunctions
{
	public static class SigmoidFunctions
	{
		public static double Sigmoid(double value) => 1.0f / (1.0 + Math.Exp(-value));
		public static double SigmoidDerivative(double value) => Sigmoid(value) * (1.0 - Sigmoid(value));
	}
}