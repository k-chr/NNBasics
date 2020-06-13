namespace NNBasicsUtilities.ActivationFunctions
{
	public static class SwishFunctions
	{
		public static double Swish(double x) => x * SigmoidFunctions.Sigmoid(x);
		public static double SwishDerivative(double x) => Swish(x) + SigmoidFunctions.Sigmoid(x) * (1.0 - Swish(x));
	}
}