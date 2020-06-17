using System;
using NNBasicsUtilities.ActivationFunctions;
using NNBasicsUtilities.Core.FlatCore.FlatNN;
using static NNBasicsUtilities.Core.Utilities.UtilitiesFunctions;

namespace Lab4
{
	class Program
	{
		static void Main(string[] args)
		{
			#region Task1

			var task1a =
				NeuralNetwork.Builder.AttachPredictionLayer(10, 40, 0.1, -0.1)
				   .AttachHiddenLayer(40, 784, 0.1, -0.1)
				   .ApplyActivationFunction(ReluFunctions.Relu)
				   .ApplyActivationFunctionDerivative(ReluFunctions.ReluDerivative)
				   .UseDropout()
				   .BuildHiddenLayer()
				   .WithAlpha(0.005)
				   .ApplyTheNameOfYourNetwork("Lab4_Task1a_MNIST_Flat")
				   .BuildNetwork();

			var (trainingSet, trainingLabels, testSet, testLabels) = LoadMnistDataBaseToFlatMatrix();

			//task1a.Train(trainingLabels[..1000, ..trainingLabels.Cols], trainingSet[..1000, ..trainingSet.Cols], 350);

			//task1a.Test(testLabels, testSet);

			var task1b =
				NeuralNetwork.Builder.AttachPredictionLayer(10, 100, 0.1, -0.1)
				   .AttachHiddenLayer(100, 784, 0.1, -0.1)
				   .ApplyActivationFunction(ReluFunctions.Relu)
				   .ApplyActivationFunctionDerivative(ReluFunctions.ReluDerivative)
				   .UseDropout()
				   .BuildHiddenLayer()
				   .WithAlpha(0.005)
				   .ApplyTheNameOfYourNetwork("Lab4_Task1b_MNIST_Flat")
				   .BuildNetwork();

			task1b.Train(trainingLabels[..10000, ..trainingLabels.Cols], trainingSet[..10000, ..trainingSet.Cols], 350);

			task1b.Test(testLabels, testSet);

			var task1c =
				NeuralNetwork.Builder.AttachPredictionLayer(10, 100, 0.1, -0.1)
				   .AttachHiddenLayer(100, 784, 0.1, -0.1)
				   .ApplyActivationFunction(ReluFunctions.Relu)
				   .ApplyActivationFunctionDerivative(ReluFunctions.ReluDerivative)
				   .UseDropout()
				   .BuildHiddenLayer()
				   .WithAlpha(0.005)
				   .ApplyTheNameOfYourNetwork("Lab4_Task1c_MNIST_Flat")
				   .BuildNetwork();

			task1c.Train(trainingLabels, trainingSet, 350);

			task1c.Test(testLabels, testSet);
		 #endregion
	  }
	}
}