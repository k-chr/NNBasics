#define task3b

using NNBasicsUtilities.ActivationFunctions;
using NNBasicsUtilities.Core.FlatCore.FlatNN;
using static NNBasicsUtilities.Core.Utilities.UtilitiesFunctions;

namespace Lab4
{
	class Program
	{
		static void Main(string[] args)
		{
			var (trainingSet, trainingLabels, testSet, testLabels) = LoadMnistDataBaseToFlatMatrix();

			#region Task1

#if task1a
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


			task1a.Train(trainingLabels[..1000, ..trainingLabels.Cols], trainingSet[..1000, ..trainingSet.Cols], 350);

			task1a.Test(testLabels, testSet);
#endif

#if task1b
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
#endif
#if task1c
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
#endif

			#endregion


			#region Task2

#if task2a
			var task2a =
				NeuralNetwork.Builder.AttachPredictionLayer(10, 40, 0.1, -0.1)
				   .AttachHiddenLayer(40, 784, 0.1, -0.1)
				   .ApplyActivationFunction(ReluFunctions.Relu)
				   .ApplyActivationFunctionDerivative(ReluFunctions.ReluDerivative)
				   .UseDropout()
				   .OfInputRows(100)
				   .BuildHiddenLayer()
				   .WithAlpha(0.001)
				   .OfInputRows(100)
				   .ApplyTheNameOfYourNetwork("Lab4_Task2a_MNIST_Flat")
				   .BuildNetwork();

			task2a.BatchTrain(trainingLabels[..1000, ..trainingLabels.Cols], trainingSet[..1000, ..trainingSet.Cols],
				350, 100);

			task2a.Test(testLabels, testSet);
#endif
#if task2b
			var task2b =
				NeuralNetwork.Builder.AttachPredictionLayer(10, 100, 0.1, -0.1)
				   .AttachHiddenLayer(100, 784, 0.1, -0.1)
				   .ApplyActivationFunction(ReluFunctions.Relu)
				   .ApplyActivationFunctionDerivative(ReluFunctions.ReluDerivative)
				   .UseDropout()
				   .OfInputRows(100)
				   .BuildHiddenLayer()
				   .WithAlpha(0.001)
				   .OfInputRows(100)
				   .ApplyTheNameOfYourNetwork("Lab4_Task2b_MNIST_Flat")
				   .BuildNetwork();

			task2b.BatchTrain(trainingLabels[..10000, ..trainingLabels.Cols], trainingSet[..10000, ..trainingSet.Cols],
				350, 100);

			task2b.Test(testLabels, testSet);
#endif
#if task2c
			var task2c =
				NeuralNetwork.Builder.AttachPredictionLayer(10, 100, 0.1, -0.1)
				   .AttachHiddenLayer(100, 784, 0.1, -0.1)
				   .ApplyActivationFunction(ReluFunctions.Relu)
				   .ApplyActivationFunctionDerivative(ReluFunctions.ReluDerivative)
				   .UseDropout()
				   .OfInputRows(100)
				   .BuildHiddenLayer()
				   .WithAlpha(0.0001)
				   .OfInputRows(100)
				   .ApplyTheNameOfYourNetwork("Lab4_Task2c_MNIST_Flat")
				   .BuildNetwork();

			task2c.BatchTrain(trainingLabels, trainingSet, 350, 100);

			task2c.Test(testLabels, testSet);
#endif

		 #endregion

		 #region Task3

#if task3a
			var task3a =
				NeuralNetwork.Builder.AttachPredictionLayer(10, 100, 0.01, -0.01)
				   .AttachHiddenLayer(100, 784, 0.01, -0.01)
				   .ApplyActivationFunction(TanHFunctions.TanH)
				   .ApplyActivationFunctionDerivative(TanHFunctions.TanHDerivative)
				   .UseDropout()
				   .OfInputRows(100)
				   .BuildHiddenLayer()
				   .WithAlpha(0.0002)
				   .OfInputRows(100)
				   .ApplyTheNameOfYourNetwork("Lab4_Task3a_MNIST_Flat")
				   .BuildNetwork();

			task3a.BatchTrain(trainingLabels[..1000, ..trainingLabels.Cols], trainingSet[..1000, ..trainingSet.Cols],
				350, 100);

			task3a.Test(testLabels, testSet);
#endif
#if task3b
			var task3b =
				NeuralNetwork.Builder.AttachPredictionLayer(10, 100, 0.01, -0.01)
				   .AttachHiddenLayer(100, 784, 0.01, -0.01)
				   .ApplyActivationFunction(TanHFunctions.TanH)
				   .ApplyActivationFunctionDerivative(TanHFunctions.TanHDerivative)
				   .UseDropout()
				   .OfInputRows(100)
				   .BuildHiddenLayer()
				   .WithAlpha(0.0006)
				   .OfInputRows(100)
				   .ApplyTheNameOfYourNetwork("Lab4_Task3b_MNIST_Flat")
				   .BuildNetwork();

			task3b.BatchTrain(trainingLabels[..10000, ..trainingLabels.Cols], trainingSet[..10000, ..trainingSet.Cols],
				350, 100);

			task3b.Test(testLabels, testSet);
#endif

		 #endregion
	  }
	}
}