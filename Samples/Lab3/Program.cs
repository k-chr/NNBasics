using System.Collections.Generic;
using NNBasicsUtilities.ActivationFunctions;
using NNBasicsUtilities.Core;
using NNBasicsUtilities.Extensions;
using static NNBasicsUtilities.Core.Utilities.UtilitiesFunctions;

namespace Lab3
{
   class Program
   {
      static void Main(string[] args)
      {
         #region Task1 & Task2 

         //var hiddenWeights = new List<List<double>>
         //{
         //   new List<double> {0.1, 0.2, -0.1},
         //   new List<double> {-0.1, 0.1, 0.9},
         //   new List<double> {0.1, 0.4, 0.1},
         //};

         //var outputWeights = new List<List<double>>
         //{
         //   new List<double> {0.3, 1.1, -0.3},
         //   new List<double> {0.1, 0.2, 0.0},
         //   new List<double> {0.0, 1.3, 0.1},
         //};

         //var series = new List<List<double>>
         //{
         //   new List<double> {8.5, 0.65, 1.2},
         //   new List<double> {9.5, 0.8, 1.3},
         //   new List<double> {9.9, 0.8, 0.5},
         //   new List<double> {9.0, 0.9, 1.0},
         //};

         //var expected = new List<List<double>>
         //{
         //   new List<double> {0.1, 1, 0.1},
         //   new List<double> {0.0, 1, 0.0},
         //   new List<double> {0.0, 0, 0.1},
         //   new List<double> {0.1, 1, 0.2},
         //};

         //var neuralNetworkTask1AndTask2 = NeuralNetwork.Builder
         //   .WithAlpha(0.01)
         //   .AddHiddenLayer(hiddenWeights.ToMatrix())
         //      .ApplyActivationFunction(ReluFunctions.Relu)
         //      .ApplyActivationFunctionDerivative(ReluFunctions.ReluDerivative)
         //      .BuildHiddenLayer()
         //   .ApplyTheNameOfYourNetwork("Lab3_Task1_And_Task2")
         //   .AttachPredictionLayer(outputWeights.ToMatrix())
         //   .BuildNetwork();

         //neuralNetworkTask1AndTask2.Train(expected.ToMatrix(), series.ToMatrix(), 10000);

         //#endregion
      
         //#region Task3

         //var (matSeries, matExpected) =
         //   ParseSeriesAndExpectedAnswersFromWeb(@"http://pduch.iis.p.lodz.pl/PSI/training_colors.txt");

         //var (testSeries, testExpected) =
         //   ParseSeriesAndExpectedAnswersFromWeb(@"http://pduch.iis.p.lodz.pl/PSI/test_colors.txt");

         //var networkTask3 =
         //   NeuralNetwork.Builder.AttachPredictionLayer(4, 4, 0.1, -0.1)
         //      .AttachHiddenLayer(4, 3, 0.2, -0.2)
         //         .ApplyActivationFunction(ReluFunctions.Relu)
         //         .ApplyActivationFunctionDerivative(ReluFunctions.ReluDerivative)
         //         .BuildHiddenLayer()
         //      .WithAlpha(0.01)
         //      .ApplyTheNameOfYourNetwork("Lab3_Task3_Colors")
         //      .BuildNetwork();

         //networkTask3.Train(matExpected, matSeries, 100);
         //networkTask3.Test(testExpected, testSeries);

         #endregion  

         #region Task4

         var networkTask4 =
            NeuralNetwork.Builder.AttachPredictionLayer(10, 40, 0.1, -0.1)
               .AttachHiddenLayer(40, 784, 0.1, -0.1)
               .ApplyActivationFunction(ReluFunctions.Relu)
               .ApplyActivationFunctionDerivative(ReluFunctions.ReluDerivative)
               .BuildHiddenLayer()
               .WithAlpha(0.01)
               .ApplyTheNameOfYourNetwork("Lab3_Task4_MNIST")
               .BuildNetwork();

         var (trainingSet, trainingLabels, testSet, testLabels) = LoadMnistDataBase();

         networkTask4.Train(trainingLabels, trainingSet, 300);

         networkTask4.Test(testLabels, testSet);

         #endregion
      }
   }
}
