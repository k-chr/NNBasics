using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using NNBasics.NNBasicsLimak.ActivationFunctions;
using NNBasics.NNBasicsLimak.Core.Layers;
using NNBasics.NNBasicsLimak.Core.Neurons;
using NNBasics.NNBasicsLimak.Core.UtilityTypes;
using NNBasics.NNBasicsLimak.Extensions;

namespace NNBasics.NNBasicsLimak.Core
{
   //TODO Implement Neural network behaviour and add Layers members and backward propagation with activation functions
   public class NeuralNetwork
   {
      private readonly List<HiddenLayer> _hiddenLayers;
      private double _currentIteration;
      private readonly PredictLayer _predictLayer;
      public event EventHandler<string> LogReport;

      private NeuralNetwork() { }

      private NeuralNetwork(NeuralNetworkBuilder neuralNetworkBuilder)
      {
         _hiddenLayers = neuralNetworkBuilder.HiddenLayers;
         _predictLayer = neuralNetworkBuilder.PredictionLayer;
         var alpha = Math.Max(neuralNetworkBuilder.Alpha, 0.01);
         _predictLayer.Alpha = alpha;
         foreach (var hiddenLayer in _hiddenLayers)
         {
            hiddenLayer.Alpha = alpha;
         }
      }

      public static NeuralNetworkBuilder Builder => new NeuralNetworkBuilder();

      public class NeuralNetworkBuilder
      {
         private double _alpha;

         internal NeuralNetworkBuilder()
         {
            HiddenLayers = new List<HiddenLayer>();
         }

         internal double Alpha
         {
            get => _alpha;
            set => _alpha = !value.Between(0, 1) ? throw new ArgumentException("Wrong alpha parameter") : (value);
         }

         private bool _softmax;
         internal List<HiddenLayer> HiddenLayers { get; set; }
         internal PredictLayer PredictionLayer { get; set; }
         private List<OutputNeuron> _predictionLayerNeurons;

         public NeuralNetworkBuilder UseSoftmax()
         {
            _softmax = true;
            return this;
         }

         public NeuralNetworkBuilder WithAlpha(double alpha)
         {
            Alpha = alpha;
            return this;
         }

         public NeuralNetworkBuilder AttachPredictionLayer(List<OutputNeuron> ons)
         {
            _predictionLayerNeurons = ons;
            return this;
         }

         public NeuralNetworkBuilder AddHiddenLayer(List<OutputNeuron> layerNeurons, Func<double, double> fx = null, Func<double, double> dfx = null)
         {
            HiddenLayers.Add(new HiddenLayer(layerNeurons, fx ?? ReluFunctions.Relu, dfx ?? ReluFunctions.ReluDerivative));
            return this;
         }

         public NeuralNetwork BuildNetwork()
         {
            PredictionLayer = new PredictLayer(_predictionLayerNeurons, _softmax);
            return new NeuralNetwork(this);
         }
      }

      public Matrix ExpectedOutputDoubles { get; set; }

      public (Matrix, Matrix, double) Train(Matrix expected, Matrix dataSeries, int iterations, bool logToFile = false)
      {
         var ans = new Matrix();
         var logger = new StringBuilder();
         var endError = 0.0;
         var endErrors = new Matrix();
         logger.Append($"[{DateTime.Now.ToShortDateString()} | { DateTime.Now.ToLongTimeString()}] [Start of learning]\n")
            .Append("Pre-Conditions:\n").Append($"\tExpected:\n {expected}\n")
            .Append($"\tCount of hidden layers: {_hiddenLayers.Count}\n").Append($"\tAlpha = {_predictLayer.Alpha}\n")
            .Append($"\tPrediction layer initial weights:\n{_predictLayer}\n");

         for (var i = 0; i < iterations; ++i, ++_currentIteration)
         {
            var errors = new Matrix(Tuple.Create(expected[0].Count, 1));
            var error = 0.0;
            for (var index = 0; index < dataSeries.Count; ++index)
            {
               var rowInput = dataSeries[index].ToInputNeurons();
               var expectedOutput = expected[index];

               #region Propagation

               foreach (var layer in _hiddenLayers)
               {
                  var res = layer.Proceed(rowInput);
                  rowInput = res.Data.ToInputNeurons();
               }

               ans = _predictLayer.Proceed(rowInput).Data.ToMatrix();

               #endregion

               #region BackPropagation

               var fAnswer = _predictLayer.GetDeltas(new EngineAnswer() { Data = expectedOutput });

               #region ErrorCummulation
               
               var seriesError = fAnswer.Deltas.Data.Sum(d => d * d);
               var seriesErrors = fAnswer.Deltas.Data.Select(d => d * d).ToList().ToMatrix();
               error += seriesError;
               errors += seriesErrors;
               logger.Append(
                     $"[{DateTime.Now.ToShortDateString()} | {DateTime.Now.ToLongTimeString()}] [Iteration No. {_currentIteration + 1}] [Series No. {index + 1}]\n")
                  .Append("Result:\n").Append($"\n{ans}\n").Append($"Error of each neuron at current series:\n\n{seriesErrors}\n")
                  .Append($"Cumulative error after current series: \n\n{seriesError}\n");

               #endregion

               foreach (var hiddenLayer in _hiddenLayers)
               {
                  fAnswer = hiddenLayer.BackPropagate(fAnswer);
               }

               #endregion
            }


            logger.Append(
                     $"[{DateTime.Now.ToShortDateString()} | {DateTime.Now.ToLongTimeString()}] [Iteration No. {_currentIteration + 1}]\n")
                  .Append("Result:\n").Append($"\n{ans}\n").Append($"Cumulative error of each neuron:\n\n{errors}\n")
                  .Append($"Total cumulative error after iteration step: \n\n{error}\n");
            endErrors = errors.ToMatrix();
            endError = error;
         }
         
         if (logToFile)
         {
            using var stream = new MemoryStream();
            using var streamWriter = new StreamWriter(stream, new UnicodeEncoding());
            streamWriter.Write(logger.ToString());
            streamWriter.Flush();
            stream.Seek(0, SeekOrigin.Begin);
            using var file = new FileStream($"log_{Guid.NewGuid().ToString().Substring(0, 5)}_{DateTime.Now.ToShortDateString()}_{DateTime.Now.ToLongTimeString()}.log", FileMode.Create, FileAccess.Write);
            stream.WriteTo(file);
         }

         LogReport?.Invoke(this, logger.ToString());

         return (ans, endErrors, endError);
      }

      public (Matrix, Matrix, double) Test(Matrix expected, Matrix dataSeries, bool logToFile = false)
      {

      }

      public int Learn(int iterations)
      {
         return iterations;
      }
   }
}
