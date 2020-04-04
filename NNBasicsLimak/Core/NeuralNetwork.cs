using System;
using System.Collections.Generic;
using System.Linq;
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
      private int _currentIteration;
      private readonly PredictLayer _predictLayer;
      public event EventHandler<string> LogReport;
      private bool _isLearned;

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



      public (Matrix, Matrix, double) Train(Matrix expected, Matrix dataSeries, int iterations)
      {
         var ans = new Matrix();
         var endError = 0.0;
         var endErrors = new Matrix();

         var logger = Logger.Instance.StartSession(true)
            .LogPreconditions(_hiddenLayers.Count, _predictLayer.Alpha, _predictLayer);

         _isLearned = false;

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

               #region GetDeltasOnPredictionLayer

               var fAnswer = _predictLayer.GetDeltas(new EngineAnswer() { Data = expectedOutput });

               #endregion

               #region ErrorCummulation

               var seriesError = fAnswer.Deltas.Data.Sum(d => d * d);
               var seriesErrors = fAnswer.Deltas.Data.Select(d => d * d).ToList().ToMatrix();
               error += seriesError;
               errors += seriesErrors;

               logger = logger.LogSeriesError(seriesErrors, ans, seriesError, index, expectedOutput.ToMatrix());

               #endregion

               #region BackPropagation

               foreach (var hiddenLayer in _hiddenLayers)
               {
                  fAnswer = hiddenLayer.BackPropagate(fAnswer);
               }

               #endregion
            }

            logger = logger.LogIteration(_currentIteration + 1, _predictLayer, errors, error);
            endErrors = errors.ToMatrix();
            endError = error;
         }
         
         LogReport?.Invoke(this, logger.ToString());
         logger.EndSession();

         _isLearned = true;

         return (ans, endErrors, endError);
      }

      public (Matrix, Matrix, double) Test(Matrix expected, Matrix dataSeries)
      {

         if (!_isLearned)
         {
            throw new AccessViolationException("Network has never been trained before!!! What is your reason for running tests before teaching your network how to fit its answer according to provided input?");
         }

         var logger = Logger.Instance.StartSession(true)
            .LogPreconditions(_hiddenLayers.Count, _predictLayer.Alpha, _predictLayer);

         var ans = new Matrix();
         var endError = 0.0;
         var endErrors = new Matrix();

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

            #region GetDeltasOnPredictionLayer

            var fAnswer = _predictLayer.GetDeltas(new EngineAnswer() {Data = expectedOutput});

            #endregion

            #region ErrorCummulation

            var seriesError = fAnswer.Deltas.Data.Sum(d => d * d);
            var seriesErrors = fAnswer.Deltas.Data.Select(d => d * d).ToList().ToMatrix();
            endError += seriesError;
            endErrors += seriesErrors;

            logger = logger.LogSeriesError(seriesErrors, ans, seriesError, index, expectedOutput.ToMatrix());

            #endregion
         }

         logger.LogTestFinalResults(_predictLayer, endErrors, endError);

         LogReport?.Invoke(this, logger.ToString());
         logger.EndSession();

         return (ans, endErrors, endError);
      }

   }
}