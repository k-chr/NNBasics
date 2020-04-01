using System;
using System.Collections.Generic;
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
      private List<InputNeuron> _ins;
      private List<HiddenLayer> _hiddenLayers;
      private PredictLayer _predictLayer;

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

         internal List<HiddenLayer> HiddenLayers { get; set; }
         internal PredictLayer PredictionLayer { get; set; }

         public NeuralNetworkBuilder WithAlpha(double alpha)
         {
            Alpha = alpha;
            return this;
         }

         public NeuralNetworkBuilder AttachPredictionLayer(List<OutputNeuron> ons)
         {
            PredictionLayer = new PredictLayer(ons);
            return this;
         }

         public NeuralNetworkBuilder AddHiddenLayer(List<OutputNeuron> layerNeurons, Func<double, double> fx = null, Func<double, double> dfx = null)
         {
            HiddenLayers.Add(new HiddenLayer(layerNeurons, fx ?? ReluFunctions.Relu, dfx ?? ReluFunctions.ReluDerivative));
            return this;
         }

         public NeuralNetwork BuildNetwork()
         {
            return new NeuralNetwork(this);
         }
      }

      public Matrix ExpectedOutputDoubles { get; set; }

     
      public int Learn(int iterations)
      {
         return iterations;
      }
   }
}
