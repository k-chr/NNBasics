using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NNBasicsUtilities.Core.Abstracts;
using NNBasicsUtilities.Core.Models;
using NNBasicsUtilities.Core.Neurons;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.Core.Layers
{
   public class HiddenLayer : Layer
   {
      public delegate double ActivationFunction(double x);
      public delegate double ActivationFunctionDerivative(double x);

      private readonly ActivationFunction _activationFunction;
      private readonly ActivationFunctionDerivative _activationFunctionDerivative;
      private readonly double _dropoutRate;
      private readonly bool _applyDropout;
      private readonly List<double> _dropout;

      public HiddenLayer(List<OutputNeuron> ons, Func<double, double> fx = null, Func<double, double> dfx = null, bool dropout = false, double dropoutRate = 0) : base(ons)
      {
         _activationFunctionDerivative += d => dfx?.Invoke(d) ?? 1;
         _applyDropout = dropout;
        
         _activationFunction += d => fx?.Invoke(d) ?? d;

         if (_applyDropout)
         {
            var len = ons[0].Weights.Count;
            var fill = (int)(len * dropoutRate);
            var trueDropout = fill / (double)len;
            _dropoutRate = trueDropout;
            _dropout = GenerateDropout(); 
         }
      }

      private List<double> GenerateDropout()
      {
         var l = new List<double>();
         var count = Ons[0].Weights.Count;
         var fill = _dropoutRate * count;

         for (var i = 0; i < count; ++i)
         {
            l[i] = i < fill ? 1 : 0;
         }

         l.Shuffle();

         return l;
      }

      public FeedbackAnswer BackPropagate(FeedbackAnswer previousLayerFeedbackAnswer)
      {
         var thisLayerResponse = LatestAnswer;
         var deltas = previousLayerFeedbackAnswer.Deltas;
         var matrix = deltas.Data.ToMatrix() * previousLayerFeedbackAnswer.Ons.ToMatrix();
         var data = matrix[0];
         data = data.Select((value, index) => value * _activationFunctionDerivative(thisLayerResponse.Data[index]))
            .ToList();
         
         if (_applyDropout)
         {
            var mat = data.ToMatrix().HadamardProduct(_dropout.ToMatrix());
            data = mat[0];
         }

         var ans = new EngineAnswer() { Data = data };
         LatestDeltas = ans;
         return new FeedbackAnswer() { Deltas = ans, Ons = Ons };
      }

      public void Update()
      {
         UpdateWeights(new GdEngineAnswer(LatestAnswer, LatestDeltas));
      }

      public new EngineAnswer Proceed(List<InputNeuron> ins)
      {
         var ans = base.Proceed(ins);
         var data = ans.Data.Select(arg => _activationFunction(arg)).ToList();

         if (_applyDropout)
         {
            var mat = data.ToMatrix();
            var r = new Random();
            var bound = r.Next(10);

            for (var j = 0; j < bound; ++j)
            {
               _dropout.Shuffle();
            }

            mat = mat.HadamardProduct(_dropout.ToMatrix());

            data = mat[0];
         }

         return new EngineAnswer() { Data = data };
      }

      public override string ToString()
      {
         var builder = new StringBuilder();
         foreach (var outputNeuron in Ons)
         {
            builder.Append(new EngineAnswer() { Data = outputNeuron.Weights.Select(d => d).ToList() });
         }

         return builder.ToString();
      }
   }
}