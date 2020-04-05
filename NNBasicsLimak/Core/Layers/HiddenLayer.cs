using System;
using System.Collections.Generic;
using System.Linq;
using NNBasics.NNBasicsLimak.Core.Abstracts;
using NNBasics.NNBasicsLimak.Core.Models;
using NNBasics.NNBasicsLimak.Core.Neurons;
using NNBasics.NNBasicsLimak.Core.UtilityTypes;
using NNBasics.NNBasicsLimak.Extensions;

namespace NNBasics.NNBasicsLimak.Core.Layers
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
         UpdateWeights(new GdEngineAnswer(thisLayerResponse, deltas));
         return new FeedbackAnswer() { Deltas = ans, Ons = Ons };
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
   }
}