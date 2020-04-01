using System;
using System.Collections.Generic;
using System.Linq;
using NNBasics.NNBasicsLimak.Core.Abstracts;
using NNBasics.NNBasicsLimak.Core.Models;
using NNBasics.NNBasicsLimak.Core.Neurons;
using NNBasics.NNBasicsLimak.Extensions;

namespace NNBasics.NNBasicsLimak.Core.Layers
{
   public class HiddenLayer : Layer
   {
      public delegate double ActivationFunction(double x);
      public delegate double ActivationFunctionDerivative(double x);

      private readonly ActivationFunction _activationFunction;
      private readonly ActivationFunctionDerivative _activationFunctionDerivative;

      public HiddenLayer(List<OutputNeuron> ons, Func<double, double> fx = null, Func<double,double> dfx = null) : base(ons)
      {
         _activationFunctionDerivative += d=>dfx?.Invoke(d)??1;
         _activationFunction += d => fx?.Invoke(d)??d;
      }

      public FeedbackAnswer BackPropagate(FeedbackAnswer previousLayerFeedbackAnswer, EngineAnswer thisLayerResponse)
      {
         var deltas = previousLayerFeedbackAnswer.Deltas;
         var matrix = deltas.Data.ToMatrix() * previousLayerFeedbackAnswer.Ons.ToMatrix();
         var data = matrix[0];
         data = data.Select((value, index) => value * _activationFunctionDerivative(thisLayerResponse.Data[index]))
            .ToList();

         var ans = new EngineAnswer(){Data = data};
         UpdateWeights(new GdEngineAnswer(thisLayerResponse, deltas));
         return new FeedbackAnswer(){Deltas = ans, Ons = Ons};
      }

      public new EngineAnswer Proceed(List<InputNeuron> ins)
      {
         var ans = base.Proceed(ins);
         var data = ans.Data.Select(arg => _activationFunction(arg)).ToList();
         return new EngineAnswer(){Data = data};
      }
   }
}