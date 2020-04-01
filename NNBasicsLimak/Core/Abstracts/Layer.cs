using System;
using System.Collections.Generic;
using System.Linq;
using NNBasics.NNBasicsLimak.Core.Models;
using NNBasics.NNBasicsLimak.Core.Neurons;
using NNBasics.NNBasicsLimak.Extensions;

namespace NNBasics.NNBasicsLimak.Core.Abstracts
{
   public abstract class Layer
   {
      protected List<InputNeuron> Ins;
      protected List<OutputNeuron> Ons;

      private double _alpha;

      public double Alpha
      {
         get => _alpha;
         set
         {
            const double min = 0.0;
            const double max = 1;
            if (!value.Between(min, max))
            {
               throw new ArgumentException($"Provided value: {value} is out of range of <{min}, {max}>");
            }

            _alpha = value;
         }
      }

      public EngineAnswer Proceed(List<InputNeuron> input)
      {
         Ins = input;
         return NeuralEngine.Proceed(input, Ons);
      }

      protected Layer(List<OutputNeuron> ons)
      {
         Ons = ons;
      }

      protected void UpdateWeights(GdEngineAnswer answer)
      {
         var deltas = answer.Deltas;
         var i = 0;
         foreach (var outputNeuron in Ons)
         {
            var collection = Ins.Select(inputNeuron => inputNeuron.Value * deltas.Data[i]).ToList();
            outputNeuron.Weights = outputNeuron.Weights.Zip(
               collection,
               (weight, weightDelta) => weight - weightDelta * Alpha
            ).ToList();
            ++i;
         }

      }
   }
}
