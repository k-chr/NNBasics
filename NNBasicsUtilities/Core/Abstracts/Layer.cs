using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using NNBasicsUtilities.Core.Models;
using NNBasicsUtilities.Core.Neurons;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.Core.Abstracts
{
   public abstract class Layer
   {
      protected List<InputNeuron> Ins;
      protected List<OutputNeuron> Ons;
      protected EngineAnswer LatestAnswer;
      protected EngineAnswer LatestDeltas;

      private double _alpha;

      public List<ImmutableList<double>> Weights => Ons.Select(neuron => neuron.Weights.ToImmutableList()).ToList();

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
         var ans = NeuralEngine.Proceed(input, Ons);
         LatestAnswer = new EngineAnswer(){Data = ans.Data.Select(d=>d).ToList()};
         return ans;
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
            var collection = Ins.Select(inputNeuron => inputNeuron.Value * deltas.Data[i]);
            outputNeuron.Weights = outputNeuron.Weights.Zip(
               collection,
               (weight, weightDelta) => weight - weightDelta * Alpha
            ).ToList();
            ++i;
         }

      }
   }
}
