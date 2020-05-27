using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using NNBasicsUtilities.Core.Models;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.Core.Abstracts
{
   public abstract class Layer
   {
      protected Matrix Ins;
      protected Matrix Ons;
      protected EngineAnswer LatestAnswer;
      protected EngineAnswer LatestDeltas;

      private double _alpha;

      public List<ImmutableList<double>> Weights => Ons.Select(neuron => neuron.ToImmutableList()).ToList();

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

      public EngineAnswer Proceed(Matrix input)
      {
         Ins = input;
         var ans = NeuralEngine.Proceed(input, Ons);
         LatestAnswer = new EngineAnswer(){Data = ans.Data};
         return ans;
      }

      protected Layer(Matrix ons)
      {
         Ons = ons;
      }

      protected void UpdateWeights(GdEngineAnswer answer)
      {
         var deltas = answer.Deltas;

         var mat = Ins.HadamardProduct(deltas.Data);
         Ons.SubtractMatrix(mat * Alpha);
      }
   }
}
