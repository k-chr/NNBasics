using System;
using System.Collections.Generic;
using NNBasics.NNBasicsLimak.Core.Neurons;
using NNBasics.NNBasicsLimak.Core.UtilityTypes;

namespace NNBasics.NNBasicsLimak.Extensions
{
   public static class ListExtensions
   {
      public static Matrix ToMatrix(this List<double> input)
      {
         var mat = new Matrix((input.Count, 1).ToTuple()) {input};
         return mat;
      }

      public static Matrix ToMatrix(this List<List<double>> input)
      {
         var mat = new Matrix(input);
         return mat;
      }

      public static Matrix ToMatrix(this List<OutputNeuron> ons)
      {
         var mat = new Matrix();
         foreach (var outputNeuron in ons)
         {
            mat.Add(outputNeuron.Weights);
         }

         return mat;
      }
   }
}
