using System;
using System.Collections.Generic;
using System.Linq;
using NNBasics.NNBasicsLimak.Core.Neurons;
using NNBasics.NNBasicsLimak.Core.UtilityTypes;

namespace NNBasics.NNBasicsLimak.Extensions
{
   public static class ListExtensions
   {

      public static List<OutputNeuron> ToOutputNeurons(this List<List<double>> data) => 
         data.Select(d => (OutputNeuron) d).ToList();

      public static List<InputNeuron> ToInputNeurons(this List<double> input) =>
         input.Select(d => (InputNeuron) d).ToList();

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

      public static List<double> Normalize(this List<double> input)
      {
         return input.Select(d => d / input.Sum()).ToList();
      }
   }
}
