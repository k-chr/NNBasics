using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
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
         var l = new List<List<double>> {input};
         var mat = new Matrix(l);
         
         return mat;
      }

      public static Matrix ToMatrix(this List<List<double>> input) => new Matrix(input);

      public static Matrix ToMatrix(this List<OutputNeuron> ons)
      {
         var mat = new Matrix( ons.Select(neuron => neuron.Weights).ToList());
         return mat;
      }

      public static void Shuffle<T>(this IList<T> list)
      {
         var provider = new RNGCryptoServiceProvider();
         var n = list.Count;
         while (n > 1)
         {
            var box = new byte[1];
            do provider.GetBytes(box);
            while (!(box[0] < n * (byte.MaxValue / n)));
            var k = (box[0] % n);
            --n;
            var value = list[k];
            list[k] = list[n];
            list[n] = value;
         }
      }

      public static int ArgMax(this List<double> input) => input.Select((d, i) => (d, i)).Where(d => d.d == input.Max()).Select(d => d.i).First();


      public static List<double> Normalize(this List<double> input) => input.Select(d => d / input.Sum()).ToList();
   }
}
