using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using NNBasicsUtilities.Core.Neurons;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;

namespace NNBasicsUtilities.Extensions
{
   public static class ListExtensions
   {

      public static List<OutputNeuron> ToOutputNeurons(this List<List<double>> data) => 
         data.Select(d => (OutputNeuron) d).ToList();

      public static List<InputNeuron> ToInputNeurons(this List<double> input) =>
         input.Select(d => (InputNeuron) d).ToList();

      public static Matrix ToMatrix(this List<double> input)
      {
         var mat = new Matrix(( input.Count, 1).ToTuple());

         var i = 0;

         foreach (var d in input)
         {
            mat[0, i] = d;
            ++i;
         }

         return mat;
      }

      public static Matrix ToMatrix(this IReadOnlyCollection<double> row)
      {
	      var mat = new Matrix(new Tuple<int, int>(1, row.Count));
	      var i = 0;

	      foreach (var d in row)
	      {
		      mat[0, i++] = d;
	      }

	      return mat;
      }

      public static Matrix ToMatrix(this IEnumerable<IEnumerable<double>> data)
      {
         var cols = data.First().Count();
         var rows = data.Count();
         var mat = new Matrix(Tuple.Create(cols, rows));
         var i = 0;
         var j = 0;

         foreach (var row in data)
         {
            foreach (var d in row)
            {
               mat[i, j] = d;
               ++j;
            }

            j = 0;
            ++i;
         }

         return mat;
      }

      public static Matrix ToMatrix(this List<List<double>> input) => new Matrix(input);

      public static Matrix ToMatrix(this List<OutputNeuron> ons)
      {
         var mat = new Matrix(( ons[0].Weights.Count, ons.Count).ToTuple());
         var (i, j) = (0, 0);

         foreach (var outputNeuron in ons)
         {
            foreach (var d in outputNeuron.Weights)
            {
               mat[i, j] = d;
               ++j;
            }

            j = 0;
            ++i;
         }

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

      public static int ArgMax(this List<double> input)
      {
         var max = input.Max();
         return input.Select((d, i) => (d, i)).First(d => Math.Abs(d.d - max) < Tolerance).i;
      }

      private const double Tolerance = double.Epsilon;


      public static List<double> Normalize(this List<double> input) => input.Select(d => d / input.Sum()).ToList();
   }
}
