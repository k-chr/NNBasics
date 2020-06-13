using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Security.Cryptography;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.Core
{
   public class NeuralEngine
   {
      public static Matrix Proceed(Matrix iNs, Matrix oNs)
      {
	      //var time = Stopwatch.GetTimestamp();
	      var rV = iNs * oNs.Transpose();

         
         //time = Stopwatch.GetTimestamp() - time;
         //Console.WriteLine($"Time to perform proceed in NeuralEngine {time}");
         return rV;
      }

      public static Matrix GenerateRandomLayer(int cols, int rows, double min, double max)
      {
         var collection = new List<List<double>>();
         using var provider = new RNGCryptoServiceProvider();

         for (var i = 0; i < rows; ++i)
         {
            var row = new List<double>();
            for (var idx = 0; idx < cols; ++idx)
            {
               var bytes = new byte[2];
               double res;
               do
               {
                  provider.GetBytes(bytes);
                  res = BitConverter.ToInt16(bytes, 0) / 30000.0;
               } while (res < min || res > max || Math.Abs(res) < max / 100.0 || double.IsNaN(res) || Math.Abs(res) < 0.001);

               row.Add(res);

            }

            collection.Add(row);
         }

         return collection.ToMatrix();
      }
   }
}