using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;

namespace NNBasicsUtilities.Core.FlatCore.FlatNN
{
   public static class NeuralEngine
   {
	   public static void Proceed(FlatMatrix iNs, FlatMatrix oNs, FlatMatrix result)
	   {
		 //var time = Stopwatch.GetTimestamp();
		 FlatMatrix.Multiply(iNs, oNs.T(), result);
		 //time = Stopwatch.GetTimestamp() - time;
		 //Console.WriteLine($"Time to perform proceed in NeuralEngine {time}");
	   }

	   public static FlatMatrix GenerateRandomLayer(int cols, int rows, double min, double max)
	   {
		   using var provider = new RNGCryptoServiceProvider();
		   var mat = FlatMatrix.Of(rows, cols);
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
				   } while (res < min || res > max || Math.Abs(res) < max / 100.0 || double.IsNaN(res) ||
							Math.Abs(res) < 0.001);

				   mat[i, idx] = res;
			   }

			  
		   }

		   return mat;
	   }
   }
}
