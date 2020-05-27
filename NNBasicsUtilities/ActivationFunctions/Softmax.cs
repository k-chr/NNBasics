using System;
using System.Linq;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.ActivationFunctions
{
   public static class SoftmaxFunction
   {
      public static double[] Softmax(this double[] input) => input.Select(Math.Exp).ToList().Normalize().ToArray();

      public static Matrix Softmax(this Matrix input)
      {
         var mat = new Matrix(new Tuple<int, int>(input.Rows, input.Cols));

         var i = 0;
         foreach (var row in input)
         {
	         var softRow = row.Softmax();
	         for (var j = 0; j < mat.Cols; ++j)
	         {
		         mat[i, j] = softRow[j];
	         }

	         ++i;
         }

         return mat;
      }
   }
}
