using System;
using System.Linq;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.ActivationFunctions
{
	public static class SoftmaxFunction
	{
		public static double[] Softmax(this double[] input)
		{
			var len = input.Length;
			var output = new double[len];
			for (var i = 0; i < len; ++i)
			{
				output[i] = Math.Exp(input[i]);
			}
			return output.Normalize();
		}

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

		public static FlatMatrix Softmax(this FlatMatrix flatMatrix)
		{
			var mat = FlatMatrix.Of(flatMatrix.Rows, flatMatrix.Cols);
			for (var i = 0; i < flatMatrix.Rows; ++i)
			{
				var row = flatMatrix[(Index) i];
				mat[(Index) i] = row.Softmax();
			}

			return mat;
		}
	}
}