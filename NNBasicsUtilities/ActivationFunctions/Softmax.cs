using System;
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

		public static void Softmax(this FlatMatrix m)
		{
			for (var i = 0; i < m.Rows; ++i)
			{
				var row = m[i];
				m[i] = row.Softmax();
			}
		}
	}
}