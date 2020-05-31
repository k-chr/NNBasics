using System;
using System.Text;

namespace NNBasicsUtilities.Core.Utilities.UtilityTypes
{
	public struct FlatMatrix
	{
		private readonly double[] _data;
		public readonly int Rows;
		public readonly int Cols;

		public override string ToString()
		{
			var builder = new StringBuilder();
			for (var i = 0; i < _data.Length; ++i)
			{
				if (i % (Cols) == 0)
				{
					builder.Append("| ");
				}
				builder.Append(_data[i]).Append(' ');
				if ((i + 1) % (Cols) == 0)
				{
					builder.Append("|\n");
				}
			}

			return builder.ToString();
		}

		private FlatMatrix(in int rows, in int cols)
		{
			if (rows < 0 || cols < 0) throw new ArgumentException("Negative size is not supported");
			Rows = rows;
			Cols = cols;
			_data = new double[rows * cols];
		}

		private FlatMatrix(FlatMatrix toCopy)
		{
			Rows = toCopy.Rows;
			Cols = toCopy.Cols;
			_data = new double[Rows * Cols];
			Buffer.BlockCopy(toCopy._data, 0, _data, 0, toCopy._data.Length);
		}


	}
}
