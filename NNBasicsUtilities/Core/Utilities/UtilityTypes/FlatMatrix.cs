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

		public double this[int x, int y]
		{
			get => _data[x * Cols + y];
			set => _data[x * Cols + y] = value;
		}

		public void ApplyFunction(Func<double, double> foo)
		{
			for (var i = 0; i < _data.Length; _data[i] = foo(_data[i]), ++i) { }
		}

		public static FlatMatrix Of(int rows, int cols)
		{
			return new FlatMatrix(rows, cols);
		}

		public static FlatMatrix Of(FlatMatrix toCopy)
		{
			return new FlatMatrix(toCopy);
		}

		public FlatMatrix T()
		{
			var dst = new FlatMatrix(Cols, Rows);
			for (var n = 0; n < _data.Length; ++n)
			{
				var i = n / Rows;
				var j = n % Rows;
				dst._data[n] = _data[Cols * j + i];
			}

			return dst;
		}
		public void SubtractMatrix(ref FlatMatrix other)
		{
			if (Cols != other.Cols || Rows != other.Rows)
			{
				throw new ArgumentException("Subtraction cannot be performed, provided matrices don't match the rule of size matching");
			}
			for (var i = 0; i < _data.Length; _data[i] -= (other._data[i]), ++i) { }
		}

		public void AddMatrix(ref FlatMatrix other)
		{
			if (Cols != other.Cols || Rows != other.Rows)
			{
				throw new ArgumentException("Addition cannot be performed, provided matrices don't match the rule of size matching");
			}
			for (var i = 0; i < _data.Length; _data[i] += (other._data[i]), ++i) { }
		}

		public static FlatMatrix operator +(FlatMatrix first, FlatMatrix other)
		{
			if (first.Cols != other.Cols || first.Rows != other.Rows)
			{
				throw new ArgumentException("Addition cannot be performed, provided matrices don't match the rule of size matching");
			}

			var dst = new FlatMatrix(first);
			for (var i = 0; i < first._data.Length; dst._data[i] += other._data[i], ++i) { }
			return dst;
		}

		public static FlatMatrix operator -(FlatMatrix first, FlatMatrix other)
		{
			if (first.Cols != other.Cols || first.Rows != other.Rows)
			{
				throw new ArgumentException("Subtraction cannot be performed, provided matrices don't match the rule of size matching");
			}

			var dst = new FlatMatrix(first);
			for (var i = 0; i < first._data.Length; dst._data[i] -= other._data[i], ++i) { }
			return dst;
		}

		


	}
}
