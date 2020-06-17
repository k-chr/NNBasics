using System;
using System.Linq;
using System.Text;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.Core.Utilities.UtilityTypes
{
	public class FlatMatrix
	{
		private double[] _data;
		public int Rows { get; private set; }
		public int Cols { get; private set; }
		private readonly FlatMatrix _transposed;
		private readonly FlatMatrix _row;

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

		public FlatMatrix()
		{
		}

		private FlatMatrix(int rows, int cols)
		{
			if (rows < 0 || cols < 0) throw new ArgumentException("Negative size is not supported");
			Rows = rows;
			Cols = cols;
			_data = new double[rows * cols];
			_transposed = new FlatMatrix {Rows = Cols, Cols = Rows, _data = new double[Rows * Cols]};
			_row = new FlatMatrix {Rows = 1, Cols = Cols, _data = new double[Cols]};
		}

		private FlatMatrix(FlatMatrix toCopy)
		{
			Rows = toCopy.Rows;
			Cols = toCopy.Cols;
			_data = new double[Rows * Cols];
			_transposed = Of(Cols, Rows);
			Buffer.BlockCopy(toCopy._data, 0, _data, 0, toCopy._data.Length * sizeof(double));
		}

		public double[] this[Index i]
		{
			get
			{
				const int doubleSize = sizeof(double);
				var newCols = Cols;
				var data = new double[newCols];
				var startInd = doubleSize * i.Value * Cols;
				var len = newCols * doubleSize;
				Buffer.BlockCopy(_data, startInd, data, 0, len);
				return data;
			}
			set
			{
				const int doubleSize = sizeof(double);
				var newCols = Cols;
				var startInd = doubleSize * i.Value * Cols;
				var len = newCols * doubleSize;
				Buffer.BlockCopy(value, 0, _data, startInd, len);
			}
		}

		public FlatMatrix GetRow(int i)
		{
			if (_row == null) return null;
			const int doubleSize = sizeof(double);
			var startInd = doubleSize * i * Cols;
			var len = Cols * doubleSize;
			Buffer.BlockCopy(_data, startInd, _row._data, 0, len);
			return _row;
		}

		public double this[int x, int y]
		{
			get => _data[x * Cols + y];
			set => _data[x * Cols + y] = value;
		}

		public FlatMatrix this[Range rows, Range cols]
		{
			get
			{
				const int doubleSize = sizeof(double);
				var newCols = cols.End.Value - cols.Start.Value;
				var newRows = rows.End.Value - rows.Start.Value;
				var startCol = cols.Start.Value;
				var startRow = rows.Start.Value;
				var data = new double[newRows * newCols];

				for (var i = 0; i < newRows; ++i)
				{
					var destInd = i * newCols * doubleSize;
					var startInd = ((startRow + i) * Cols + startCol) * doubleSize;
					var len = newCols * doubleSize;
					Buffer.BlockCopy(_data, startInd, data, destInd, len);
				}

				return new FlatMatrix {_data = data, Cols = newCols, Rows = newRows};
			}
			set
			{
				const int doubleSize = sizeof(double);
				var rowsCount = value.Rows;
				var colsCount = value.Cols;
				var startCol = cols.Start.Value;
				var startRow = rows.Start.Value;
				var buf = value._data;

				for (var i = 0; i < rowsCount; ++i)
				{
					var start = ((startRow + i) * Cols + startCol) * doubleSize;
					var srcStart = i * colsCount * doubleSize;
					Buffer.BlockCopy(buf, srcStart, _data, start, colsCount * doubleSize);
				}
			}
		}

		public void ApplyFunction(Func<double, double> foo)
		{
			var len = _data.Length;
			unsafe
			{
				fixed (double* arr1 = _data)
				{
					for (var i = 0; i < len; ++i)
					{
						*(arr1 + i) = foo(*(arr1 + i));
					}
				}
			}
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
			var cols = Cols;
			var len = _data.Length;
			var rows = Rows;

			unsafe
			{
				fixed (double* arr1 = _data, arr2 = _transposed._data)
				{
					for (var n = 0; n < len; ++n)
					{
						var i = n / rows;
						var j = n % rows;
						*(arr2 + n) = *(arr1 + (cols * j + i));
					}
				}
			}

			return _transposed;
		}

		public void SubtractMatrix(FlatMatrix other)
		{
			if (Cols != other.Cols || Rows != other.Rows)
			{
				throw new ArgumentException(
					"Subtraction cannot be performed, provided matrices don't match the rule of size matching");
			}

			var len = _data.Length;
			unsafe
			{
				fixed (double* arr1 = _data, arr2 = other._data)
				{
					for (var i = 0; i < len; ++i)
					{
						*(arr1 + i) -= *(arr2 + i);
					}
				}
			}
		}

		public static void SubtractMatrix(FlatMatrix first, FlatMatrix other, FlatMatrix result)
		{
			if (first.Cols != other.Cols || first.Rows != other.Rows)
			{
				throw new ArgumentException(
					"Subtraction cannot be performed, provided matrices don't match the rule of size matching");
			}

			var len = first._data.Length;
			unsafe
			{
				fixed (double* arr1 = first._data, arr2 = other._data, res = result._data)
				{
					for (var i = 0; i < len; ++i)
					{
						*(res + i) = *(arr1 + i) - *(arr2 + i);
					}
				}
			}
		}

		public void AddMatrix(FlatMatrix other)
		{
			if (Cols != other.Cols || Rows != other.Rows)
			{
				throw new ArgumentException(
					"Addition cannot be performed, provided matrices don't match the rule of size matching");
			}

			unsafe
			{
				fixed (double* arr1 = _data, arr2 = other._data)
				{
					for (var i = 0; i < _data.Length; ++i)
					{
						*(arr1 + i) += *(arr2 + i);
					}
				}
			}
		}

		public static void AddMatrix(FlatMatrix first, FlatMatrix other, FlatMatrix result)
		{
			if (first.Cols != other.Cols || first.Rows != other.Rows)
			{
				throw new ArgumentException(
					"Addition cannot be performed, provided matrices don't match the rule of size matching");
			}

			var len = first._data.Length;
			unsafe
			{
				fixed (double* arr1 = first._data, arr2 = other._data, arr3 = result._data)
				{
					for (var i = 0; i < len; ++i)
					{
						*(arr3 + i) = *(arr2 + i) + *(arr1 + i);
					}
				}
			}
		}

		public static void Multiply(FlatMatrix first, FlatMatrix other, FlatMatrix result)
		{
			if (first.Cols != other.Rows)
			{
				throw new ArgumentException(
					$"Multiplication cannot be performed, provided matrices don't match the rule of size left.Cols = {first.Cols} != right.Rows = {other.Rows} ");
			}

			var len = result._data.Length;
			var otherCols = other.Cols;
			var firstCols = first.Cols;
			unsafe
			{
				fixed (double* arr1 = first._data, arr2 = other._data, arr3 = result._data)
				{
					for (var n = 0; n < len; ++n)
					{
						var i = n / otherCols;
						var j = n % otherCols;
						var res = 0.0;
						for (var k = 0; k < firstCols; ++k)
						{
							res += *(arr1 + (i * firstCols + k)) * *(arr2 + (k * otherCols + j));
						}

						*(arr3 + (i * otherCols + j)) = res;
					}
				}
			}
		}


		public static void MultiplyByAlpha(FlatMatrix src, double alpha, FlatMatrix destination)
		{
			var len = src._data.Length;
			unsafe
			{
				fixed (double* arr1 = src._data, arr3 = destination._data)
				{
					for (var n = 0; n < len; ++n)
					{
						*(arr3 + n) = *(arr1 + n) * alpha;
					}
				}
			}
		}

		public void MultiplyByAlpha(double alpha)
		{
			var len = _data.Length;
			unsafe
			{
				fixed (double* arr1 = _data)
				{
					for (var n = 0; n < len; ++n)
					{
						*(arr1 + n) = *(arr1 + n) * alpha;
					}
				}
			}
		}

		public void HadamardProduct(FlatMatrix other, FlatMatrix mat)
		{
			if (Cols != other.Cols || Rows != other.Rows)
			{
				throw new ArgumentException(
					$"Hadamard product cannot be computed: ({Rows}, {Cols}) != ({other.Rows}, {other.Cols})");
			}

			var len = _data.Length;
			unsafe
			{
				fixed (double* arr1 = _data, arr2 = other._data, arr3 = mat._data)
				{
					for (var n = 0; n < len; ++n)
					{
						*(arr3 + n) = *(arr1 + n) * *(arr2 + n);
					}
				}
			}
		}

		public void HadamardProduct(FlatMatrix other)
		{
			if (Cols != other.Cols || Rows != other.Rows)
			{
				throw new ArgumentException(
					$"Hadamard product cannot be computed: ({Rows}, {Cols}) != ({other.Rows}, {other.Cols})");
			}

			var len = _data.Length;
			unsafe
			{
				fixed (double* arr1 = _data, arr2 = other._data)
				{
					for (var n = 0; n < len; ++n)
					{
						*(arr1 + n) = *(arr1 + n) * *(arr2 + n);
					}
				}
			}
		}


		public void Assign(FlatMatrix other)
		{
			var len = _data.Length;
			unsafe
			{
				fixed (double* arr1 = _data, arr2 = other._data)
				{
					for (var n = 0; n < len; ++n)
					{
						*(arr1 + n) = *(arr2 + n);
					}
				}
			}
		}

		public void Shuffle()
		{
			_data.Shuffle();
		}

		public double Min() => _data.Min();
		public double Max() => _data.Max();
	}
}