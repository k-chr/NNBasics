using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.Core.Utilities.UtilityTypes
{
	public class Matrix : IDisposable, IEnumerable<double[]>
	{
		public int Rows { get; }

		public int Cols { get; }

		private readonly double[][] _data;

		public IEnumerator<double[]> GetEnumerator()
		{
			return (IEnumerator<double[]>)_data.GetEnumerator();
		}

		public IReadOnlyCollection<double> this[int x]
		{
			get => new ReadOnlyCollection<double>(_data[x]);
			set
			{
				if (value.Count != Cols) throw new ArgumentException($"Invalid size of collection, expected {Cols} but got {value.Count}");
				_data[x] = value.ToArray();
			}
		}

		public void ApplyFunction(Func<double, double> fun)
		{
			for (var i = 0; i < Rows; ++i)
			{
				for (var j = 0; j < Cols; ++j)
				{
					_data[i][j] = fun(_data[i][j]);
				}
			}
		}

		public double this[int x, int y]
		{
			get => _data[x][y];
			set => _data[x][y] = value;
		}

		public override string ToString()
		{
			var builder = new StringBuilder();
			foreach (var row in _data)
			{
				builder.Append("| ");
				foreach (var d in row)
				{
					builder.Append(d).Append(" ");
				}

				builder.Append("|\n");
			}

			return builder.ToString();
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			return GetEnumerator();
		}

		public Matrix(List<List<double>> values)
		{
			var cols = values[0].Count;
			if (values.Any(row => row.Count != cols))
			{
				throw new ArgumentException("Provided list of list of doubles isn't a good candidate to converse it into matrix");
			}

			Cols = cols;
			Rows = values.Count;

			_data = new double[Rows][];

			CreateRows();

			SetValues(values);

		}

		private void SetValues(IReadOnlyList<List<double>> values)
		{
			for (var i = 0; i < Rows; ++i)
			{
				for (var j = 0; j < Cols; ++j)
				{
					_data[i][j] = values[i][j];
				}
			}
		}

		private void CreateRows()
		{
			for (var i = 0; i < Rows; ++i)
			{
				_data[i] = new double[Cols];
			}
		}

		public Matrix(Tuple<int, int> size = null)
		{
			var (cols, rows) = size ?? Tuple.Create(1, 1);
			Rows = rows;
			Cols = cols;
			CreateRows();
			SetValues(0);
		}

		private Matrix(Matrix values)
		{
			Cols = values.Cols;
			Rows = values.Rows;
			CreateRows();
			for (var i = 0; i < Rows; ++i)
			{
				for (var j = 0; j < Cols; ++j)
				{
					_data[i][j] = values._data[i][j];
				}
			}
		}

		private void SetValues(double value)
		{
			for (var i = 0; i < Rows; ++i)
			{
				for (var j = 0; j < Cols; ++j)
				{
					_data[i][j] = value;
				}
			}
		}

		public Matrix Transpose()
		{
			var mat = this.SelectMany(inner => inner.Select((item, index) => new { item, index }))
				.GroupBy(i => i.index, i => i.item).ToMatrix();
			return mat;
		}

		public void AddMatrix(Matrix other)
		{
			if (Cols != other.Cols || Rows != other.Rows)
			{
				throw new ArgumentException("Addition cannot be performed, provided matrices don't match the rule of size matching");
			}

			var i = 0;
			foreach (var row in other)
			{
				var j = 0;

				foreach (var d in row)
				{
					_data[i][j++] += d;
				}

				++i;
			}
		}

		public static Matrix operator -(Matrix first, Matrix other)
		{
			if (first.Cols != other.Cols || first.Rows != other.Rows)
			{
				throw new ArgumentException("Addition cannot be performed, provided matrices don't match the rule of size matching");
			}

			return first.Select((row, rowId) => row.Zip(other[rowId], (d, d1) => d - d1)).ToMatrix();
		}

		public static Matrix operator +(Matrix first, Matrix other)
		{
			if (first.Cols != other.Cols || first.Rows != other.Rows)
			{
				throw new ArgumentException("Addition cannot be performed, provided matrices don't match the rule of size matching");
			}

			return first.Select((row, rowId) => row.Zip(other[rowId], (d, d1) => d + d1)).ToMatrix();
		}

		public static Matrix operator *(Matrix first, Matrix other)
		{
			if (first.Cols != other.Rows)
			{
				throw new ArgumentException("Multiplication cannot be performed, provided matrices don't match the rule: A.cols == B.rows, where A, B are matrices, cols is count of columns and rows is count of rows");
			}

			var mat = first.Select(
				(row, rowId) => other.Transpose()
											.Select((col, colId) => col.Zip(row, (colCell, rowCell) => colCell * rowCell).Sum()
											)
				).ToMatrix();

			return mat;
		}

		public static Matrix operator *(Matrix first, double alpha)
		{
			var mat = new Matrix(first);
			mat.ApplyFunction(d => d * alpha);
			return mat;
		}

		public Matrix HadamardProduct(Matrix other)
		{
			return this.Zip(other, (row1, row2) => row1.Zip(row2, (d1, d2) => d1 * d2)).ToMatrix();
		}

		public void Dispose()
		{
			GC.SuppressFinalize(this);
		}
	}
}
