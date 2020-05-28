using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading;
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
			return _data.AsEnumerable().GetEnumerator();
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
			var (rows, cols) = size ?? Tuple.Create(1, 1);
			Rows = rows;
			Cols = cols;
			_data = new double[Rows][];
			CreateRows();
			SetValues(0);
		}

		private Matrix(Matrix values)
		{
			Cols = values.Cols;
			Rows = values.Rows;
			_data = new double[Rows][];
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

		public void SubtractMatrix(Matrix other)
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
					_data[i][j++] -= d;
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


			var mat = new Matrix((first.Rows, other.Cols).ToTuple());
			mat.SetValues(0);
			
			var threads = new Thread[mat.Rows][];

			for (var index = 0; index < threads.Length; index++)
			{
				threads[index] = new Thread[mat.Cols];
			}


			for (var i = 0; i < mat.Rows; ++i)
			{
				for (var j = 0; j < mat.Cols; ++j)
				{
					var (x, y) = (i, j);
					////for (var k = 0; k < other.Rows; ++k)
					////{
					////	mat._data[i][j] += first._data[i][k] * other._data[k][j];
					////}
					//
					//mat._data[x][y] = ComputeCell(x, y, first.Cols, first, other);
					threads[i][j] = new Thread(_ => mat._data[x][y] = ComputeCell(x, y, other.Rows, first, other));
					threads[i][j].Start();
				}
			}

			foreach (var threadsArr in threads)
			{
				foreach (var t in threadsArr)
				{
					t.Join();
				}
			}

			//first.Select(
			//	(row, rowId) => other.Select((col, colId) => col.Zip(row, (colCell, rowCell) => colCell * rowCell).Sum()
			//								)
			//	).ToMatrix();

			return mat;
		}

		private static double ComputeCell(int x, int y, int z, Matrix sourceLeft, Matrix sourceRight)
		{
			var result = 0.0;

			for (var i = 0; i < z; ++i)
			{
				result += sourceLeft._data[x][i] * sourceRight._data[i][y];
			}

			return result;
		}

		public static Matrix operator *(Matrix first, double alpha)
		{
			var mat = new Matrix(first);
			mat.ApplyFunction(d => d * alpha);
			return mat;
		}

		public Matrix HadamardProduct(Matrix other)
		{
			if (this.Cols != other.Cols || this.Rows != other.Rows)
			{
				throw new ArgumentException($"Hadamard product cannot be computed: ({this.Rows}, {this.Cols}) != ({other.Rows}, {other.Cols})");
			}
			var mat = new Matrix(this);

			return this.Zip(other, (row1, row2) => row1.Zip(row2, (d1, d2) => d1 * d2)).ToMatrix();
		}

		public void Dispose()
		{
			GC.SuppressFinalize(this);
		}
	}
}
