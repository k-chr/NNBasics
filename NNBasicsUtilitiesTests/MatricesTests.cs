using System;
using System.Diagnostics;
using System.Globalization;
using NHamcrest;
using NNBasicsUtilities.ActivationFunctions;
using NNBasicsUtilities.Core;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;
using Xunit;
using Assert = NHamcrest.XUnit.Assert;
using Xunit.Abstractions;

namespace NNBasicsUtilitiesTests
{
	public class MatricesTests
	{
		private readonly ITestOutputHelper _testOutputHelper;
		private Matrix _matrix;
		private Matrix _range;
		private FlatMatrix _flat;
		double d = 8.8;
		double dd = 4.4;

		public MatricesTests(ITestOutputHelper testOutputHelper)
		{
			_testOutputHelper = testOutputHelper;
			_matrix = NeuralEngine.GenerateRandomLayer(500, 500, 0.4, 0.9);
			_range = new Matrix((3, 3).ToTuple());
			_range.ApplyFunction(d => 1);
			_flat = FlatMatrix.Of(500, 500);
			_flat.ApplyFunction(f => 5);
		}

		[Fact]
		public void FlatMatrixSoftmaxTest()
		{
			_flat[0, 0] = 33;
			var start = Stopwatch.GetTimestamp();
			var mat = _flat.Softmax();
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"FlatMatrix softmax cycles: {start}");
			_testOutputHelper.WriteLine(mat.ToString());
		}

		[Fact]
		public void RangeOperationTests()
		{
			_testOutputHelper.WriteLine(_matrix.ToString());
			var range = _matrix[..3, ..3];
			_testOutputHelper.WriteLine(range.ToString());
			range.ApplyFunction(d => 1);
			_testOutputHelper.WriteLine(range.ToString());
			_testOutputHelper.WriteLine(_matrix.ToString());
			_matrix[..3, ..3] = range;
			_testOutputHelper.WriteLine(_matrix.ToString());
		}

		[Fact]
		public void SetRangeTest()
		{
			var start = Stopwatch.GetTimestamp();
			_matrix[..3, ..3] = _range;
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Range assignment cycles: {start}");
		}

		[Fact]
		public void GetRangeTest()
		{
			var start = Stopwatch.GetTimestamp();
			var range = _matrix[..3, ..3];
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Get range cycles: {start}");
		}

		[Fact]
		public void TransposeTest()
		{
			var start = Stopwatch.GetTimestamp();
			_matrix = _matrix.Transpose();
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Transpose cycles: {start}");
		}

		[Fact]
		public void MultiplicationTest()
		{
			var start = Stopwatch.GetTimestamp();

			_matrix *= _matrix;
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Multiplication cycles: {start}");
		}

		[Fact]
		public void HadamardProductTest()
		{
			var start = Stopwatch.GetTimestamp();

			_matrix = _matrix.HadamardProduct(_matrix);

			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Hadamard Multiplication cycles: {start}");
		}

		[Fact]
		public void SingleMultiplicationTest()
		{
			var start = Stopwatch.GetTimestamp();
			var double_ = d * dd;
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"single double Multiplication cycles: {start}");

			_testOutputHelper.WriteLine(double_.ToString(CultureInfo.InvariantCulture));
		}

		[Fact]
		public void MultipleDoublesTest()
		{
			double double_ = 0.0;

			var start = Stopwatch.GetTimestamp();
			if (_matrix.Cols != _matrix.Cols) throw new ArgumentException();
			for (var i = 0; i < _matrix.Rows; ++i)
			{
				for (var j = 0; j < _matrix.Cols; ++j)
				{
					for (var k = 0; k < _matrix.Cols; ++k)
					{
						double_ = _matrix[i, j] * _matrix[i, j];
						_matrix[i, j] += double_;
					}
				}
			}

			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"single double Multiplication cycles: {start}");
			_testOutputHelper.WriteLine(double_.ToString(CultureInfo.InvariantCulture));
		}

		[Fact]
		public void FlatMatrixTransposeTest()
		{
			var start = Stopwatch.GetTimestamp();
			var transposed = _flat.T();
			start = Stopwatch.GetTimestamp() - start;

			_testOutputHelper.WriteLine($"Transpose cycles: {start}");
		}

		[Fact]
		public void FlatMatrixAddTest()
		{
			var transposed = _flat.T();
			_testOutputHelper.WriteLine(_flat.ToString());
			var start = Stopwatch.GetTimestamp();
			_flat.AddMatrix(transposed);
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"After add:\n{_flat}");
			_testOutputHelper.WriteLine($"Addition cycles: {start}");
		}

		[Fact]
		public void FlatMatrixPlusOpVsRefAddMatrix()
		{
			var transposed = _flat.T();
			var start = Stopwatch.GetTimestamp();
			var mat = FlatMatrix.AddMatrix(_flat,  transposed);
			start = Stopwatch.GetTimestamp() - start;
			//_testOutputHelper.WriteLine($"After add:\n{_flat}");
			_testOutputHelper.WriteLine($"Ref Addition cycles: {start}");

			start = Stopwatch.GetTimestamp();
			mat = _flat + transposed;
			start = Stopwatch.GetTimestamp() - start;
			//_testOutputHelper.WriteLine($"After add:\n{_flat}");
			_testOutputHelper.WriteLine($"Op Addition cycles: {start}");
		}

		[Fact]
		public void MatricesAddTest()
		{
			var mat = _matrix.ToMatrix();

			_testOutputHelper.WriteLine(_matrix.ToString());
			var start = Stopwatch.GetTimestamp();
			_matrix.AddMatrix(mat);
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"After add:\n{_matrix}");
			_testOutputHelper.WriteLine($"Addition cycles: {start}");
		}

		[Fact]
		public void AddRefFlatMatrix()
		{
			var start = Stopwatch.GetTimestamp();
			var mat = FlatMatrix.AddMatrix(_flat,  _flat);
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Addition op cycles: {start}");
		}

		[Fact]
		public void GetRowFromFlatMatrix()
		{
			_flat[200, 0] = 123;
			var start = Stopwatch.GetTimestamp();
			var mat = _flat[200];
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Get row op cycles: {start}");
			_testOutputHelper.WriteLine(mat.ToString());
		}

		[Fact]
		public void GetRowFromFlatMatrixAsArray()
		{
			_flat[200, 0] = 123;
			var start = Stopwatch.GetTimestamp();
			var mat = _flat[(Index) 200];
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Get row op cycles: {start}");
			_testOutputHelper.WriteLine(mat.ToString());
		}

		[Fact]
		public void SetRowOfFlatMatrixAsArray()
		{
			_flat[200, 0] = 123;
			var mat = _flat[(Index) 200];
			mat[1] = 0xD;
			var start = Stopwatch.GetTimestamp();
			_flat[(Index) 200] = mat;
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Set row op cycles: {start}");
			_testOutputHelper.WriteLine(_flat[200].ToString());
		}

		[Fact]
		public void AddOperatorFlatMatTest()
		{
			var start = Stopwatch.GetTimestamp();
			var mat = _flat + _flat;
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Addition op cycles: {start}");
		}

		[Fact]
		public void AddOperatorNotFlatMatTest()
		{
			var start = Stopwatch.GetTimestamp();
			var mat = _matrix + _matrix;
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Addition op cycles: {start}");
		}

		[Fact]
		public void CopyFlatTest()
		{
			var start = Stopwatch.GetTimestamp();
			var mat = FlatMatrix.Of(_flat);
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Copy op cycles: {start}");
			mat[0, 0] = 2137;
			Assert.That(mat[0, 0], Is.Not(_flat[0, 0]));
		}

		[Fact]
		public void MulOpTestFlat()
		{
			var start = Stopwatch.GetTimestamp();
			var mat3 = _flat * _flat;
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Mul flat op cycles: {start}");
			_testOutputHelper.WriteLine(mat3.ToString());
		}

		[Fact]
		public void TwoLoopMulOpTest()
		{
			var start = Stopwatch.GetTimestamp();
			var mat3 = FlatMatrix.Multiply( _flat, _flat);
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Mul flat op cycles: {start}");
			_testOutputHelper.WriteLine(mat3.ToString());
		}

		[Fact]
		public void RangeFlatTest()
		{
			var start = Stopwatch.GetTimestamp();
			var mat2 = _flat[..3, ..3];
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Range copy cycles: {start}");
			_testOutputHelper.WriteLine(mat2.ToString());
		}

		[Fact]
		public void RangeSetFlatTest()
		{
			var mat2 = _flat[..3, ..3];
			mat2.ApplyFunction(d1 => 48);
			//_testOutputHelper.WriteLine(_flat.ToString());

			var start = Stopwatch.GetTimestamp();
			_flat[..3, ..3] = mat2;
			start = Stopwatch.GetTimestamp() - start;
			_testOutputHelper.WriteLine($"Range copy cycles: {start}");
			_testOutputHelper.WriteLine(_flat.ToString());
		}
	}
}