using System;
using System.Diagnostics;
using System.Globalization;
using NNBasicsUtilities.Core;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;
using Xunit;
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
			_flat.ApplyFunction(f => 3);
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

	
		
	}
}
