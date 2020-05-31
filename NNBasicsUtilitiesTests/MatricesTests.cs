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


		
		
	}
}
