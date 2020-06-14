using System;
using System.Collections.Generic;
using System.Diagnostics;
using NNBasicsUtilities.Core.FlatCore.FlatAbstracts;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.Core.FlatCore.FlatLayers
{
	public class HiddenLayer : Layer
	{
		public delegate double ActivationFunction(double x);

		public delegate double ActivationFunctionDerivative(double x);

		private readonly ActivationFunction _activationFunction;
		private readonly ActivationFunctionDerivative _activationFunctionDerivative;
		private readonly double _dropoutRate;
		private readonly bool _applyDropout;
		private FlatMatrix _dropout;

		public HiddenLayer(FlatMatrix ons, Func<double, double> fx = null, Func<double, double> dfx = null,
			bool dropout = false, double dropoutRate = 0) : base(ons)
		{
			_activationFunctionDerivative += d => dfx?.Invoke(d) ?? 1;
			_applyDropout = dropout;

			_activationFunction += d => fx?.Invoke(d) ?? d;

			if (_applyDropout)
			{
				var len = ons.Cols;
				var fill = (int) (len * dropoutRate);
				var trueDropout = fill / (double) len;
				_dropoutRate = trueDropout;
				_dropout = GenerateDropout();
			}
		}

		private FlatMatrix GenerateDropout()
		{
			var l = new List<double>();
			var count = Ons.Cols;
			var fill = _dropoutRate * count;

			for (var i = 0; i < count; ++i)
			{
				l[i] = i < fill ? 1 : 0;
			}

			l.Shuffle();
			var mat = FlatMatrix.Of(1, l.Count);
			mat[(Index) 0] = l.ToArray();
			return mat;
		}

		public (FlatMatrix, FlatMatrix) BackPropagate(FlatMatrix deltas, FlatMatrix ons)
		{
			ref var thisLayerResponse = ref LatestAnswer;
			var data = deltas * ons;
			thisLayerResponse.ApplyFunction(d => _activationFunctionDerivative(d));
			data = data.HadamardProduct(thisLayerResponse);

			if (_applyDropout)
			{
				data = data.HadamardProduct(_dropout);
			}

			LatestDeltas = data;
			return (LatestDeltas, Ons);
		}

		public void Update()
		{
			UpdateWeights(LatestDeltas);
		}

		public new FlatMatrix Proceed(FlatMatrix ins)
		{
			var time = Stopwatch.GetTimestamp();
			var subTime = time;
			var ans = base.Proceed(ins);
			subTime = Stopwatch.GetTimestamp() - subTime;
			Console.WriteLine($"Proceed time in hidden layer calling base method: {subTime}");
			subTime = Stopwatch.GetTimestamp();
			ans.ApplyFunction(d => _activationFunction(d));
			subTime = Stopwatch.GetTimestamp() - subTime;
			Console.WriteLine($"Proceed time in hidden layer applying activation function: {subTime}");
			if (_applyDropout)
			{
				var mat = ans;
				var r = new Random();
				var bound = r.Next(10);

				for (var j = 0; j < bound; ++j)
				{
					_dropout.Shuffle();
				}

				mat = mat.HadamardProduct(_dropout);

				ans = mat;
			}

			//time = Stopwatch.GetTimestamp() - time;
			//Console.WriteLine($"Proceed time in hidden layer in total: {time}");
			return ans;
		}

		public override string ToString()
		{
			return Ons.ToString();
		}
	}
}