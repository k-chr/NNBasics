#undef Verbose
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NNBasicsUtilities.ActivationFunctions;
using NNBasicsUtilities.Core.Layers;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.Core
{
	//TODO Implement Neural network behaviour and add Layers members and backward propagation with activation functions
	public class NeuralNetwork
	{
		private readonly List<HiddenLayer> _hiddenLayers;
		private int _currentIteration;
		private readonly PredictLayer _predictLayer;
		public event EventHandler<string> LogReport;
		private bool _isLearned;
		private readonly string _name;

		private NeuralNetwork(NeuralNetworkBuilder neuralNetworkBuilder)
		{
			_name = neuralNetworkBuilder.Name;
			_hiddenLayers = neuralNetworkBuilder.HiddenLayers;
			_predictLayer = neuralNetworkBuilder.PredictionLayer;
			var alpha = Math.Max(neuralNetworkBuilder.Alpha, 0.005);
			_predictLayer.Alpha = alpha;
			foreach (var hiddenLayer in _hiddenLayers)
			{
				hiddenLayer.Alpha = alpha;
			}
		}

		public static NeuralNetworkBuilder Builder => new NeuralNetworkBuilder();

		public sealed class NeuralNetworkBuilder
		{
			public sealed class HiddenLayerBuilder
			{
				private readonly Matrix _layerNeurons;
				private Func<double, double> _fx;
				private Func<double, double> _dfx;
				private readonly NeuralNetworkBuilder _parentBuilder;
				private bool _dropout;
				private readonly double _defaultDropoutRate = 0.5;
				private double _dropoutRate;

				internal HiddenLayerBuilder(NeuralNetworkBuilder parentBuilder, Matrix layerNeurons)
				{
					_layerNeurons = layerNeurons;
					_parentBuilder = parentBuilder;
				}

				public HiddenLayerBuilder ApplyActivationFunction(Func<double, double> fx)
				{
					_fx = fx;
					return this;
				}

				public HiddenLayerBuilder ApplyActivationFunctionDerivative(Func<double, double> dfx)
				{
					_dfx = dfx;
					return this;
				}

				public HiddenLayerBuilder UseDropout()
				{
					_dropout = true;
					return this;
				}

				public HiddenLayerBuilder UseCustomDropoutRate(double rate)
				{
					_dropoutRate = rate;
					return this;
				}

				public NeuralNetworkBuilder BuildHiddenLayer()
				{
					_parentBuilder.HiddenLayers.Add(new HiddenLayer(_layerNeurons, _fx ?? ReluFunctions.Relu,
						_dfx ?? ReluFunctions.ReluDerivative, _dropout,
						_dropoutRate > 0 ? _dropoutRate : _defaultDropoutRate));
					return _parentBuilder;
				}
			}

			private double _alpha;

			internal NeuralNetworkBuilder()
			{
				HiddenLayers = new List<HiddenLayer>();
			}

			internal double Alpha
			{
				get => _alpha;
				private set => _alpha =
					!value.Between(0, 1) ? throw new ArgumentException("Wrong alpha parameter") : (value);
			}

			private bool _softmax;
			internal List<HiddenLayer> HiddenLayers { get; set; }
			internal PredictLayer PredictionLayer { get; set; }
			internal string Name { get; private set; }

			private Matrix _predictionLayerNeurons;

			public NeuralNetworkBuilder UseSoftmax()
			{
				_softmax = true;
				return this;
			}

			public NeuralNetworkBuilder ApplyTheNameOfYourNetwork(string name)
			{
				Name = name;
				return this;
			}

			public NeuralNetworkBuilder WithAlpha(double alpha)
			{
				Alpha = alpha;
				return this;
			}

			public NeuralNetworkBuilder AttachPredictionLayer(int rows, int cols, double max, double min)
			{
				_predictionLayerNeurons = NeuralEngine.GenerateRandomLayer(cols, rows, min, max);
				return this;
			}

			public HiddenLayerBuilder AttachHiddenLayer(int rows, int cols, double max, double min)
			{
				var mat = NeuralEngine.GenerateRandomLayer(cols, rows, min, max);
				return new HiddenLayerBuilder(this, mat);
			}

			public NeuralNetworkBuilder AttachPredictionLayer(Matrix ons)
			{
				_predictionLayerNeurons = ons;
				return this;
			}

			public HiddenLayerBuilder AddHiddenLayer(Matrix layerNeurons)
			{
				return new HiddenLayerBuilder(this, layerNeurons);
			}

			public NeuralNetwork BuildNetwork()
			{
				PredictionLayer = new PredictLayer(_predictionLayerNeurons, _softmax);
				return new NeuralNetwork(this);
			}
		}

		public (Matrix, Matrix, double) Train(Matrix expected, Matrix dataSeries, int iterations, int period = 1)
		{
			var ans = new Matrix();
			var endError = 0.0;
			var endErrors = new Matrix(Tuple.Create(1, _predictLayer.Weights.Count));

			var logger = Logger.Instance.StartSession(true, _name)
			   .LogPreconditions(_hiddenLayers.Count, _predictLayer.Alpha, _predictLayer);

			_isLearned = false;

			for (var i = 0; i < iterations; ++i, ++_currentIteration)
			{
				var errors = new Matrix(Tuple.Create(1, expected[0].Count));
				var error = 0.0;
				var accuracy = 0;
				for (var index = 0; index < dataSeries.Rows; ++index)
				{
					var rowInput = dataSeries[index].ToMatrix();
					var expectedOutput = expected[index].ToMatrix();

					#region Propagation

					//var time = Stopwatch.GetTimestamp();

					foreach (var layer in _hiddenLayers)
					{
						var res = layer.Proceed(rowInput);
						rowInput = res;
					}

					ans = _predictLayer.Proceed(rowInput);

					//time = Stopwatch.GetTimestamp() - time;
					//Console.WriteLine($"Propagation time: {time}");

					#endregion

					#region GetDeltasOnPredictionLayer

					var fAnswer = _predictLayer.GetDeltas(expectedOutput);

					#endregion

					#region ErrorCummulation

					var seriesError = fAnswer.Item1[0].Sum(d => d * d);
					var seriesErrors = fAnswer.Item1.HadamardProduct(fAnswer.Item1);
					error += seriesError;
					errors.AddMatrix(seriesErrors);

					if (fAnswer.Item1.Cols > 1)
					{
						accuracy += ans[0].ArgMax() == expectedOutput[0].ArgMax() ? 1 : 0;
					}

#if Verbose
                  logger = logger.LogSeriesError(seriesErrors, ans, seriesError, index + 1, expectedOutput.ToMatrix());
#endif

					#endregion

					#region BackPropagation

					//time = Stopwatch.GetTimestamp();

					foreach (var hiddenLayer in _hiddenLayers)
					{
						fAnswer = hiddenLayer.BackPropagate(fAnswer.Item1, fAnswer.Item2);
					}

					//time = Stopwatch.GetTimestamp() - time;
					//Console.WriteLine($"Back propagation time: {time}");

					#endregion

					#region UpdateWeights

					//time = Stopwatch.GetTimestamp();


					_predictLayer.Update();

					foreach (var hiddenLayer in _hiddenLayers)
					{
						hiddenLayer.Update();
					}

					//time = Stopwatch.GetTimestamp() - time;
					//Console.WriteLine($"Update time: {time}");

#if Verbose
                  //logger = logger.LogLayerInfo(_predictLayer, _hiddenLayers);
#endif

					#endregion
				}

				if ((_currentIteration + 1) % period == 0)
					logger = logger.LogIteration(_currentIteration + 1, _predictLayer, errors, error, accuracy,
						expected.Cols > 1 ? expected.Rows : 0);
				endErrors = errors;
				endError = error;
			}

			LogReport?.Invoke(this, logger.ToString());
			logger.EndSession();

			_isLearned = true;

			return (ans, endErrors, endError);
		}

		public (Matrix, Matrix, double) Test(Matrix expected, Matrix dataSeries)
		{
			if (!_isLearned)
			{
				throw new AccessViolationException(
					"Network has never been trained before!!! What is your reason for running tests before teaching your network how to fit its answer according to provided input?");
			}

			var logger = Logger.Instance.StartSession(name: _name)
			   .LogPreconditions(_hiddenLayers.Count, _predictLayer.Alpha, _predictLayer);

			var ans = new Matrix();
			var endError = 0.0;
			var endErrors = new Matrix(Tuple.Create(1, _predictLayer.Weights.Count));
			var accuracy = 0;

			for (var index = 0; index < dataSeries.Rows; ++index)
			{
				var rowInput = dataSeries[index].ToMatrix();
				var expectedOutput = expected[index].ToMatrix();

				#region Propagation

				foreach (var layer in _hiddenLayers)
				{
					var res = layer.Proceed(rowInput);
					rowInput = res;
				}

				ans = _predictLayer.Proceed(rowInput);

				#endregion

				#region GetDeltasOnPredictionLayer

				var fAnswer = _predictLayer.GetDeltas(expectedOutput);

				#endregion

				#region ErrorCummulation

				var seriesError = fAnswer.Item1[0].Sum(d => d * d);
				var seriesErrors = fAnswer.Item1.HadamardProduct(fAnswer.Item1);
				endError += seriesError;
				endErrors.AddMatrix(seriesErrors);

				if (fAnswer.Item1.Cols > 1)
				{
					accuracy += ans[0].ArgMax() == expectedOutput[0].ArgMax() ? 1 : 0;
				}

				logger = logger.LogSeriesError(seriesErrors, ans, seriesError, index, expectedOutput.ToMatrix());

				#endregion
			}

			logger.LogTestFinalResults(_predictLayer, endErrors, endError, accuracy,
				expected.Cols > 1 ? expected.Rows : 0);

			LogReport?.Invoke(this, logger.ToString());
			logger.EndSession();

			return (ans, endErrors, endError);
		}
	}
}