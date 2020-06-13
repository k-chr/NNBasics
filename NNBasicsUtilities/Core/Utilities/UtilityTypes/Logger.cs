using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using NNBasicsUtilities.Core.Layers;

namespace NNBasicsUtilities.Core.Utilities.UtilityTypes
{
	public class Logger
	{
		public static Logger Instance => _instance ??= new Logger();

		private Logger()
		{
		}

		private string PredictionLayerRangeTag(PredictLayer layer) =>
			$"[Prediction layer initial weight range]<{layer.Weights.Min(list => list.Min())}; {layer.Weights.Max(list => list.Max())}>\n";

		private string AccuracyTag(double accuracy, double seriesCount) =>
			$"Correct: {accuracy} of {seriesCount} ({(accuracy / (double) seriesCount) * 100}%)\n";

		private string AlphaTag(double alpha) => $"[Alpha]\n{alpha}\n";
		private string CumulativeErrorTag(double error) => $"[Cumulative error]\n{error}\n";
		private string ErrorsTag(Matrix errors) => $"[Error of each neuron]\n{errors}";
		private string IterationTag(int index) => $"[After Iteration No. {index}]\n";
		private string SeriesTag(int index) => $"[Series No. {index}]\n";
		private string ExpectedTag(Matrix expected) => $"[Expected]\n{expected}";
		private string ResultTag(Matrix ans) => $"[Result]\n{ans}";
		private string HiddenLayersCountTag(int count) => $"[Count of hidden layers]\n{count}\n";
		private string HiddenLayerTag(HiddenLayer layer) => $"[Hidden layer weights]\n{layer}";
		private string PredictLayerTag(PredictLayer layer) => $"[Prediction layer weights]\n{layer}";

		private string Start => _isLearning
			? $"[Start of training the \"{_name}\" network]\n\n"
			: $"[Start of testing the \"{_name}\" network]\n\n";

		private string CurrentDate => $"\n[{DateTime.Now.ToShortDateString()} | {DateTime.Now.ToLongTimeString()}]\n";
		private string End => $"[End of logging for \"{_name}\" network]";
		private string TestResult => "[Test results]\n";
		private string Preconditions => "[Preconditions]\n";

		private bool _isSessionOpened;
		private bool _isLearning;
		private bool _verbose;
		private string _currentFile;
		private StringBuilder _logBuilder;

		private static Logger _instance;
		private string _name;

		public Logger StartSession(bool isLearningSession = false, string name = null, bool verbose = false)
		{
			if (_isSessionOpened)
			{
				throw new AccessViolationException(
					"Session of logging is currently created, dump data first before you start another one session!");
			}

			_verbose = verbose;
			_logBuilder = new StringBuilder();
			_isLearning = isLearningSession;
			_isSessionOpened = true;
			_name = name;
			_currentFile =
				$"log_{Guid.NewGuid().ToString().Substring(0, 5)}_{_name}_{DateTime.Now.ToShortDateString()}_{DateTime.Now.ToLongTimeString()}_{(isLearningSession ? "training" : "testing")}_session.log";
			_currentFile = _currentFile.Replace(':', '_');
			using var file = new FileStream(_currentFile, FileMode.Append, FileAccess.Write);
			using var stream = new MemoryStream();
			using var writer = new StreamWriter(stream);

			var date = CurrentDate;

			writer.Write(date);
			writer.Write(Start);
			writer.Flush();
			stream.Seek(0, SeekOrigin.Begin);
			stream.WriteTo(file);

			if (_verbose)
			{
				_logBuilder.Append(date).Append(Start);
			}

			return this;
		}

		public Logger LogLayerInfo(PredictLayer layer, List<HiddenLayer> hiddenLayers)
		{
			if (!_isSessionOpened)
			{
				throw new AccessViolationException(
					"Session of logging is currently closed, open session first!");
			}

			using var file = new FileStream(_currentFile, FileMode.Append, FileAccess.Write);
			using var stream = new MemoryStream();
			using var writer = new StreamWriter(stream);

			var date = CurrentDate;

			writer.Write(date);
			writer.Write(PredictLayerTag(layer));

			foreach (var hiddenLayer in hiddenLayers)
			{
				writer.Write(HiddenLayerTag(hiddenLayer));
			}

			writer.Flush();
			stream.Seek(0, SeekOrigin.Begin);
			stream.WriteTo(file);

			if (_verbose)
			{
				_logBuilder.Append(date)
				   .Append(PredictLayerTag(layer));
			}

			foreach (var hiddenLayer in hiddenLayers)
			{
				_logBuilder.Append(HiddenLayerTag(hiddenLayer));
			}

			return this;
		}

		public Logger LogTestFinalResults(PredictLayer layer, Matrix testErrors, double testError, int accuracy = 0,
			int seriesCount = 0)
		{
			if (!_isSessionOpened)
			{
				throw new AccessViolationException(
					"Session of logging is currently closed, open session first!");
			}

			using var file = new FileStream(_currentFile, FileMode.Append, FileAccess.Write);
			using var stream = new MemoryStream();
			using var writer = new StreamWriter(stream);

			var date = CurrentDate;

			writer.Write(date);
			writer.Write(TestResult);
			writer.Write(_verbose ? PredictLayerTag(layer) : PredictionLayerRangeTag(layer));
			writer.Write(ErrorsTag(testErrors));
			writer.Write(CumulativeErrorTag(testError));

			if (seriesCount > 0)
			{
				writer.Write(AccuracyTag(accuracy, seriesCount));
			}

			writer.Flush();
			stream.Seek(0, SeekOrigin.Begin);
			stream.WriteTo(file);

			if (_verbose)
			{
				_logBuilder.Append(date)
				   .Append(TestResult)
				   .Append(PredictLayerTag(layer))
				   .Append(ErrorsTag(testErrors))
				   .Append(CumulativeErrorTag(testError));

				if (seriesCount > 0)
				{
					_logBuilder.Append(AccuracyTag(accuracy, seriesCount));
				}
			}

			return this;
		}

		public Logger LogPreconditions(int hiddenCount, double alpha, PredictLayer predictionLayer)
		{
			if (!_isSessionOpened)
			{
				throw new AccessViolationException(
					"Session of logging is currently closed, open session first!");
			}

			using var file = new FileStream(_currentFile, FileMode.Append, FileAccess.Write);
			using var stream = new MemoryStream();
			using var writer = new StreamWriter(stream);

			var date = CurrentDate;

			writer.Write(date);
			writer.Write(Preconditions);
			writer.Write(_verbose ? PredictLayerTag(predictionLayer) : PredictionLayerRangeTag(predictionLayer));
			writer.Write(HiddenLayersCountTag(hiddenCount));
			writer.Write(AlphaTag(alpha));
			writer.Flush();
			stream.Seek(0, SeekOrigin.Begin);
			stream.WriteTo(file);

			if (_verbose)
			{
				_logBuilder.Append(date)
				   .Append(Preconditions)
				   .Append(PredictLayerTag(predictionLayer))
				   .Append(HiddenLayersCountTag(hiddenCount))
				   .Append(AlphaTag(alpha));
			}

			return this;
		}

		public Logger LogIteration(int iterationIdx, PredictLayer layer, Matrix iterationErrors,
			double iterationError, int accuracy = 0, int seriesCount = 0)
		{
			if (!_isSessionOpened)
			{
				throw new AccessViolationException(
					"Session of logging is currently closed, open session first!");
			}

			using var file = new FileStream(_currentFile, FileMode.Append, FileAccess.Write);
			using var stream = new MemoryStream();
			using var writer = new StreamWriter(stream);

			var date = CurrentDate;

			writer.Write(date);
			writer.Write(IterationTag(iterationIdx));
			writer.Write(_verbose ? PredictLayerTag(layer) : PredictionLayerRangeTag(layer));
			writer.Write(ErrorsTag(iterationErrors));
			writer.Write(CumulativeErrorTag(iterationError));

			if (seriesCount > 0)
			{
				writer.Write(AccuracyTag(accuracy, seriesCount));
			}

			writer.Flush();
			stream.Seek(0, SeekOrigin.Begin);
			stream.WriteTo(file);

			if (_verbose)
			{
				_logBuilder.Append(date)
				   .Append(IterationTag(iterationIdx))
				   .Append(PredictLayerTag(layer))
				   .Append(ErrorsTag(iterationErrors))
				   .Append(CumulativeErrorTag(iterationError));

				if (seriesCount > 0)
				{
					_logBuilder.Append(AccuracyTag(accuracy, seriesCount));
				}
			}

			return this;
		}

		public Logger LogSeriesError(Matrix seriesErrors, Matrix ans, double seriesError, int seriesIdx,
			Matrix expected)
		{
			if (!_isSessionOpened)
			{
				throw new AccessViolationException(
					"Session of logging is currently closed, open session first!");
			}

			if (!_verbose) return this;
			using var file = new FileStream(_currentFile, FileMode.Append, FileAccess.Write);
			using var stream = new MemoryStream();
			using var writer = new StreamWriter(stream);

			var date = CurrentDate;

			writer.Write(date);
			writer.Write(SeriesTag(seriesIdx));
			writer.Write(ResultTag(ans));
			writer.Write(ExpectedTag(expected));
			writer.Write(ErrorsTag(seriesErrors));
			writer.Write(CumulativeErrorTag(seriesError));
			writer.Flush();
			stream.Seek(0, SeekOrigin.Begin);
			stream.WriteTo(file);

			_logBuilder.Append(date)
			   .Append(SeriesTag(seriesIdx))
			   .Append(ResultTag(ans))
			   .Append(ExpectedTag(expected))
			   .Append(ErrorsTag(seriesErrors))
			   .Append(CumulativeErrorTag(seriesError));

			return this;
		}

		public override string ToString()
		{
			_logBuilder.Append(CurrentDate)
			   .Append(End);
			return _logBuilder.ToString();
		}

		public void EndSession()
		{
			if (!_isSessionOpened)
			{
				throw new AccessViolationException(
					"Session of logging is currently closed, open session first!");
			}

			_isSessionOpened = false;
			_isLearning = false;

			using var file = new FileStream(_currentFile, FileMode.Append, FileAccess.Write);
			using var stream = new MemoryStream();
			using var writer = new StreamWriter(stream);

			writer.Write(CurrentDate);
			writer.Write(End);
			writer.Flush();
			stream.Seek(0, SeekOrigin.Begin);
			stream.WriteTo(file);

			_currentFile = "";
		}
	}
}