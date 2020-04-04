using System;
using System.IO;
using System.Text;
using NNBasics.NNBasicsLimak.Core.Layers;

namespace NNBasics.NNBasicsLimak.Core.UtilityTypes
{
   public class Logger
   {
      public static Logger Instance => _instance ??= new Logger();

      private Logger()
      {
      }

      private string AlphaTag(double alpha) => $"[Alpha]\n{alpha}\n";
      private string CumulativeErrorTag(double error) => $"[Cumulative error]\n{error}\n";
      private string ErrorsTag(Matrix errors) => $"[Error of each neuron]\n{errors}\n";
      private string IterationTag(int index) => $"[After Iteration No. {index}]\n";
      private string SeriesTag(int index) => $"[Series No. {index}]\n";
      private string ExpectedTag(Matrix expected) => $"[Expected]\n{expected}\n";
      private string ResultTag(Matrix ans) => $"[Result]\n{ans}\n";
      private string HiddenLayersCountTag(int count) => $"[Count of hidden layers]\n{count}\n";
      private string PredictLayerTag(PredictLayer layer) => $"[Prediction layer weights]\n{layer}";
      private string Start => _isLearning ? "[Start of training]\n" : "[Start of testing]\n";
      private string CurrentDate => $"[{DateTime.Now.ToShortDateString()} | {DateTime.Now.ToLongTimeString()}]\n";
      private string End => "[End of logging]\n";

      private bool _isSessionOpened;
      private bool _isLearning;
      private string _currentFile;
      private StringBuilder _logBuilder;

      private static Logger _instance;

      public Logger StartSession(bool isLearningSession = false)
      {
         if (_isSessionOpened)
         {
            throw new AccessViolationException(
               "Session of logging is currently created, dump data first before you start another one session!");
         }

         _logBuilder = new StringBuilder();
         _isLearning = isLearningSession;
         _isSessionOpened = true;
         _currentFile =
            $"log_{Guid.NewGuid().ToString().Substring(0, 5)}_{DateTime.Now.ToShortDateString()}_{DateTime.Now.ToLongTimeString()}_{(isLearningSession ? "training" : "testing")}_session.log";

         using var file = new FileStream(_currentFile, FileMode.Append, FileAccess.Write);
         using var stream = new MemoryStream();
         using var writer = new StreamWriter(stream);

         var date = CurrentDate;

         writer.Write(date);
         writer.Write(Start);
         stream.WriteTo(file);

         _logBuilder.Append(date).Append(Start);

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
         writer.Write(PredictLayerTag(predictionLayer));
         writer.Write(HiddenLayersCountTag(hiddenCount));
         writer.Write(AlphaTag(alpha));
         stream.WriteTo(file);

         _logBuilder.Append(date)
            .Append(PredictLayerTag(predictionLayer))
            .Append(HiddenLayersCountTag(hiddenCount))
            .Append(AlphaTag(alpha));

         return this;
      }

      public Logger LogIteration(int iterationIdx, PredictLayer layer, Matrix iterationErrors,
         double iterationError)
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
         writer.Write(PredictLayerTag(layer));
         writer.Write(ErrorsTag(iterationErrors));
         writer.Write(CumulativeErrorTag(iterationError));
         stream.WriteTo(file);

         _logBuilder.Append(date)
            .Append(IterationTag(iterationIdx))
            .Append(PredictLayerTag(layer))
            .Append(ErrorsTag(iterationErrors))
            .Append(CumulativeErrorTag(iterationError));

         return this;
      }

      public Logger LogSeriesError(Matrix seriesErrors, Matrix ans, double seriesError, int seriesIdx, Matrix expected)
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
         writer.Write(SeriesTag(seriesIdx));
         writer.Write(ResultTag(ans));
         writer.Write(ExpectedTag(expected));
         writer.Write(ErrorsTag(seriesErrors));
         writer.Write(CumulativeErrorTag(seriesError));
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
         _logBuilder.Append(CurrentDate).Append(End);
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
         stream.WriteTo(file);
         _currentFile = "";
      }
   }
}