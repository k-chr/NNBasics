using System;
using System.Collections.Generic;
using System.Linq;

namespace NNBasics.NNBasicsLimak.Core.UtilityTypes
{
   public class Matrix : List<List<double>>
   {
      private int _rows;
      private readonly int _cols;

      public Matrix(List<List<double>> values)
      {
         var cols = values[0].Count;
         if (values.Any(row => row.Count != cols))
         {
            throw new ArgumentException("Provided list of list of doubles isn't a good candidate to converse it into matrix");
         }
         
         AddRange(values);
      }

      public Matrix(Tuple<int, int> size = null)
      {
         var (rows, cols) = size?? Tuple.Create(1,1);
         _rows = rows;
         _cols = cols;
      }

      public new void Add(List<double> row)
      {
         if (row.Count != _cols)
         {
            throw new ArgumentException($"Provided row can't match its size with number of columns of matrix, expected size: {_cols}, got: {row.Count}");
         }

         base.Add(row);
         if(Count > _rows)
            ++_rows;
      }

      public Matrix Transpose()
      {
         var mat = this.SelectMany(inner => inner.Select((item, index) => new { item, index }))
            .GroupBy(i => i.index, i => i.item)
            .Select(g => g.ToList())
            .ToList();
         return new Matrix(mat);
      }

      public static Matrix operator * (Matrix first, Matrix other)
      {
         if (first._cols != other._rows)
         {
            throw new ArgumentException("Multiplication cannot be performed, provided matrices don't match the rule: A.cols == B.rows, where A, B are matrices, cols is count of columns and rows is count of rows");
         }

         var mat = first.Select(
            (row, rowId) => other.Transpose()
                                 .Select((col, colId) => col.Zip(row, (colCell, rowCell) => colCell*rowCell).Sum()
                                 ).ToList()
            ).ToList();

         var matrix = new Matrix(mat);

         return matrix;
      }

      public static Matrix operator * (Matrix first, double alpha)
      {
        var mat = first.Select(
            (row, rowId) => row.Select(elem => elem * alpha).ToList()
         ).ToList();

         var matrix = new Matrix(mat);

         return matrix;
      }

   }
}
