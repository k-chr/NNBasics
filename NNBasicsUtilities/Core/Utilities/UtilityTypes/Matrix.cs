using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.Core.Utilities.UtilityTypes
{
   public class Matrix :  IDisposable, IEnumerable<double[][]>
   { 
      private int _rows;
      private readonly int _cols;

      private double[][] _data;


      public IEnumerator<double[][]> GetEnumerator()
      {
	      return (IEnumerator<double[][]>)_data.GetEnumerator();
      }

      public double[] this[int x]
      {
	      get => _data[x];
	      set => _data[x] = value;
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

         _cols = cols;
         _rows = values.Count;

         
      }

      public Matrix(Tuple<int, int> size = null)
      {
         var (cols, rows) = size?? Tuple.Create(1,1);
         _rows = rows;
         _cols = cols;

         for (var i = 0; i < _rows; ++i)
         {
            var l = new List<double>();
            for (var j = 0; j < _cols; ++j)
            {
               l.Add(0);
            }
            Add(l);
         }

      }

      public void Add(List<double> row)
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
            .GroupBy(i => i.index, i => i.item).ToMatrix();
         return mat;
      }

      public void AddMatrix(Matrix other)
      {
         if (_cols != other._cols || _rows != other._rows)
         {
            throw new ArgumentException("Addition cannot be performed, provided matrices don't match the rule of size matching");
         }

         var i = 0;
         foreach (var row in other)
         {
            var j = 0;

            foreach (var d in row)
            {
               this[i][j++] += d;
            }

            ++i;
         }
      }

      public static Matrix operator + (Matrix first, Matrix other)
      {
         if (first._cols != other._cols || first._rows != other._rows)
         {
            throw new ArgumentException("Addition cannot be performed, provided matrices don't match the rule of size matching");
         }

         return first.Select((row, rowId) => row.Zip(other[rowId], (d, d1) => d + d1)).ToMatrix();
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
                                 )
            ).ToMatrix();

         return mat;
      }

      public static Matrix operator * (Matrix first, double alpha)
      {
        var mat = first.Select(
            (row, rowId) => row.Select(elem => elem * alpha
         )).ToMatrix();

        return mat;
      }

      public Matrix HadamardProduct(Matrix other)
      {
         return this.Zip(other, (row1, row2) => row1.Zip(row2, (d1, d2) => d1 * d2)).ToMatrix();
      }

      //public void HadamardProduct(Matrix other)
      //{

      //}

      public void Dispose()
      {
         GC.SuppressFinalize(this);
      }
   }
}
