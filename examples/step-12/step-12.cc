
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include "./../../../tests/simplex/simplex_grids.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>


#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/lapack_templates.h>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif


using namespace dealii;

template <bool transpose_matrix, bool add, typename Number, typename Number2>
void apply_matrix_vector_product(const Number2 *matrix,
                                 const Number  *in0,
                                 Number        *out0,
                                 const int      n_rows,
                                 const int      n_columns)
{
  const int mm = transpose_matrix ? n_rows : n_columns,
            nn = transpose_matrix ? n_columns : n_rows;
  Assert(n_rows > 0 && n_columns > 0,
         ExcInternalError("Empty evaluation task!"));
  Assert(n_rows > 0 && n_columns > 0,
         ExcInternalError("The evaluation needs n_rows, n_columns > 0, but " +
                          std::to_string(n_rows) + ", " +
                          std::to_string(n_columns) + " was passed!"));

  const Number *in1 = in0 + mm, *in2 = in1 + mm, *in3 = in2 + mm;
  Number       *out1 = out0 + nn, *out2 = out1 + nn, *out3 = out2 + nn;

  int nn_regular = (nn / 4) * 4;
  for (int col = 0; col < nn_regular; col += 4)
    {
      ndarray<Number, 4, 4> res;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + col;
          const Number   a = in0[0], b = in1[0], c = in2[0], d = in3[0];
          for (unsigned int k = 0; k < 4; ++k)
            {
              const Number m = matrix_ptr[k];
              res[0][k]      = m * a;
              res[1][k]      = m * b;
              res[2][k]      = m * c;
              res[3][k]      = m * d;
            }
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              const Number a = in0[i], b = in1[i], c = in2[i], d = in3[i];
              for (unsigned int k = 0; k < 4; ++k)
                {
                  const Number m = matrix_ptr[k];
                  res[0][k] += m * a;
                  res[1][k] += m * b;
                  res[2][k] += m * c;
                  res[3][k] += m * d;
                }
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + col * n_columns;
          const Number2 *matrix_1 = matrix + (col + 1) * n_columns;
          const Number2 *matrix_2 = matrix + (col + 2) * n_columns;
          const Number2 *matrix_3 = matrix + (col + 3) * n_columns;

          const Number a = in0[0], b = in1[0], c = in2[0], d = in3[0];
          Number       m = matrix_0[0];
          res[0][0]      = m * a;
          res[1][0]      = m * b;
          res[2][0]      = m * c;
          res[3][0]      = m * d;
          m              = matrix_1[0];
          res[0][1]      = m * a;
          res[1][1]      = m * b;
          res[2][1]      = m * c;
          res[3][1]      = m * d;
          m              = matrix_2[0];
          res[0][2]      = m * a;
          res[1][2]      = m * b;
          res[2][2]      = m * c;
          res[3][2]      = m * d;
          m              = matrix_3[0];
          res[0][3]      = m * a;
          res[1][3]      = m * b;
          res[2][3]      = m * c;
          res[3][3]      = m * d;
          for (int i = 1; i < mm; ++i)
            {
              const Number a = in0[i], b = in1[i], c = in2[i], d = in3[i];
              m = matrix_0[i];
              res[0][0] += m * a;
              res[1][0] += m * b;
              res[2][0] += m * c;
              res[3][0] += m * d;
              m = matrix_1[i];
              res[0][1] += m * a;
              res[1][1] += m * b;
              res[2][1] += m * c;
              res[3][1] += m * d;
              m = matrix_2[i];
              res[0][2] += m * a;
              res[1][2] += m * b;
              res[2][2] += m * c;
              res[3][2] += m * d;
              m = matrix_3[i];
              res[0][3] += m * a;
              res[1][3] += m * b;
              res[2][3] += m * c;
              res[3][3] += m * d;
            }
        }
      for (unsigned int i = 0; i < 4; ++i)
        {
          if (add)
            {
              out0[i] += res[0][i];
              out1[i] += res[1][i];
              out2[i] += res[2][i];
              out3[i] += res[3][i];
            }
          else
            {
              out0[i] = res[0][i];
              out1[i] = res[1][i];
              out2[i] = res[2][i];
              out3[i] = res[3][i];
            }
        }
      out0 += 4;
      out1 += 4;
      out2 += 4;
      out3 += 4;
    }
  if (nn - nn_regular == 3)
    {
      Number res0, res1, res2, res3, res4, res5, res6, res7, res8, res9, res10,
        res11;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[1] * in0[0];
          res2                      = matrix_ptr[2] * in0[0];
          res3                      = matrix_ptr[0] * in1[0];
          res4                      = matrix_ptr[1] * in1[0];
          res5                      = matrix_ptr[2] * in1[0];
          res6                      = matrix_ptr[0] * in2[0];
          res7                      = matrix_ptr[1] * in2[0];
          res8                      = matrix_ptr[2] * in2[0];
          res9                      = matrix_ptr[0] * in3[0];
          res10                     = matrix_ptr[1] * in3[0];
          res11                     = matrix_ptr[2] * in3[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[1] * in0[i];
              res2 += matrix_ptr[2] * in0[i];
              res3 += matrix_ptr[0] * in1[i];
              res4 += matrix_ptr[1] * in1[i];
              res5 += matrix_ptr[2] * in1[i];
              res6 += matrix_ptr[0] * in2[i];
              res7 += matrix_ptr[1] * in2[i];
              res8 += matrix_ptr[2] * in2[i];
              res9 += matrix_ptr[0] * in3[i];
              res10 += matrix_ptr[1] * in3[i];
              res11 += matrix_ptr[2] * in3[i];
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + nn_regular * n_columns;
          const Number2 *matrix_1 = matrix + (nn_regular + 1) * n_columns;
          const Number2 *matrix_2 = matrix + (nn_regular + 2) * n_columns;

          res0  = matrix_0[0] * in0[0];
          res1  = matrix_1[0] * in0[0];
          res2  = matrix_2[0] * in0[0];
          res3  = matrix_0[0] * in1[0];
          res4  = matrix_1[0] * in1[0];
          res5  = matrix_2[0] * in1[0];
          res6  = matrix_0[0] * in2[0];
          res7  = matrix_1[0] * in2[0];
          res8  = matrix_2[0] * in2[0];
          res9  = matrix_0[0] * in3[0];
          res10 = matrix_1[0] * in3[0];
          res11 = matrix_2[0] * in3[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_0[i] * in0[i];
              res1 += matrix_1[i] * in0[i];
              res2 += matrix_2[i] * in0[i];
              res3 += matrix_0[i] * in1[i];
              res4 += matrix_1[i] * in1[i];
              res5 += matrix_2[i] * in1[i];
              res6 += matrix_0[i] * in2[i];
              res7 += matrix_1[i] * in2[i];
              res8 += matrix_2[i] * in2[i];
              res9 += matrix_0[i] * in3[i];
              res10 += matrix_1[i] * in3[i];
              res11 += matrix_2[i] * in3[i];
            }
        }
      if (add)
        {
          out0[0] += res0;
          out0[1] += res1;
          out0[2] += res2;
          out1[0] += res3;
          out1[1] += res4;
          out1[2] += res5;
          out2[0] += res6;
          out2[1] += res7;
          out2[2] += res8;
          out3[0] += res9;
          out3[1] += res10;
          out3[2] += res11;
        }
      else
        {
          out0[0] = res0;
          out0[1] = res1;
          out0[2] = res2;
          out1[0] = res3;
          out1[1] = res4;
          out1[2] = res5;
          out2[0] = res6;
          out2[1] = res7;
          out2[2] = res8;
          out3[0] = res9;
          out3[1] = res10;
          out3[2] = res11;
        }
    }
  else if (nn - nn_regular == 2)
    {
      Number res0, res1, res2, res3, res4, res5, res6, res7;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[1] * in0[0];
          res2                      = matrix_ptr[0] * in1[0];
          res3                      = matrix_ptr[1] * in1[0];
          res4                      = matrix_ptr[0] * in2[0];
          res5                      = matrix_ptr[1] * in2[0];
          res6                      = matrix_ptr[0] * in3[0];
          res7                      = matrix_ptr[1] * in3[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[1] * in0[i];
              res2 += matrix_ptr[0] * in1[i];
              res3 += matrix_ptr[1] * in1[i];
              res4 += matrix_ptr[0] * in2[i];
              res5 += matrix_ptr[1] * in2[i];
              res6 += matrix_ptr[0] * in3[i];
              res7 += matrix_ptr[1] * in3[i];
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + nn_regular * n_columns;
          const Number2 *matrix_1 = matrix + (nn_regular + 1) * n_columns;

          res0 = matrix_0[0] * in0[0];
          res1 = matrix_1[0] * in0[0];
          res2 = matrix_0[0] * in1[0];
          res3 = matrix_1[0] * in1[0];
          res4 = matrix_0[0] * in2[0];
          res5 = matrix_1[0] * in2[0];
          res6 = matrix_0[0] * in3[0];
          res7 = matrix_1[0] * in3[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_0[i] * in0[i];
              res1 += matrix_1[i] * in0[i];
              res2 += matrix_0[i] * in1[i];
              res3 += matrix_1[i] * in1[i];
              res4 += matrix_0[i] * in2[i];
              res5 += matrix_1[i] * in2[i];
              res6 += matrix_0[i] * in3[i];
              res7 += matrix_1[i] * in3[i];
            }
        }
      if (add)
        {
          out0[0] += res0;
          out0[1] += res1;
          out1[0] += res2;
          out1[1] += res3;
          out2[0] += res4;
          out2[1] += res5;
          out3[0] += res6;
          out3[1] += res7;
        }
      else
        {
          out0[0] = res0;
          out0[1] = res1;
          out1[0] = res2;
          out1[1] = res3;
          out2[0] = res4;
          out2[1] = res5;
          out3[0] = res6;
          out3[1] = res7;
        }
    }
  else if (nn - nn_regular == 1)
    {
      Number res0, res1, res2, res3;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[0] * in1[0];
          res2                      = matrix_ptr[0] * in2[0];
          res3                      = matrix_ptr[0] * in3[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[0] * in1[i];
              res2 += matrix_ptr[0] * in2[i];
              res3 += matrix_ptr[0] * in3[i];
            }
        }
      else
        {
          const Number2 *matrix_ptr = matrix + nn_regular * n_columns;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[0] * in1[0];
          res2                      = matrix_ptr[0] * in2[0];
          res3                      = matrix_ptr[0] * in3[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_ptr[i] * in0[i];
              res1 += matrix_ptr[i] * in1[i];
              res2 += matrix_ptr[i] * in2[i];
              res3 += matrix_ptr[i] * in3[i];
            }
        }
      if (add)
        {
          out0[0] += res0;
          out1[0] += res1;
          out2[0] += res2;
          out3[0] += res3;
        }
      else
        {
          out0[0] = res0;
          out1[0] = res1;
          out2[0] = res2;
          out3[0] = res3;
        }
    }
}



template <bool transpose_matrix,
          bool add,
          int  n_rows,
          int  n_columns,
          typename Number,
          typename Number2>
void apply_matrix_vector_product_templated(const Number2 *matrix,
                                           const Number  *in0,
                                           Number        *out0)
{
  constexpr int mm = transpose_matrix ? n_rows : n_columns,
                nn = transpose_matrix ? n_columns : n_rows;
  Assert(n_rows > 0 && n_columns > 0,
         ExcInternalError("Empty evaluation task!"));
  Assert(n_rows > 0 && n_columns > 0,
         ExcInternalError("The evaluation needs n_rows, n_columns > 0, but " +
                          std::to_string(n_rows) + ", " +
                          std::to_string(n_columns) + " was passed!"));

  const Number *in1 = in0 + mm, *in2 = in1 + mm, *in3 = in2 + mm;
  Number       *out1 = out0 + nn, *out2 = out1 + nn, *out3 = out2 + nn;

  constexpr int nn_regular = (nn / 4) * 4;
  for (int col = 0; col < nn_regular; col += 4)
    {
      ndarray<Number, 4, 4> res;
      if constexpr (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + col;
          const Number   a = in0[0], b = in1[0], c = in2[0], d = in3[0];
          for (unsigned int k = 0; k < 4; ++k)
            {
              const Number m = matrix_ptr[k];
              res[0][k]      = m * a;
              res[1][k]      = m * b;
              res[2][k]      = m * c;
              res[3][k]      = m * d;
            }
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              const Number a = in0[i], b = in1[i], c = in2[i], d = in3[i];
              for (unsigned int k = 0; k < 4; ++k)
                {
                  const Number m = matrix_ptr[k];
                  res[0][k] += m * a;
                  res[1][k] += m * b;
                  res[2][k] += m * c;
                  res[3][k] += m * d;
                }
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + col * n_columns;
          const Number2 *matrix_1 = matrix + (col + 1) * n_columns;
          const Number2 *matrix_2 = matrix + (col + 2) * n_columns;
          const Number2 *matrix_3 = matrix + (col + 3) * n_columns;

          const Number a = in0[0], b = in1[0], c = in2[0], d = in3[0];
          Number       m = matrix_0[0];
          res[0][0]      = m * a;
          res[1][0]      = m * b;
          res[2][0]      = m * c;
          res[3][0]      = m * d;
          m              = matrix_1[0];
          res[0][1]      = m * a;
          res[1][1]      = m * b;
          res[2][1]      = m * c;
          res[3][1]      = m * d;
          m              = matrix_2[0];
          res[0][2]      = m * a;
          res[1][2]      = m * b;
          res[2][2]      = m * c;
          res[3][2]      = m * d;
          m              = matrix_3[0];
          res[0][3]      = m * a;
          res[1][3]      = m * b;
          res[2][3]      = m * c;
          res[3][3]      = m * d;
          for (int i = 1; i < mm; ++i)
            {
              const Number a = in0[i], b = in1[i], c = in2[i], d = in3[i];
              m = matrix_0[i];
              res[0][0] += m * a;
              res[1][0] += m * b;
              res[2][0] += m * c;
              res[3][0] += m * d;
              m = matrix_1[i];
              res[0][1] += m * a;
              res[1][1] += m * b;
              res[2][1] += m * c;
              res[3][1] += m * d;
              m = matrix_2[i];
              res[0][2] += m * a;
              res[1][2] += m * b;
              res[2][2] += m * c;
              res[3][2] += m * d;
              m = matrix_3[i];
              res[0][3] += m * a;
              res[1][3] += m * b;
              res[2][3] += m * c;
              res[3][3] += m * d;
            }
        }
      for (unsigned int i = 0; i < 4; ++i)
        {
          if constexpr (add)
            {
              out0[i] += res[0][i];
              out1[i] += res[1][i];
              out2[i] += res[2][i];
              out3[i] += res[3][i];
            }
          else
            {
              out0[i] = res[0][i];
              out1[i] = res[1][i];
              out2[i] = res[2][i];
              out3[i] = res[3][i];
            }
        }
      out0 += 4;
      out1 += 4;
      out2 += 4;
      out3 += 4;
    }
  if constexpr (nn - nn_regular == 3)
    {
      Number res0, res1, res2, res3, res4, res5, res6, res7, res8, res9, res10,
        res11;
      if constexpr (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[1] * in0[0];
          res2                      = matrix_ptr[2] * in0[0];
          res3                      = matrix_ptr[0] * in1[0];
          res4                      = matrix_ptr[1] * in1[0];
          res5                      = matrix_ptr[2] * in1[0];
          res6                      = matrix_ptr[0] * in2[0];
          res7                      = matrix_ptr[1] * in2[0];
          res8                      = matrix_ptr[2] * in2[0];
          res9                      = matrix_ptr[0] * in3[0];
          res10                     = matrix_ptr[1] * in3[0];
          res11                     = matrix_ptr[2] * in3[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[1] * in0[i];
              res2 += matrix_ptr[2] * in0[i];
              res3 += matrix_ptr[0] * in1[i];
              res4 += matrix_ptr[1] * in1[i];
              res5 += matrix_ptr[2] * in1[i];
              res6 += matrix_ptr[0] * in2[i];
              res7 += matrix_ptr[1] * in2[i];
              res8 += matrix_ptr[2] * in2[i];
              res9 += matrix_ptr[0] * in3[i];
              res10 += matrix_ptr[1] * in3[i];
              res11 += matrix_ptr[2] * in3[i];
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + nn_regular * n_columns;
          const Number2 *matrix_1 = matrix + (nn_regular + 1) * n_columns;
          const Number2 *matrix_2 = matrix + (nn_regular + 2) * n_columns;

          res0  = matrix_0[0] * in0[0];
          res1  = matrix_1[0] * in0[0];
          res2  = matrix_2[0] * in0[0];
          res3  = matrix_0[0] * in1[0];
          res4  = matrix_1[0] * in1[0];
          res5  = matrix_2[0] * in1[0];
          res6  = matrix_0[0] * in2[0];
          res7  = matrix_1[0] * in2[0];
          res8  = matrix_2[0] * in2[0];
          res9  = matrix_0[0] * in3[0];
          res10 = matrix_1[0] * in3[0];
          res11 = matrix_2[0] * in3[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_0[i] * in0[i];
              res1 += matrix_1[i] * in0[i];
              res2 += matrix_2[i] * in0[i];
              res3 += matrix_0[i] * in1[i];
              res4 += matrix_1[i] * in1[i];
              res5 += matrix_2[i] * in1[i];
              res6 += matrix_0[i] * in2[i];
              res7 += matrix_1[i] * in2[i];
              res8 += matrix_2[i] * in2[i];
              res9 += matrix_0[i] * in3[i];
              res10 += matrix_1[i] * in3[i];
              res11 += matrix_2[i] * in3[i];
            }
        }
      if constexpr (add)
        {
          out0[0] += res0;
          out0[1] += res1;
          out0[2] += res2;
          out1[0] += res3;
          out1[1] += res4;
          out1[2] += res5;
          out2[0] += res6;
          out2[1] += res7;
          out2[2] += res8;
          out3[0] += res9;
          out3[1] += res10;
          out3[2] += res11;
        }
      else
        {
          out0[0] = res0;
          out0[1] = res1;
          out0[2] = res2;
          out1[0] = res3;
          out1[1] = res4;
          out1[2] = res5;
          out2[0] = res6;
          out2[1] = res7;
          out2[2] = res8;
          out3[0] = res9;
          out3[1] = res10;
          out3[2] = res11;
        }
    }
  else if constexpr (nn - nn_regular == 2)
    {
      Number res0, res1, res2, res3, res4, res5, res6, res7;
      if constexpr (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[1] * in0[0];
          res2                      = matrix_ptr[0] * in1[0];
          res3                      = matrix_ptr[1] * in1[0];
          res4                      = matrix_ptr[0] * in2[0];
          res5                      = matrix_ptr[1] * in2[0];
          res6                      = matrix_ptr[0] * in3[0];
          res7                      = matrix_ptr[1] * in3[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[1] * in0[i];
              res2 += matrix_ptr[0] * in1[i];
              res3 += matrix_ptr[1] * in1[i];
              res4 += matrix_ptr[0] * in2[i];
              res5 += matrix_ptr[1] * in2[i];
              res6 += matrix_ptr[0] * in3[i];
              res7 += matrix_ptr[1] * in3[i];
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + nn_regular * n_columns;
          const Number2 *matrix_1 = matrix + (nn_regular + 1) * n_columns;

          res0 = matrix_0[0] * in0[0];
          res1 = matrix_1[0] * in0[0];
          res2 = matrix_0[0] * in1[0];
          res3 = matrix_1[0] * in1[0];
          res4 = matrix_0[0] * in2[0];
          res5 = matrix_1[0] * in2[0];
          res6 = matrix_0[0] * in3[0];
          res7 = matrix_1[0] * in3[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_0[i] * in0[i];
              res1 += matrix_1[i] * in0[i];
              res2 += matrix_0[i] * in1[i];
              res3 += matrix_1[i] * in1[i];
              res4 += matrix_0[i] * in2[i];
              res5 += matrix_1[i] * in2[i];
              res6 += matrix_0[i] * in3[i];
              res7 += matrix_1[i] * in3[i];
            }
        }
      if constexpr (add)
        {
          out0[0] += res0;
          out0[1] += res1;
          out1[0] += res2;
          out1[1] += res3;
          out2[0] += res4;
          out2[1] += res5;
          out3[0] += res6;
          out3[1] += res7;
        }
      else
        {
          out0[0] = res0;
          out0[1] = res1;
          out1[0] = res2;
          out1[1] = res3;
          out2[0] = res4;
          out2[1] = res5;
          out3[0] = res6;
          out3[1] = res7;
        }
    }
  else if constexpr (nn - nn_regular == 1)
    {
      Number res0, res1, res2, res3;
      if constexpr (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[0] * in1[0];
          res2                      = matrix_ptr[0] * in2[0];
          res3                      = matrix_ptr[0] * in3[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[0] * in1[i];
              res2 += matrix_ptr[0] * in2[i];
              res3 += matrix_ptr[0] * in3[i];
            }
        }
      else
        {
          const Number2 *matrix_ptr = matrix + nn_regular * n_columns;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[0] * in1[0];
          res2                      = matrix_ptr[0] * in2[0];
          res3                      = matrix_ptr[0] * in3[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_ptr[i] * in0[i];
              res1 += matrix_ptr[i] * in1[i];
              res2 += matrix_ptr[i] * in2[i];
              res3 += matrix_ptr[i] * in3[i];
            }
        }
      if constexpr (add)
        {
          out0[0] += res0;
          out1[0] += res1;
          out2[0] += res2;
          out3[0] += res3;
        }
      else
        {
          out0[0] = res0;
          out1[0] = res1;
          out2[0] = res2;
          out3[0] = res3;
        }
    }
}



const double FREQUENCY = 3.0 * dealii::numbers::PI;
template <int dim>
class Solution : public dealii::Function<dim>
{
public:
  Solution(const unsigned int n_components = 1, const double time = 0.)
    : dealii::Function<dim>(n_components, time)
  {}

  double value(const dealii::Point<dim> &p,
               const unsigned int /*component*/) const final
  {
    double result = 1.0;
    for (unsigned int d = 0; d < dim; ++d)
      result *= std::sin(FREQUENCY * p[d]);

    return result;
  }
};

template <int dim>
class RightHandSide : public dealii::Function<dim>
{
public:
  RightHandSide(const unsigned int n_components = 1, const double time = 0.)
    : dealii::Function<dim>(n_components, time)
  {}

  double value(const dealii::Point<dim> &p,
               const unsigned int /* component */) const final
  {
    double result = FREQUENCY * FREQUENCY * dim;
    for (unsigned int d = 0; d < dim; ++d)
      result *= std::sin(FREQUENCY * p[d]);

    return result;
  }

  VectorizedArray<double>
  value_array(const Point<dim, VectorizedArray<double>> &p)
  {
    Point<dim>              point;
    VectorizedArray<double> results;

    for (unsigned int v = 0; v < results.size(); ++v)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            point[d] = p[d][v];
          }

        results[v] = value(point, 0);
      }

    return results;
  }
};

template <int fe_degree>
constexpr unsigned int compute_dofs_tet()
{
  return (fe_degree + 1) * (fe_degree + 2) * (fe_degree + 3) / 6;
}

template <int fe_degree>
constexpr unsigned int compute_n_q_tet()
{
  if constexpr (fe_degree == 1)
    return 6;
  if constexpr (fe_degree == 2)
    return 14;
  if constexpr (fe_degree == 3)
    return 35;
  return (fe_degree + 1) * (fe_degree + 1) * (fe_degree + 1);
}

template <int dim_,
          int fe_degree,
          int q_block_size,
          int batch_size_dgemm,
          int q_block_size_dgemm,
          int n_components = dim_,
          typename Number  = double>
class Operator : public Subscriptor
{
public:
  using value_type = Number;
  using number     = Number;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  static const int dim = dim_;

  using FECellIntegrator = FEEvaluation<dim, -1, 0, n_components, Number>;

  void reinit(const MappingFE<dim>            &mapping,
              const DoFHandler<dim>           &dof_handler,
              const QGaussSimplex<dim>        &quad,
              const AffineConstraints<number> &constraints,
              const unsigned int mg_level = numbers::invalid_unsigned_int,
              const bool         ones_on_diagonal = false)
  {
    this->constraints.copy_from(constraints);

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags = update_values | update_gradients |
                                update_JxW_values | update_quadrature_points;
    data.mg_level             = mg_level;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::cout << "Sizes shape info: "
                  << matrix_free.get_shape_info()
                       .data[0]
                       .shape_values.memory_consumption()
                  << " "
                  << matrix_free.get_shape_info()
                       .data[0]
                       .shape_gradients.memory_consumption()
                  << std::endl;
        std::cout << "DoFs per cell, n quadrature points: "
                  << dof_handler.get_fe().dofs_per_cell << " "
                  << matrix_free.get_shape_info().n_q_points << std::endl;
        std::cout << "dofs per cell and n dof indices: "
                  << matrix_free.get_shape_info().dofs_per_component_on_cell
                  << " " << matrix_free.get_dof_info(0).dof_indices.size()
                  << std::endl;
        std::cout << "n_active_cells, number of dofs "
                  << dof_handler.get_triangulation().n_active_cells() << " "
                  << dof_handler.n_dofs() << std::endl;
      }

    constrained_indices.clear();

    if (ones_on_diagonal)
      for (auto i : this->matrix_free.get_constrained_dofs())
        constrained_indices.push_back(i);


    constexpr unsigned int n_lanes = VectorizedArray<number>::size();
    manual_dof_indices.reinit(
      matrix_free.n_cell_batches(),
      matrix_free.get_dof_handler().get_fe().dofs_per_cell * n_lanes,
      true);
    manual_dof_indices.fill(numbers::invalid_unsigned_int);
    std::vector<types::global_dof_index> dof_indices(
      matrix_free.get_dof_handler().get_fe().dofs_per_cell);

    dof_indices_have_constraints.clear();
    dof_indices_have_constraints.resize(matrix_free.n_cell_batches());

    for (unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
      {
        bool has_constraints =
          matrix_free.n_active_entries_per_cell_batch(c) < n_lanes;
        for (unsigned int v = 0;
             v < matrix_free.n_active_entries_per_cell_batch(c);
             ++v)
          {
            matrix_free.get_cell_iterator(c, v)->get_dof_indices(dof_indices);
            for (unsigned int i = 0; i < dof_indices.size(); ++i)
              if (!constraints.is_constrained(dof_indices[i]))
                manual_dof_indices(c, i * n_lanes + v) =
                  matrix_free.get_dof_info()
                    .vector_partitioner->global_to_local(dof_indices[i]);
              else
                has_constraints = true;
          }
        dof_indices_have_constraints[c] = has_constraints;
      }


    const unsigned int n_q        = quad.size();
    const unsigned int n_dofs     = dof_handler.get_fe().dofs_per_cell;
    const unsigned int array_size = n_dofs * n_q * dim;

    shape_gradients_transpose.resize_fast(array_size);

    const auto &shape_gradients =
      matrix_free.get_shape_info().data[0].shape_gradients.data();

    for (unsigned int i = 0; i < n_dofs; ++i)
      for (unsigned int q = 0; q < n_q; ++q)
        for (unsigned int d = 0; d < dim; ++d)
          shape_gradients_transpose[i + n_dofs * d + q * n_dofs * dim] =
            shape_gradients[i * n_q * dim + q * dim + d];


    constexpr unsigned int q_block_size_effective = q_block_size;

    unsigned int pack_total_size = 0;
    for (unsigned int q_begin = 0; q_begin < n_q;
         q_begin += q_block_size_effective)
      {
        const unsigned int n_q_block =
          std::min(static_cast<unsigned int>(q_block_size_effective),
                   n_q - q_begin);
        pack_total_size += n_dofs * n_q_block * dim;
      }

    shape_gradients_packed.resize_fast(pack_total_size);
    pack_total_size = 0;
    for (unsigned int q_begin = 0; q_begin < n_q;
         q_begin += q_block_size_effective)
      {
        const unsigned int n_q_block =
          std::min(static_cast<unsigned int>(q_block_size_effective),
                   n_q - q_begin);

        number *integration = shape_gradients_packed.begin() + pack_total_size;

        for (unsigned int q_local = 0; q_local < n_q_block; ++q_local)
          {
            const unsigned int q = q_begin + q_local;
            for (unsigned int d = 0; d < dim; ++d)
              {
                const unsigned int k = q_local * dim + d;
                for (unsigned int i = 0; i < n_dofs; ++i)
                  {
                    const number value =
                      shape_gradients[i * n_q * dim + q * dim + d];
                    integration[i * n_q_block * dim + k] = value;
                  }
              }
          }
        pack_total_size += n_dofs * n_q_block * dim;
      }
  }

  virtual types::global_dof_index m() const
  {
    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      return this->matrix_free.get_dof_handler().n_dofs(
        this->matrix_free.get_mg_level());
    else
      return this->matrix_free.get_dof_handler().n_dofs();
  }

  Number el(unsigned int, unsigned int) const
  {
    DEAL_II_NOT_IMPLEMENTED();
    return 0;
  }

  virtual void initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  virtual void vmult(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_range, this, dst, src, true);

    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
  }


  virtual void vmult_masked_gather(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_masked_gather, this, dst, src, true);

    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
  }


  virtual void vmult_dgemm(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_dgemm, this, dst, src, true);

    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
  }


  void Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  void rhs(VectorType &rhs) const
  {
    VectorType dummy;
    initialize_dof_vector(dummy);
    dummy = 0.0;

    this->matrix_free.cell_loop(
      &Operator::do_rhs_range, this, rhs, dummy, true);

    // for (unsigned int i = 0; i < constrained_indices.size(); ++i)
    //   rhs.local_element(constrained_indices[i]) = 0.0;
  }

  const MatrixFree<dim, number> &get_matrix_free() const
  {
    return matrix_free;
  }

private:
  void do_cell_integral_global(FECellIntegrator &integrator,
                               VectorType       &dst,
                               const VectorType &src) const
  {
    integrator.gather_evaluate(src, EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate_scatter(EvaluationFlags::gradients, dst);
  }

  void do_cell_integral_range(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator integrator(matrix_free, range);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);
        do_cell_integral_global(integrator, dst, src);
      }
  }


  void do_rhs_range(const MatrixFree<dim, number> &matrix_free,
                    VectorType                    &dst,
                    const VectorType &,
                    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator   integrator(matrix_free, range);
    RightHandSide<dim> rhs_function;

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);
        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          integrator.submit_value(
            rhs_function.value_array(integrator.quadrature_point(q)), q);

        integrator.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  void do_cell_integral_masked_gather_no_blocking(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    AlignedVector<VectorizedArray<number>> *scratch_data =
      matrix_free.acquire_scratch_data();
    const internal::MatrixFreeFunctions::ShapeInfo<number> &shape_info =
      matrix_free.get_shape_info();
    const unsigned int dofs_per_cell  = shape_info.dofs_per_component_on_cell;
    constexpr unsigned int batch_size = 4;
    const unsigned int     n_q_points = shape_info.n_q_points;
    constexpr unsigned int n_lanes    = VectorizedArray<number>::size();

    const auto   &mapping_data = matrix_free.get_mapping_info().cell_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();

    scratch_data->resize_fast(batch_size * (dim * n_q_points + dofs_per_cell));
    VectorizedArray<number> *values_dofs = scratch_data->begin();
    VectorizedArray<number> *gradients_quad =
      scratch_data->begin() + batch_size * dofs_per_cell;

    for (unsigned int cell = range.first; cell < range.second;
         cell += batch_size)
      {
        // read dof values
        const unsigned int my_batch_size =
          cell + batch_size <= range.second ? batch_size : range.second - cell;
        const unsigned int *dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const number *src_ptr = src.begin();
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  {
#if 1
                    values_dofs[batch * dofs_per_cell + i] = {};
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        values_dofs[batch * dofs_per_cell + i][v] =
                          src_ptr[dof_indices[v]];
#else
                    values_dofs[batch * dofs_per_cell + i].gather(src_ptr,
                                                                  dof_indices);
#endif
                  }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  values_dofs[batch * dofs_per_cell + i] = {};
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    values_dofs[batch * dofs_per_cell + i][v] =
                      src_ptr[dof_indices[v]];
                }
          }

        // interpolate
        apply_matrix_vector_product<true, false>(
          shape_info.data[0].shape_gradients.data(),
          values_dofs,
          gradients_quad,
          dofs_per_cell,
          n_q_points * dim);

        // quadrature point operation
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int offsets =
              mapping_data.data_index_offsets[cell + batch];
            const Tensor<2, dim, VectorizedArray<number>> *jac =
              mapping_data.jacobians[0].data() + offsets;
            const VectorizedArray<number> *j_value =
              &mapping_data.JxW_values[offsets];
            VectorizedArray<number> *grad_ptr =
              gradients_quad + batch * n_q_points * dim;
            if (matrix_free.get_mapping_info().cell_type[cell + batch] <=
                internal::MatrixFreeFunctions::affine)
              {
                // const SymmetricTensor<2, dim, VectorizedArray<number>>
                //  my_metric = j_value[0] * symmetrize(transpose(jac[0]) *
                //  jac[0]);
                SymmetricTensor<2, dim, VectorizedArray<number>> my_metric;
                for (unsigned int d = 0; d < dim; ++d)
                  for (unsigned int f = d; f < dim; ++f)
                    {
                      VectorizedArray<number> sum = jac[0][0][d] * jac[0][0][f];
                      for (unsigned int e = 1; e < dim; ++e)
                        sum += jac[0][e][d] * jac[0][e][f];
                      my_metric[d][f] = sum * j_value[0];
                    }

                for (unsigned int q = 0; q < n_q_points; ++q, grad_ptr += dim)
                  {
                    Tensor<1, dim, VectorizedArray<number>> grad;
                    for (unsigned int d = 0; d < dim; ++d)
                      grad[d] = grad_ptr[d];
                    Tensor<1, dim, VectorizedArray<number>> result =
                      my_metric * grad;
                    const number weight = quadrature_weights[q];
                    for (unsigned int d = 0; d < dim; ++d)
                      grad_ptr[d] = weight * result[d];
                  }
              }
            else
              {
                for (unsigned int q = 0; q < n_q_points; ++q, grad_ptr += dim)
                  {
                    Tensor<1, dim, VectorizedArray<number>> grad;
                    for (unsigned int d = 0; d < dim; ++d)
                      grad[d] = grad_ptr[d];
                    Tensor<1, dim, VectorizedArray<number>> result =
                      j_value[q] * (transpose(jac[q]) * (jac[q] * grad));
                    for (unsigned int d = 0; d < dim; ++d)
                      grad_ptr[d] = result[d];
                  }
              }
          }

        // integrate
        apply_matrix_vector_product<false, false>(
          shape_info.data[0].shape_gradients.data(),
          gradients_quad,
          values_dofs,
          dofs_per_cell,
          n_q_points * dim);

        // distribute local to global
        dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  {
#if 1 || DEAL_II_VECTORIZATION_WIDTH_IN_BITS < 512
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        dst.local_element(dof_indices[v]) +=
                          values_dofs[batch * dofs_per_cell + i][v];
#else
                    VectorizedArray<number> val;
                    val.gather(dst.begin(), dof_indices);
                    val += values_dofs[batch * dofs_per_cell + i];
                    val.scatter(dof_indices, dst.begin());
#endif
                  }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    dst.local_element(dof_indices[v]) +=
                      values_dofs[batch * dofs_per_cell + i][v];
                }
          }
      }

    matrix_free.release_scratch_data(scratch_data);
  }

  // with quadrature blocking and packed integration matrix
  void do_cell_integral_masked_gather_q_blocking_packed(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    AlignedVector<VectorizedArray<number>> *scratch_data =
      matrix_free.acquire_scratch_data();
    const internal::MatrixFreeFunctions::ShapeInfo<number> &shape_info =
      matrix_free.get_shape_info();
    const unsigned int dofs_per_cell  = shape_info.dofs_per_component_on_cell;
    constexpr unsigned int batch_size = 4;
    // constexpr unsigned int q_block_size = 32;
    const unsigned int     n_q_points = shape_info.n_q_points;
    constexpr unsigned int n_lanes    = VectorizedArray<number>::size();

    const auto   &mapping_data = matrix_free.get_mapping_info().cell_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();

    scratch_data->resize_fast(batch_size *
                              (dim * q_block_size + 2 * dofs_per_cell));
    VectorizedArray<number> *values_dofs_in = scratch_data->begin();
    VectorizedArray<number> *values_dofs_out =
      scratch_data->begin() + batch_size * dofs_per_cell;
    VectorizedArray<number> *gradients_quad =
      scratch_data->begin() + 2 * batch_size * dofs_per_cell;

    const number *src_ptr = src.begin();

    for (unsigned int cell = range.first; cell < range.second;
         cell += batch_size)
      {
        // read dof values
        const unsigned int my_batch_size =
          cell + batch_size <= range.second ? batch_size : range.second - cell;

        std::fill(values_dofs_out,
                  values_dofs_out + dofs_per_cell * batch_size,
                  VectorizedArray<number>(0));

        std::fill(values_dofs_in,
                  values_dofs_in + dofs_per_cell * batch_size,
                  VectorizedArray<number>(0)); // TODO:remove

        const unsigned int *dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  {
                    values_dofs_in[batch * dofs_per_cell + i] = {};
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        values_dofs_in[batch * dofs_per_cell + i][v] =
                          src_ptr[dof_indices[v]];
                  }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  values_dofs_in[batch * dofs_per_cell + i] = {};
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    values_dofs_in[batch * dofs_per_cell + i][v] =
                      src_ptr[dof_indices[v]];
                }
          }

        // block over quadrature size
        const number *shape_block = shape_gradients_transpose.data();
        // shape_info.data[0].shape_gradients.data();
        for (unsigned int q_begin = 0; q_begin < n_q_points;
             q_begin += q_block_size)
          {
            const unsigned int n_q_block =
              std::min(static_cast<unsigned int>(q_block_size),
                       n_q_points - q_begin);

            const unsigned int n_block_columns = n_q_block * dim;

            // interpolate
            apply_matrix_vector_product<false, false>(shape_block,
                                                      values_dofs_in,
                                                      gradients_quad,
                                                      n_block_columns,
                                                      dofs_per_cell);

            // quadrature point operation
            for (unsigned int batch = 0; batch < my_batch_size; ++batch)
              {
                const unsigned int offsets =
                  mapping_data.data_index_offsets[cell + batch];
                const Tensor<2, dim, VectorizedArray<number>> *jac =
                  mapping_data.jacobians[0].data() + offsets;
                const VectorizedArray<number> *j_value =
                  &mapping_data.JxW_values[offsets];
                VectorizedArray<number> *grad_ptr =
                  gradients_quad + batch * n_block_columns;
                if (matrix_free.get_mapping_info().cell_type[cell + batch] <=
                    internal::MatrixFreeFunctions::affine)
                  {
                    // const SymmetricTensor<2, dim, VectorizedArray<number>>
                    //  my_metric = j_value[0] * symmetrize(transpose(jac[0]) *
                    //  jac[0]);
                    SymmetricTensor<2, dim, VectorizedArray<number>> my_metric;
                    for (unsigned int d = 0; d < dim; ++d)
                      for (unsigned int f = d; f < dim; ++f)
                        {
                          VectorizedArray<number> sum =
                            jac[0][0][d] * jac[0][0][f];
                          for (unsigned int e = 1; e < dim; ++e)
                            sum += jac[0][e][d] * jac[0][e][f];
                          my_metric[d][f] = sum * j_value[0];
                        }

                    for (unsigned int q_local = 0; q_local < n_q_block;
                         ++q_local, grad_ptr += dim)
                      {
                        Tensor<1, dim, VectorizedArray<number>> grad;
                        for (unsigned int d = 0; d < dim; ++d)
                          grad[d] = grad_ptr[d];
                        Tensor<1, dim, VectorizedArray<number>> result =
                          my_metric * grad;
                        const number weight =
                          quadrature_weights[q_begin + q_local];
                        for (unsigned int d = 0; d < dim; ++d)
                          grad_ptr[d] = weight * result[d];
                      }
                  }
                else
                  {
                    DEAL_II_NOT_IMPLEMENTED();
                    for (unsigned int q = 0; q < n_q_points;
                         ++q, grad_ptr += dim)
                      {
                        Tensor<1, dim, VectorizedArray<number>> grad;
                        for (unsigned int d = 0; d < dim; ++d)
                          grad[d] = grad_ptr[d];
                        Tensor<1, dim, VectorizedArray<number>> result =
                          j_value[q] * (transpose(jac[q]) * (jac[q] * grad));
                        for (unsigned int d = 0; d < dim; ++d)
                          grad_ptr[d] = result[d];
                      }
                  }
              }

            // integrate
            apply_matrix_vector_product<false, true>(
              shape_gradients_packed.begin() + q_begin * dim * dofs_per_cell,
              gradients_quad,
              values_dofs_out,
              dofs_per_cell,
              n_block_columns);

            // advance shape block
            shape_block += dim * dofs_per_cell * n_q_block;
          }

        // distribute local to global
        dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  {
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        dst.local_element(dof_indices[v]) +=
                          values_dofs_out[batch * dofs_per_cell + i][v];
                  }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    dst.local_element(dof_indices[v]) +=
                      values_dofs_out[batch * dofs_per_cell + i][v];
                }
          }
      }

    matrix_free.release_scratch_data(scratch_data);
  }

  // with quadrature blocking and packed integration matrix,
  // templated matrix-vector kernels
  void do_cell_integral_masked_gather(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    AlignedVector<VectorizedArray<number>> *scratch_data =
      matrix_free.acquire_scratch_data();
    const internal::MatrixFreeFunctions::ShapeInfo<number> &shape_info =
      matrix_free.get_shape_info();
    // const unsigned int dofs_per_cell  =
    // shape_info.dofs_per_component_on_cell;
    constexpr unsigned int dofs_per_cell = compute_dofs_tet<fe_degree>();
    constexpr unsigned int q_block_size_effective = q_block_size;
    // compute_n_q_tet<fe_degree>(); // q_block_size;
    // Assert(q_block_size_effective == n_q_points, ExcInternalError());
    constexpr unsigned int batch_size = 4;
    // constexpr unsigned int q_block_size = 32;
    const unsigned int     n_q_points = shape_info.n_q_points;
    constexpr unsigned int n_lanes    = VectorizedArray<number>::size();

    const auto   &mapping_data = matrix_free.get_mapping_info().cell_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();

    scratch_data->resize_fast(
      batch_size * (dim * q_block_size_effective + 2 * dofs_per_cell));
    VectorizedArray<number> *values_dofs_in = scratch_data->begin();
    VectorizedArray<number> *values_dofs_out =
      scratch_data->begin() + batch_size * dofs_per_cell;
    VectorizedArray<number> *gradients_quad =
      scratch_data->begin() + 2 * batch_size * dofs_per_cell;

    const number *src_ptr = src.begin();

    for (unsigned int cell = range.first; cell < range.second;
         cell += batch_size)
      {
        // read dof values
        const unsigned int my_batch_size =
          cell + batch_size <= range.second ? batch_size : range.second - cell;

        std::fill(values_dofs_out,
                  values_dofs_out + dofs_per_cell * batch_size,
                  VectorizedArray<number>(0));

        const unsigned int *dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  {
                    values_dofs_in[batch * dofs_per_cell + i] = {};
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        values_dofs_in[batch * dofs_per_cell + i][v] =
                          src_ptr[dof_indices[v]];
                  }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  values_dofs_in[batch * dofs_per_cell + i] = {};
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    values_dofs_in[batch * dofs_per_cell + i][v] =
                      src_ptr[dof_indices[v]];
                }
          }

        // block over quadrature size
        const number *shape_block = shape_gradients_transpose.data();
        // shape_info.data[0].shape_gradients.data();
        for (unsigned int q_begin = 0; q_begin < n_q_points;
             q_begin += q_block_size_effective)
          {
            const unsigned int n_q_block =
              std::min(static_cast<unsigned int>(q_block_size_effective),
                       n_q_points - q_begin);

            const unsigned int n_block_columns = n_q_block * dim;

            // interpolate
            if (n_q_block == q_block_size_effective)
              apply_matrix_vector_product_templated<
                false,
                false,
                q_block_size_effective * dim,
                dofs_per_cell>(shape_block, values_dofs_in, gradients_quad);
            else
              apply_matrix_vector_product<false, false>(shape_block,
                                                        values_dofs_in,
                                                        gradients_quad,
                                                        n_block_columns,
                                                        dofs_per_cell);

            // quadrature point operation
            for (unsigned int batch = 0; batch < my_batch_size; ++batch)
              {
                const unsigned int offsets =
                  mapping_data.data_index_offsets[cell + batch];
                const Tensor<2, dim, VectorizedArray<number>> *jac =
                  mapping_data.jacobians[0].data() + offsets;
                const VectorizedArray<number> *j_value =
                  &mapping_data.JxW_values[offsets];
                VectorizedArray<number> *grad_ptr =
                  gradients_quad + batch * n_block_columns;
                if (matrix_free.get_mapping_info().cell_type[cell + batch] <=
                    internal::MatrixFreeFunctions::affine)
                  {
                    // const SymmetricTensor<2, dim, VectorizedArray<number>>
                    //  my_metric = j_value[0] * symmetrize(transpose(jac[0]) *
                    //  jac[0]);
                    SymmetricTensor<2, dim, VectorizedArray<number>> my_metric;
                    for (unsigned int d = 0; d < dim; ++d)
                      for (unsigned int f = d; f < dim; ++f)
                        {
                          VectorizedArray<number> sum =
                            jac[0][0][d] * jac[0][0][f];
                          for (unsigned int e = 1; e < dim; ++e)
                            sum += jac[0][e][d] * jac[0][e][f];
                          my_metric[d][f] = sum * j_value[0];
                        }

                    for (unsigned int q_local = 0; q_local < n_q_block;
                         ++q_local, grad_ptr += dim)
                      {
                        Tensor<1, dim, VectorizedArray<number>> grad;
                        for (unsigned int d = 0; d < dim; ++d)
                          grad[d] = grad_ptr[d];
                        Tensor<1, dim, VectorizedArray<number>> result =
                          my_metric * grad;
                        const number weight =
                          quadrature_weights[q_begin + q_local];
                        for (unsigned int d = 0; d < dim; ++d)
                          grad_ptr[d] = weight * result[d];
                      }
                  }
                else
                  {
                    DEAL_II_NOT_IMPLEMENTED();
                    for (unsigned int q = 0; q < n_q_points;
                         ++q, grad_ptr += dim)
                      {
                        Tensor<1, dim, VectorizedArray<number>> grad;
                        for (unsigned int d = 0; d < dim; ++d)
                          grad[d] = grad_ptr[d];
                        Tensor<1, dim, VectorizedArray<number>> result =
                          j_value[q] * (transpose(jac[q]) * (jac[q] * grad));
                        for (unsigned int d = 0; d < dim; ++d)
                          grad_ptr[d] = result[d];
                      }
                  }
              }

            // integrate
            // apply_matrix_vector_product<true, true>(shape_block,
            //                                         gradients_quad,
            //                                         values_dofs_out,
            //                                         n_block_columns,
            //                                         dofs_per_cell);
            if (n_q_block == q_block_size_effective)
              apply_matrix_vector_product_templated<false,
                                                    true,
                                                    dofs_per_cell,
                                                    q_block_size_effective *
                                                      dim>(
                shape_gradients_packed.begin() + q_begin * dim * dofs_per_cell,
                gradients_quad,
                values_dofs_out);
            else
              apply_matrix_vector_product<false, true>(
                shape_gradients_packed.begin() + q_begin * dim * dofs_per_cell,
                gradients_quad,
                values_dofs_out,
                dofs_per_cell,
                n_block_columns);


            // advance shape block
            shape_block += n_block_columns * dofs_per_cell;
          }

        // distribute local to global
        dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  {
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        dst.local_element(dof_indices[v]) +=
                          values_dofs_out[batch * dofs_per_cell + i][v];
                  }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    dst.local_element(dof_indices[v]) +=
                      values_dofs_out[batch * dofs_per_cell + i][v];
                }
          }
      }

    matrix_free.release_scratch_data(scratch_data);
  }



  void do_cell_integral_dgemm_non_opt(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    AlignedVector<VectorizedArray<number>> *scratch_data =
      matrix_free.acquire_scratch_data();
    const internal::MatrixFreeFunctions::ShapeInfo<number> &shape_info =
      matrix_free.get_shape_info();
    const unsigned int dofs_per_cell = shape_info.dofs_per_component_on_cell;
    // constexpr unsigned int batch_size_dgemm = 16;
    const unsigned int     n_q_points = shape_info.n_q_points;
    constexpr unsigned int n_lanes    = VectorizedArray<number>::size();

    const auto   &mapping_data = matrix_free.get_mapping_info().cell_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();

    scratch_data->resize_fast(batch_size_dgemm *
                              (dim * n_q_points + dofs_per_cell));
    number *values_dofs = &((scratch_data->begin())[0][0]);
    number *gradients_quad =
      &((scratch_data->begin() + batch_size_dgemm * dofs_per_cell)[0][0]);

    const number            *src_ptr = src.begin();
    VectorizedArray<number> *grad_ptr =
      reinterpret_cast<VectorizedArray<number> *>(gradients_quad);

    const Number          alpha = 1.;
    const Number          beta  = 0.;
    const types::blas_int m     = static_cast<types::blas_int>(dofs_per_cell);
    const types::blas_int n = static_cast<types::blas_int>(n_q_points * dim);

    for (unsigned int cell = range.first; cell < range.second;
         cell += batch_size_dgemm)
      {
        // TODO: use remainder loop for last batch
        // read dof values
        const unsigned int my_batch_size =
          cell + batch_size_dgemm <= range.second ? batch_size_dgemm :
                                                    range.second - cell;
        const unsigned int current_batch_size = my_batch_size * n_lanes;

        const unsigned int *dof_indices = &manual_dof_indices(cell, 0);

        std::fill(values_dofs,
                  values_dofs + dofs_per_cell * current_batch_size,
                  number(0));

        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    {
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        values_dofs[i * current_batch_size + batch * n_lanes +
                                    v] = src_ptr[dof_indices[v]];
                    }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    values_dofs[i * current_batch_size + batch * n_lanes + v] =
                      src_ptr[dof_indices[v]];
                }
          }

        // interpolate
        const types::blas_int k =
          static_cast<types::blas_int>(current_batch_size);
        // Use the BLAS function gemm for calculating the matrix-matrix
        // product.
        gemm("n",
             "n",
             &k,
             &n,
             &m,
             &alpha,
             values_dofs,
             &k,
             shape_gradients_transpose.data(),
             &m,
             &beta,
             gradients_quad,
             &k);

        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int offsets =
              mapping_data.data_index_offsets[cell + batch];
            const Tensor<2, dim, VectorizedArray<number>> *jac =
              mapping_data.jacobians[0].data() + offsets;
            const VectorizedArray<number> *j_value =
              &mapping_data.JxW_values[offsets];


            if (matrix_free.get_mapping_info().cell_type[cell + batch] <=
                internal::MatrixFreeFunctions::affine)
              {
                SymmetricTensor<2, dim, VectorizedArray<number>> my_metric;
                for (unsigned int d = 0; d < dim; ++d)
                  for (unsigned int f = d; f < dim; ++f)
                    {
                      VectorizedArray<number> sum = jac[0][0][d] * jac[0][0][f];
                      for (unsigned int e = 1; e < dim; ++e)
                        sum += jac[0][e][d] * jac[0][e][f];
                      my_metric[d][f] = sum * j_value[0];
                    }

                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    Tensor<1, dim, VectorizedArray<number>> grad;
                    for (unsigned int d = 0; d < dim; ++d)
                      grad[d] = grad_ptr[(q * dim + d) * my_batch_size + batch];


                    Tensor<1, dim, VectorizedArray<number>> result =
                      my_metric * grad;
                    const number weight = quadrature_weights[q];
                    for (unsigned int d = 0; d < dim; ++d)
                      grad_ptr[(q * dim + d) * my_batch_size + batch] =
                        weight * result[d];
                  }
              }
            else
              {
                // TODO: this is wrong
                DEAL_II_NOT_IMPLEMENTED();
                for (unsigned int q = 0; q < n_q_points; ++q, grad_ptr += dim)
                  {
                    Tensor<1, dim, VectorizedArray<number>> grad;
                    for (unsigned int d = 0; d < dim; ++d)
                      grad[d] = grad_ptr[d];
                    Tensor<1, dim, VectorizedArray<number>> result =
                      j_value[q] * (transpose(jac[q]) * (jac[q] * grad));
                    for (unsigned int d = 0; d < dim; ++d)
                      grad_ptr[d] = result[d];
                  }
              }
          }

        // integrate
        gemm("n",
             "t",
             &k,
             &m,
             &n,
             &alpha,
             gradients_quad,
             &k,
             shape_gradients_transpose.data(),
             &m,
             &beta,
             values_dofs,
             &k);

        // distribute local to global
        dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    {
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        dst.local_element(dof_indices[v]) +=
                          values_dofs[i * current_batch_size + batch * n_lanes +
                                      v];
                    }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    dst.local_element(dof_indices[v]) +=
                      values_dofs[i * current_batch_size + batch * n_lanes + v];
                }
          }
      }

    matrix_free.release_scratch_data(scratch_data);
  }


  // DGEMM with quadrature blocking
  void do_cell_integral_dgemm(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    AlignedVector<VectorizedArray<number>> *scratch_data =
      matrix_free.acquire_scratch_data();
    const internal::MatrixFreeFunctions::ShapeInfo<number> &shape_info =
      matrix_free.get_shape_info();
    const unsigned int dofs_per_cell = shape_info.dofs_per_component_on_cell;
    // constexpr unsigned int batch_size_dgemm   = 16;
    // constexpr unsigned int q_block_size_dgemm = 16;
    const unsigned int     n_q_points = shape_info.n_q_points;
    constexpr unsigned int n_lanes    = VectorizedArray<number>::size();

    const auto   &mapping_data = matrix_free.get_mapping_info().cell_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();


    Assert((matrix_free.get_mapping_info().cell_type[range.first] <=
            internal::MatrixFreeFunctions::affine),
           ExcInternalError());
    // TODO: this should be checked for all cells

    scratch_data->resize_fast(batch_size_dgemm *
                              (dim * q_block_size_dgemm + 2 * dofs_per_cell));
    number *values_dofs_input = &((scratch_data->begin())[0][0]);
    number *values_dofs_output =
      &((scratch_data->begin() + batch_size_dgemm * dofs_per_cell)[0][0]);
    number *gradients_quad =
      &((scratch_data->begin() + 2 * batch_size_dgemm * dofs_per_cell)[0][0]);

    const number            *src_ptr = src.begin();
    VectorizedArray<number> *grad_ptr =
      reinterpret_cast<VectorizedArray<number> *>(gradients_quad);

    const Number          alpha = 1.;
    const Number          beta  = 0.;
    const types::blas_int m     = static_cast<types::blas_int>(dofs_per_cell);

    for (unsigned int cell = range.first; cell < range.second;
         cell += batch_size_dgemm)
      {
        // TODO: use remainder loop for last batch
        // read dof values
        const unsigned int my_batch_size =
          cell + batch_size_dgemm <= range.second ? batch_size_dgemm :
                                                    range.second - cell;
        const unsigned int    current_batch_size = my_batch_size * n_lanes;
        const types::blas_int k =
          static_cast<types::blas_int>(current_batch_size);

        const unsigned int *dof_indices = &manual_dof_indices(cell, 0);
        std::fill(values_dofs_input,
                  values_dofs_input + dofs_per_cell * current_batch_size,
                  number(0));
        std::fill(values_dofs_output,
                  values_dofs_output + dofs_per_cell * current_batch_size,
                  number(0));

        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    {
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        values_dofs_input[i * current_batch_size +
                                          batch * n_lanes + v] =
                          src_ptr[dof_indices[v]];
                    }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    values_dofs_input[i * current_batch_size + batch * n_lanes +
                                      v] = src_ptr[dof_indices[v]];
                }
          }

        // block over quadrature points
        const number *shape_block = shape_gradients_transpose.data();
        for (unsigned int q_begin = 0; q_begin < n_q_points;
             q_begin += q_block_size_dgemm)
          {
            const unsigned int n_q_block =
              std::min(static_cast<unsigned int>(q_block_size_dgemm),
                       n_q_points - q_begin);

            const types::blas_int n =
              static_cast<types::blas_int>(n_q_block * dim);

            // interpolate
            // Use the BLAS function gemm for calculating the matrix-matrix
            // product.
            gemm("n",
                 "n",
                 &k,
                 &n,
                 &m,
                 &alpha,
                 values_dofs_input,
                 &k,
                 shape_block,
                 &m,
                 &beta,
                 gradients_quad,
                 &k);


            for (unsigned int batch = 0; batch < my_batch_size; ++batch)
              {
                const unsigned int offsets =
                  mapping_data.data_index_offsets[cell + batch];
                const Tensor<2, dim, VectorizedArray<number>> *jac =
                  mapping_data.jacobians[0].data() + offsets;
                const VectorizedArray<number> *j_value =
                  &mapping_data.JxW_values[offsets];

                if (matrix_free.get_mapping_info().cell_type[cell + batch] <=
                    internal::MatrixFreeFunctions::affine)
                  {
                    SymmetricTensor<2, dim, VectorizedArray<number>> my_metric;
                    for (unsigned int d = 0; d < dim; ++d)
                      for (unsigned int f = d; f < dim; ++f)
                        {
                          VectorizedArray<number> sum =
                            jac[0][0][d] * jac[0][0][f];
                          for (unsigned int e = 1; e < dim; ++e)
                            sum += jac[0][e][d] * jac[0][e][f];
                          my_metric[d][f] = sum * j_value[0];
                        }

                    for (unsigned int q_local = 0; q_local < n_q_block;
                         ++q_local)
                      {
                        const unsigned int q = q_local + q_begin;
                        Tensor<1, dim, VectorizedArray<number>> grad;
                        for (unsigned int d = 0; d < dim; ++d)
                          grad[d] =
                            grad_ptr[(q_local * dim + d) * my_batch_size +
                                     batch];

                        Tensor<1, dim, VectorizedArray<number>> result =
                          my_metric * grad;
                        const number weight = quadrature_weights[q];
                        for (unsigned int d = 0; d < dim; ++d)
                          grad_ptr[(q_local * dim + d) * my_batch_size +
                                   batch] = weight * result[d];
                      }
                  }
                else
                  {
                    // TODO: this is wrong
                    DEAL_II_NOT_IMPLEMENTED();
                    for (unsigned int q = 0; q < n_q_points;
                         ++q, grad_ptr += dim)
                      {
                        Tensor<1, dim, VectorizedArray<number>> grad;
                        for (unsigned int d = 0; d < dim; ++d)
                          grad[d] = grad_ptr[d];
                        Tensor<1, dim, VectorizedArray<number>> result =
                          j_value[q] * (transpose(jac[q]) * (jac[q] * grad));
                        for (unsigned int d = 0; d < dim; ++d)
                          grad_ptr[d] = result[d];
                      }
                  }
              }

            // integrate
            gemm("n",
                 "t",
                 &k,
                 &m,
                 &n,
                 &alpha,
                 gradients_quad,
                 &k,
                 shape_block,
                 &m,
                 &alpha,
                 values_dofs_output,
                 &k);

            // advance shape_block
            shape_block += dim * dofs_per_cell * n_q_block;
          }

        // distribute local to global
        dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    {
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        dst.local_element(dof_indices[v]) +=
                          values_dofs_output[i * current_batch_size +
                                             batch * n_lanes + v];
                    }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    dst.local_element(dof_indices[v]) +=
                      values_dofs_output[i * current_batch_size +
                                         batch * n_lanes + v];
                }
          }
      }

    matrix_free.release_scratch_data(scratch_data);
  }

  MatrixFree<dim, number> matrix_free;

  AffineConstraints<number> constraints;

  std::vector<unsigned int> constrained_indices;

  Table<2, unsigned int> manual_dof_indices;

  std::vector<unsigned char> dof_indices_have_constraints;

  AlignedVector<number> shape_gradients_transpose;

  AlignedVector<number> shape_gradients_packed;
};



template <int dim,
          int fe_degree,
          int q_block_size,
          int batch_size_dgemm,
          int q_block_size_dgemm,
          typename Number>
void do_test(const unsigned int n_cycles_max)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);
  pcout
    << "Running in " << dim << "D with degree " << fe_degree
    << " on tet elements with q_block_size, batch_size_dgemm, q_block_size_dgemm: "
    << q_block_size << ", " << batch_size_dgemm << ", " << q_block_size_dgemm
    << std::endl;

  FE_SimplexP<dim>          mapping_fe_simplex(1, true);
  MappingFE<dim>            mapping(mapping_fe_simplex);
  QGaussSimplex<dim>        quad(fe_degree + 1);
  AffineConstraints<double> constraint;

  for (unsigned int cycle = 1; cycle < n_cycles_max; ++cycle)
    {
      const auto serial_grid_generator =
        [&cycle](dealii::Triangulation<dim, dim> &tria_serial) {
          // set up triangulation
          GridGenerator::subdivided_hyper_cube_with_simplices(tria_serial, 2);
          if (cycle > 0)
            tria_serial.refine_global(cycle);
        };
      const auto serial_grid_partitioner =
        [&](dealii::Triangulation<dim, dim> &tria_serial,
            const MPI_Comm                   comm,
            const unsigned int) {
          dealii::GridTools::partition_triangulation(
            dealii::Utilities::MPI::n_mpi_processes(comm), tria_serial);
        };

      const unsigned int group_size = 20;

      parallel::fullydistributed::Triangulation<dim> tria(MPI_COMM_WORLD);
      typename dealii::TriangulationDescription::Settings
                 triangulation_description_setting =
                   dealii::TriangulationDescription::default_setting;
      const auto description = dealii::TriangulationDescription::Utilities::
        create_description_from_triangulation_in_groups<dim, dim>(
          serial_grid_generator,
          serial_grid_partitioner,
          tria.get_mpi_communicator(),
          group_size,
          dealii::Triangulation<dim>::none,
          triangulation_description_setting);

      tria.create_triangulation(description);
      pcout << "Cycle " << cycle << " set up triangulation" << std::endl;

      DoFHandler<dim>  dof_handler(tria);
      FE_SimplexP<dim> fe(fe_degree, false);

      pcout << "reinit triangulation done...";
      dof_handler.distribute_dofs(fe);
      pcout << " distributed dofs" << std::endl;

      // set up constraints, then renumber dofs, and set up constraints
      // again
      if (true)
        {
          const IndexSet locally_relevant_dofs =
            DoFTools::extract_locally_relevant_dofs(dof_handler);
          constraint.reinit(dof_handler.locally_owned_dofs(),
                            locally_relevant_dofs);
          VectorTools::interpolate_boundary_values(
            mapping,
            dof_handler,
            0,
            Functions::ZeroFunction<dim>(),
            constraint);
          constraint.close();
          typename MatrixFree<dim, Number>::AdditionalData data;
          DoFRenumbering::matrix_free_data_locality(dof_handler,
                                                    constraint,
                                                    data);
        }
      const IndexSet locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);
      constraint.reinit(dof_handler.locally_owned_dofs(),
                        locally_relevant_dofs);
      VectorTools::interpolate_boundary_values(
        mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraint);
      constraint.close();



      pcout << "Set up operator of degree " << fe_degree << std::endl;
      Operator<dim,
               fe_degree,
               q_block_size,
               batch_size_dgemm,
               q_block_size_dgemm,
               1,
               Number>
        op;
      // set up operator
      op.reinit(mapping,
                dof_handler,
                quad,
                constraint,
                numbers::invalid_unsigned_int,
                false);

      LinearAlgebra::distributed::Vector<Number> vec1, vec2, vec3, vec4;
      op.initialize_dof_vector(vec1);
      op.initialize_dof_vector(vec2);
      op.initialize_dof_vector(vec3);
      op.initialize_dof_vector(vec4);
      for (Number &a : vec1)
        a = static_cast<double>(rand()) / RAND_MAX;

      for (unsigned int r = 0; r < 1; ++r)
        {
          Timer time;
          for (unsigned int t = 0; t < 10; ++t)
            op.vmult(vec2, vec1);
          const double run_time = time.wall_time();
          pcout << "n_dofs mf basic  " << dof_handler.n_dofs() << "  time "
                << run_time / 10 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 10 / run_time << std::endl;
        }
      for (unsigned int r = 0; r < 5; ++r)
        {
          Timer time;
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("matvec_masked_gather_p" +
                               std::to_string(fe_degree) + "_s" +
                               std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          for (unsigned int t = 0; t < 100; ++t)
            op.vmult_masked_gather(vec3, vec1);
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(("matvec_masked_gather_p" +
                              std::to_string(fe_degree) + "_s" +
                              std::to_string(dof_handler.n_dofs()))
                               .c_str());
#endif
          const double run_time = time.wall_time();
          pcout << "n_dofs mf mk gthr " << dof_handler.n_dofs() << " time "
                << run_time / 100 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
        }
      if (false)
        for (unsigned int r = 0; r < 5; ++r)
          {
            Timer time;
#ifdef LIKWID_PERFMON
            LIKWID_MARKER_START(("matvec_dgemm_p" + std::to_string(fe_degree) +
                                 "_s" + std::to_string(dof_handler.n_dofs()))
                                  .c_str());
#endif
            for (unsigned int t = 0; t < 100; ++t)
              op.vmult_dgemm(vec4, vec1);
#ifdef LIKWID_PERFMON
            LIKWID_MARKER_STOP(("matvec_dgemm_p" + std::to_string(fe_degree) +
                                "_s" + std::to_string(dof_handler.n_dofs()))
                                 .c_str());
#endif
            const double run_time = time.wall_time();
            pcout << "n_dofs mf dgemm " << dof_handler.n_dofs() << "  time "
                  << run_time / 100 << "  GDoFs/s "
                  << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
          }
      pcout << std::endl;

      vec3 -= vec2;
      // vec4 -= vec2;

      pcout << "   Error MF variants: " << vec3.l2_norm() / vec2.l2_norm()
            << std::endl;
      //   << " " << vec4.l2_norm() / vec2.l2_norm() << std::endl;
      pcout << std::endl << std::endl;
    }
  pcout << std::endl;
  pcout << std::endl;
}


int main(int argc, char **argv)
{
  constexpr int dim                = 3;
  constexpr int q_block_size       = 8;
  constexpr int batch_size_dgemm   = 16;
  constexpr int q_block_size_dgemm = 16;

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  int min_degree   = 1;
  int max_degree   = 7;
  int n_cycles_max = 7;
  if (argc > 1)
    min_degree = std::atoi(argv[1]);
  if (argc > 2)
    max_degree = std::atoi(argv[2]);
  if (argc > 3)
    n_cycles_max = std::atoi(argv[3]);

  if (min_degree == 1)
    do_test<dim, 1, q_block_size, batch_size_dgemm, q_block_size_dgemm, double>(
      n_cycles_max);
  if (min_degree <= 2 && 2 <= max_degree)
    do_test<dim, 2, q_block_size, batch_size_dgemm, q_block_size_dgemm, double>(
      n_cycles_max);
  if (min_degree <= 3 && 3 <= max_degree)
    do_test<dim, 3, q_block_size, batch_size_dgemm, q_block_size_dgemm, double>(
      n_cycles_max);
  if (min_degree <= 4 && 4 <= max_degree)
    do_test<dim, 4, q_block_size, batch_size_dgemm, q_block_size_dgemm, double>(
      n_cycles_max);
  if (min_degree <= 5 && 5 <= max_degree)
    do_test<dim, 5, q_block_size, batch_size_dgemm, q_block_size_dgemm, double>(
      n_cycles_max);
  if (min_degree <= 6 && 6 <= max_degree)
    do_test<dim, 6, q_block_size, batch_size_dgemm, q_block_size_dgemm, double>(
      n_cycles_max);
  if (min_degree <= 7 && 7 <= max_degree)
    do_test<dim, 7, q_block_size, batch_size_dgemm, q_block_size_dgemm, double>(
      n_cycles_max);
  if (min_degree <= 8 && 8 <= max_degree)
    do_test<dim, 8, q_block_size, batch_size_dgemm, q_block_size_dgemm, double>(
      n_cycles_max);
  if (min_degree <= 9 && 9 <= max_degree)
    do_test<dim, 9, q_block_size, batch_size_dgemm, q_block_size_dgemm, double>(
      n_cycles_max);
  if (min_degree <= 10 && 10 <= max_degree)
    do_test<dim,
            10,
            q_block_size,
            batch_size_dgemm,
            q_block_size_dgemm,
            double>(n_cycles_max);



#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
