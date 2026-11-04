
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

template <bool transpose_matrix, typename Number, typename Number2>
void apply_matrix_vector_product_dgemm(const Number2 *matrix,
                                       const Number  *in,
                                       Number        *out,
                                       const int      n_rows,
                                       const int      n_columns)
{
#ifdef DEAL_II_WITH_LAPACK
  // if constexpr(transpose_matrix)
  //{}
  // else
  {
    const types::blas_int m       = static_cast<types::blas_int>(n_columns);
    const types::blas_int n       = static_cast<types::blas_int>(n_rows);
    const types::blas_int k       = static_cast<types::blas_int>(n_columns);
    const char           *notrans = "n";
    const Number2         alpha   = 1.;
    const Number2         beta    = 0.;

    dgemm(notrans,
          notrans,
          &m,
          &n,
          &k,
          &alpha,
          in, //&src(0, 0),
          &m,
          matrix,
          &k,
          &beta,
          out, //&dst(0, 0),
          &m);
  }

  return;
#endif

  DEAL_II_NOT_IMPLEMENTED();
}

template <bool transpose_matrix, typename Number, typename Number2>
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
          out0[i] = res[0][i];
          out1[i] = res[1][i];
          out2[i] = res[2][i];
          out3[i] = res[3][i];
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
      out0[0] = res0;
      out0[1] = res1;
      out1[0] = res2;
      out1[1] = res3;
      out2[0] = res4;
      out2[1] = res5;
      out3[0] = res6;
      out3[1] = res7;
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
      out0[0] = res0;
      out1[0] = res1;
      out2[0] = res2;
      out3[0] = res3;
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

template <int dim_, int n_components = dim_, typename Number = double>
class Operator : public Subscriptor
{
public:
  using value_type = Number;
  using number     = Number;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  static const int dim = dim_;

  using FECellIntegrator = FEEvaluation<dim, -1, 0, n_components, Number>;

  void reinit(const hp::MappingCollection<dim> &mapping,
              const DoFHandler<dim>            &dof_handler,
              const hp::QCollection<dim>       &quad,
              const AffineConstraints<number>  &constraints,
              const unsigned int mg_level = numbers::invalid_unsigned_int,
              const bool         ones_on_diagonal = false)
  {
    this->constraints.copy_from(constraints);

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags = update_values | update_gradients |
                                update_JxW_values | update_quadrature_points;
    data.mg_level = mg_level;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Sizes shape info: "
                << matrix_free.get_shape_info()
                     .data[0]
                     .shape_values.memory_consumption()
                << " "
                << matrix_free.get_shape_info()
                     .data[0]
                     .shape_gradients.memory_consumption()
                << " " << dof_handler.get_fe().dofs_per_cell << " "
                << matrix_free.get_shape_info().n_q_points << " "
                << matrix_free.get_shape_info().dofs_per_component_on_cell
                << " " << matrix_free.get_dof_info(0).dof_indices.size() << " "
                << dof_handler.get_triangulation().n_active_cells() << " "
                << dof_handler.n_dofs() << std::endl;

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
        apply_matrix_vector_product<true>(
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
        apply_matrix_vector_product<false>(
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



  void do_cell_integral_dgemm(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    // AlignedVector<VectorizedArray<number>> *scratch_data =
    //   matrix_free.acquire_scratch_data();
    AlignedVector<number>                                   data;
    const internal::MatrixFreeFunctions::ShapeInfo<number> &shape_info =
      matrix_free.get_shape_info();
    const unsigned int dofs_per_cell  = shape_info.dofs_per_component_on_cell;
    constexpr unsigned int batch_size = 4;
    const unsigned int     n_q_points = shape_info.n_q_points;
    constexpr unsigned int n_lanes    = VectorizedArray<number>::size();

    const auto   &mapping_data = matrix_free.get_mapping_info().cell_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();

    data.resize_fast(n_lanes * batch_size * (dim * n_q_points + dofs_per_cell));
    number *values_dofs = data.begin();
    number *gradients_quad =
      data.begin() + n_lanes * batch_size * dofs_per_cell;

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
                    // values_dofs[n_lanes * batch * dofs_per_cell + n_lanes *
                    // i] = {};
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        values_dofs[n_lanes * batch * dofs_per_cell +
                                    n_lanes * i + v] = src_ptr[dof_indices[v]];
                  }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  // values_dofs[n_lanes * batch * dofs_per_cell + n_lanes * i]
                  // = {};
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    values_dofs[n_lanes * batch * dofs_per_cell + n_lanes * i +
                                v] = src_ptr[dof_indices[v]];
                }
          }

        // interpolate
        apply_matrix_vector_product_dgemm<true>(
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
        apply_matrix_vector_product_dgemm<false>(
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

  MatrixFree<dim, number> matrix_free;

  AffineConstraints<number> constraints;

  std::vector<unsigned int> constrained_indices;

  Table<2, unsigned int> manual_dof_indices;

  std::vector<unsigned char> dof_indices_have_constraints;
};



template <int dim, typename Number>
void do_test(const unsigned int min_degree,
             const unsigned int max_degree,
             const unsigned int n_cycles_max)
{
  const bool solve = false;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);
  pcout << "Running in " << dim << "D with degrees between " << min_degree
        << " and " << max_degree << " on mixed elements" << std::endl;

  std::vector<ConvergenceTable> convergence_tables;
  convergence_tables.resize(2 * (max_degree - min_degree + 1));

  FE_PyramidP<dim> mapping_fe_pyramid(1, true);
  FE_WedgeP<dim>   mapping_fe_wedge(1, true);
  FE_SimplexP<dim> mapping_fe_simplex(1, true);
  FE_Q<dim>        mapping_fe_hypercube(1);

  MappingFE<dim> mapping_pyramid(mapping_fe_pyramid);
  MappingFE<dim> mapping_wedge(mapping_fe_wedge);
  MappingFE<dim> mapping_simplex(mapping_fe_simplex);
  MappingFE<dim> mapping_hypercube(mapping_fe_hypercube);

  const hp::MappingCollection<dim> mapping(mapping_pyramid,
                                           mapping_wedge,
                                           mapping_simplex,
                                           mapping_hypercube);

  AffineConstraints<double> constraint;
  const unsigned int        n_cells_max = 200000000;
  unsigned int              n_cells     = 1;

  const unsigned int n_dofs_max = 33000000;
  unsigned int       n_dofs     = 1;

  for (unsigned int cycle = 0; cycle < n_cycles_max && n_cells < n_cells_max;
       ++cycle)
    {
      const auto serial_grid_generator =
        [&cycle](dealii::Triangulation<dim, dim> &tria_serial) {
          // set up triangulation
          std::vector<Point<dim>>    vertices;
          std::vector<CellData<dim>> cells;
          vertices.emplace_back(0.0, 0.0, 0.0);  // 0
          vertices.emplace_back(1.0, 0.0, 0.0);  // 1
          vertices.emplace_back(0.0, 1.0, 0.0);  // 2
          vertices.emplace_back(1.0, 1.0, 0.0);  // 3
          vertices.emplace_back(0.0, 0.0, 1.0);  // 4
          vertices.emplace_back(1.0, 0.0, 1.0);  // 5
          vertices.emplace_back(0.0, 1.0, 1.0);  // 6
          vertices.emplace_back(1.0, 1.0, 1.0);  // 7
          vertices.emplace_back(2, 0.5, 0.25);   // 8
          vertices.emplace_back(2, 0.5, 0.75);   // 9
          vertices.emplace_back(-1.0, 0.5, 0.5); // 10
          vertices.emplace_back(-1.0, 0.5, 1);   // 11
          vertices.emplace_back(-1.0, 0.5, 0);   // 12
          vertices.emplace_back(-1.0, 0.0, 1.0); // 13
          vertices.emplace_back(-1.0, 1.0, 1.0); // 14
          vertices.emplace_back(-1.0, 0.0, 0.0); // 15
          vertices.emplace_back(-1.0, 1.0, 0.0); // 16
          vertices.emplace_back(-1.0, 1.0, 0.5); // 17
          vertices.emplace_back(-1.0, 0.0, 0.5); // 18
          vertices.emplace_back(2, 0.5, 0.0);    // 19
          vertices.emplace_back(2, 0.5, 1.0);    // 20
          vertices.emplace_back(2, 0.0, 0.0);    // 21
          vertices.emplace_back(2, 0.0, 1.0);    // 22
          vertices.emplace_back(2, 1.0, 0.0);    // 23
          vertices.emplace_back(2, 1.0, 1.0);    // 24
          for (auto &p : vertices)
            {
              const double x = p[0];
              p[0]           = (x + 1.0) / 3.0;
            }
          {
            CellData<dim> hex;
            hex.vertices = {0, 1, 2, 3, 4, 5, 6, 7};
            cells.push_back(hex);
          }
          {
            CellData<dim> wedge;
            wedge.vertices = {1, 8, 3, 5, 9, 7};
            cells.push_back(wedge);
          }
          {
            CellData<dim> pyramid;
            pyramid.vertices = {0, 4, 2, 6, 10};
            cells.push_back(pyramid);
          }
          {
            CellData<dim> tet;
            tet.vertices = {4, 6, 10, 11};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {0, 2, 12, 10};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 11, 13, 4};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 11, 6, 14};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 12, 0, 15};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 12, 16, 2};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {17, 10, 6, 14};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 17, 6, 2};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 17, 2, 16};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 18, 15, 0};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 18, 0, 4};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 18, 4, 13};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {5, 9, 7, 20};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {1, 3, 8, 19};
            cells.push_back(tet);
          }
          {
            CellData<dim> wedge;
            wedge.vertices = {8, 1, 21, 9, 5, 22};
            cells.push_back(wedge);
          }
          {
            CellData<dim> tet;
            tet.vertices = {5, 20, 22, 9};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {1, 21, 19, 8};
            cells.push_back(tet);
          }
          {
            CellData<dim> wedge;
            wedge.vertices = {3, 8, 23, 7, 9, 24};
            cells.push_back(wedge);
          }
          {
            CellData<dim> tet;
            tet.vertices = {20, 9, 7, 24};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {19, 3, 8, 23};
            cells.push_back(tet);
          }

          tria_serial.create_triangulation(vertices, cells, SubCellData());

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

      const unsigned int group_size = 32;

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

      unsigned int cycle_max_degree = max_degree;
      if (cycle > 4)
        cycle_max_degree = std::min(cycle_max_degree, 5U);
      if (cycle > 5)
        cycle_max_degree = std::min(cycle_max_degree, 4U);
      if (cycle > 6)
        cycle_max_degree = std::min(cycle_max_degree, 3U);

      bool continue_iterating = true;
      for (unsigned int fe_degree = min_degree;
           fe_degree <= cycle_max_degree && n_dofs < n_dofs_max &&
           continue_iterating;
           ++fe_degree)
        for (const bool use_equidistant_points :
             std::vector<bool>{{true, false}})
          {
            DoFHandler<dim> dof_handler(tria);

            for (const auto &cell : dof_handler.active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  if (cell->reference_cell() == ReferenceCells::Pyramid)
                    cell->set_active_fe_index(0);
                  else if (cell->reference_cell() == ReferenceCells::Wedge)
                    cell->set_active_fe_index(1);
                  else if (cell->reference_cell().is_simplex())
                    cell->set_active_fe_index(2);
                  else if (cell->reference_cell().is_hyper_cube())
                    cell->set_active_fe_index(3);
                  else
                    DEAL_II_NOT_IMPLEMENTED();
                }

            FE_PyramidP<dim> fe_pyramidp(fe_degree, use_equidistant_points);
            FE_WedgeP<dim>   fe_wedgep(fe_degree, use_equidistant_points);
            FE_SimplexP<dim> fe_simplexp(fe_degree, use_equidistant_points);
            FE_Q<dim>        fe_q =
              use_equidistant_points ?
                       FE_Q<dim>(QIterated<1>(QTrapezoid<1>(), fe_degree)) :
                       FE_Q<dim>(fe_degree);

            const hp::FECollection<dim> fe(fe_pyramidp,
                                           fe_wedgep,
                                           fe_simplexp,
                                           fe_q);

            pcout << "reinit triangulation done, there are "
                  << tria.get_reference_cells().size()
                  << " different types of cells and and fe collection of size "
                  << fe.size() << std::endl;

            dof_handler.distribute_dofs(fe);

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
              mapping,
              dof_handler,
              0,
              Functions::ZeroFunction<dim>(),
              constraint);
            constraint.close();


            QGaussPyramid<dim> quad_pyramid(fe_degree + 1);
            QGaussWedge<dim>   quad_wedge(fe_degree + 1);
            QGaussSimplex<dim> quad_simplex(fe_degree + 1);
            QGauss<dim>        quad_hypercube(fe_degree + 1);

            const hp::QCollection<dim> quad(quad_pyramid,
                                            quad_wedge,
                                            quad_simplex,
                                            quad_hypercube);

            pcout << "Set up operator of degree " << fe_degree << std::endl;
            Operator<dim, 1, Number> op;
            // set up operator
            op.reinit(mapping,
                      dof_handler,
                      quad,
                      constraint,
                      numbers::invalid_unsigned_int,
                      false);

            if (!solve)
              {
                LinearAlgebra::distributed::Vector<Number> vec1, vec2, vec3,
                  vec4;
                op.initialize_dof_vector(vec1);
                op.initialize_dof_vector(vec2);
                op.initialize_dof_vector(vec3);
                op.initialize_dof_vector(vec4);
                for (Number &a : vec1)
                  a = static_cast<double>(rand()) / RAND_MAX;

                for (unsigned int r = 0; r < 5; ++r)
                  {
                    Timer time;
#ifdef LIKWID_PERFMON
                    LIKWID_MARKER_START(("matvec_p" +
                                         std::to_string(fe_degree) + "_s" +
                                         std::to_string(dof_handler.n_dofs()))
                                          .c_str());
#endif
                    for (unsigned int t = 0; t < 100; ++t)
                      op.vmult(vec2, vec1);
#ifdef LIKWID_PERFMON
                    LIKWID_MARKER_STOP(("matvec_p" + std::to_string(fe_degree) +
                                        "_s" +
                                        std::to_string(dof_handler.n_dofs()))
                                         .c_str());
#endif
                    const double run_time = time.wall_time();
                    pcout << "n_dofs mf basic  " << dof_handler.n_dofs()
                          << "  time " << run_time / 100 << "  GDoFs/s "
                          << 1e-9 * dof_handler.n_dofs() * 100 / run_time
                          << std::endl;
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
                    pcout << "n_dofs mf masked gather " << dof_handler.n_dofs()
                          << "  time " << run_time / 100 << "  GDoFs/s "
                          << 1e-9 * dof_handler.n_dofs() * 100 / run_time
                          << std::endl;
                  }
                for (unsigned int r = 0; r < 5; ++r)
                  {
                    Timer time;
#ifdef LIKWID_PERFMON
                    LIKWID_MARKER_START(("matvec_dgemm_p" +
                                         std::to_string(fe_degree) + "_s" +
                                         std::to_string(dof_handler.n_dofs()))
                                          .c_str());
#endif
                    for (unsigned int t = 0; t < 100; ++t)
                      op.vmult_dgemm(vec4, vec1);
#ifdef LIKWID_PERFMON
                    LIKWID_MARKER_STOP(("matvec_dgemm_p" +
                                        std::to_string(fe_degree) + "_s" +
                                        std::to_string(dof_handler.n_dofs()))
                                         .c_str());
#endif
                    const double run_time = time.wall_time();
                    pcout << "n_dofs mf dgemm " << dof_handler.n_dofs()
                          << "  time " << run_time / 100 << "  GDoFs/s "
                          << 1e-9 * dof_handler.n_dofs() * 100 / run_time
                          << std::endl;
                  }
                pcout << std::endl;

                vec3 -= vec2;
                vec4 -= vec2;

                pcout << "   Error MF variants: "
                      << vec3.l2_norm() / vec2.l2_norm() << " "
                      << vec4.l2_norm() / vec2.l2_norm() << " " << std::endl;
              }
            if (solve)
              {
                LinearAlgebra::distributed::Vector<Number> x, rhs;
                op.initialize_dof_vector(x);
                op.initialize_dof_vector(rhs);
                x   = 0.;
                rhs = 0.;
                op.rhs(rhs);

                ReductionControl reduction_control(dof_handler.n_dofs(),
                                                   1e-12,
                                                   1e-12);
                SolverCG<LinearAlgebra::distributed::Vector<Number>> solver(
                  reduction_control);
                PreconditionIdentity preconditioner;

                constraint.set_zero(x);
                solver.solve(op, x, rhs, preconditioner);
                constraint.distribute(x);

                pcout << "Solved in " << reduction_control.last_step()
                      << " iterations with final residual "
                      << std::setprecision(16) << reduction_control.last_value()
                      << std::endl;

                x.update_ghost_values();

                Vector<double> difference_per_cell;

                QGaussPyramid<dim> error_quad_pyramid(fe_degree + 3);
                QGaussWedge<dim>   error_quad_wedge(fe_degree + 3);
                QGaussSimplex<dim> error_quad_simplex(fe_degree + 3);
                QGauss<dim>        error_quad_hypercube(fe_degree + 3);

                const hp::QCollection<dim> error_quad(error_quad_pyramid,
                                                      error_quad_wedge,
                                                      error_quad_simplex,
                                                      error_quad_hypercube);

                VectorTools::integrate_difference(mapping,
                                                  dof_handler,
                                                  x,
                                                  Solution<dim>(),
                                                  difference_per_cell,
                                                  error_quad,
                                                  VectorTools::L2_norm);

                const double L2_error =
                  VectorTools::compute_global_error(tria,
                                                    difference_per_cell,
                                                    VectorTools::L2_norm);

                if (L2_error < 1e-10)
                  continue_iterating = false;

                const unsigned int n_active_cells =
                  tria.n_global_active_cells();
                n_dofs = dof_handler.n_dofs();

                if (use_equidistant_points)
                  pcout << "Cycle " << cycle << ':' << std::endl
                        << " mixed equidistant" << std::endl
                        << "   Number of active cells:       " << n_active_cells
                        << std::endl
                        << "   Number of degrees of freedom: " << n_dofs
                        << std::endl
                        << "   L2 error:                     " << L2_error
                        << std::endl;
                else
                  pcout << "Cycle " << cycle << ':' << std::endl
                        << " mixed blend and warp" << std::endl
                        << "   Number of active cells:       " << n_active_cells
                        << std::endl
                        << "   Number of degrees of freedom: " << n_dofs
                        << std::endl
                        << "   L2 error:                     " << L2_error
                        << std::endl;

                unsigned int offset = fe_degree - min_degree;
                if (!use_equidistant_points)
                  offset += max_degree - min_degree + 1;

                convergence_tables[offset].add_value("cycle", cycle);
                convergence_tables[offset].add_value("cells", n_active_cells);
                convergence_tables[offset].add_value("dofs", n_dofs);
                convergence_tables[offset].add_value("L2", L2_error);

                pcout << std::endl;
              }
          }
      pcout << std::endl;

      n_dofs  = 0;
      n_cells = tria.n_global_active_cells();
    }
  pcout << std::endl;
  pcout << std::endl;

  unsigned int degree_counter  = min_degree;
  bool         use_equi_points = true;
  if (solve)
    for (auto &convergence_table : convergence_tables)
      {
        convergence_table.set_precision("L2", 3);
        convergence_table.set_scientific("L2", true);

        convergence_table.set_tex_caption("cells", "\\# cells");
        convergence_table.set_tex_caption("dofs", "\\# dofs");
        convergence_table.set_tex_caption("L2", "$L^2$-error");

        convergence_table.set_tex_format("cells", "r");
        convergence_table.set_tex_format("dofs", "r");

        convergence_table.evaluate_convergence_rates(
          "L2", ConvergenceTable::reduction_rate);
        convergence_table.evaluate_convergence_rates(
          "L2", ConvergenceTable::reduction_rate_log2);

        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            convergence_table.write_text(std::cout);

            std::string error_filename = "error_CG_MF_";
            error_filename += "Mixed_p_" + std::to_string(degree_counter);
            if (use_equi_points)
              error_filename += "_equidistant";
            else
              error_filename += "_blend_and_warp";

            error_filename += ".tex";
            std::ofstream error_table_file(error_filename);

            convergence_table.write_tex(error_table_file);
          }
        pcout << std::endl;
        ++degree_counter;
        if (degree_counter > max_degree)
          {
            degree_counter  = min_degree;
            use_equi_points = false;
          }
      }
}


int main(int argc, char **argv)
{
  constexpr int dim = 3;

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

  do_test<dim, double>(min_degree, max_degree, n_cycles_max);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
