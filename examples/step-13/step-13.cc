
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix_ez.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

using namespace dealii;


// Duffy transform
template <int dim>
Point<dim> duffy_transform(const Point<dim> p)
{
  if constexpr (dim == 2)
    return Point<dim>(p[0] * (1.0 - p[1]), p[1]);

  DEAL_II_NOT_IMPLEMENTED();
  return Point<dim>();
}

template <int dim>
std::vector<Point<dim>>
adopted_fe_p_support_points(const FE_Q<dim>               &fe_q,
                            const FiniteElement<dim, dim> &fe_p)
{
  // build a basis of triangular support points, this gives the mapping of the
  // interior points
  std::vector<Point<dim>> fe_p_support_points(fe_p.n_dofs_per_cell(),
                                              Point<dim>());

  std::vector<bool> already_included_point(fe_q.n_dofs_per_cell(), false);

  const unsigned int first_interior_index_p =
    dim == 2 ? fe_p.get_first_quad_index() : fe_p.get_first_hex_index();
  const unsigned int first_interior_index_q =
    dim == 2 ? fe_q.get_first_quad_index() : fe_q.get_first_hex_index();

  for (unsigned int i = 0; i < first_interior_index_p; ++i)
    fe_p_support_points[i] = fe_p.unit_support_point(i);

  for (unsigned int i = first_interior_index_p; i < fe_p.n_dofs_per_cell(); ++i)
    {
      const Point<dim> &p = fe_p.unit_support_point(i);

      double       min_distance       = std::numeric_limits<double>::max();
      unsigned int min_distance_index = numbers::invalid_unsigned_int;
      // find the nearest quad point
      for (unsigned int j = first_interior_index_q; j < fe_q.n_dofs_per_cell();
           ++j)
        if (already_included_point[j] == false)
          {
            if (dim == 3)
              DEAL_II_NOT_IMPLEMENTED();

            const double distance =
              p.distance(duffy_transform(fe_q.unit_support_point(j)));
            if (distance < min_distance)
              {
                min_distance       = distance;
                min_distance_index = j;
              }
          }

      fe_p_support_points[i] =
        duffy_transform(fe_q.unit_support_point(min_distance_index));
      already_included_point[min_distance_index] = true;
    }

  return fe_p_support_points;
}

template <typename Number>
Number jacobi_polynomial_derivative(const unsigned int degree,
                                    const int          alpha,
                                    const int          beta,
                                    const Number       x)
{
  // Take x in invertal [0,1]
  Assert(alpha >= 0 && beta >= 0,
         ExcNotImplemented("Negative alpha/beta coefficients not supported"));

  // The derivative of the Jacobi polynomial is evaluated using the recurrence
  // relations
  if (degree == 0)
    return 0.0;

  return (1 + alpha + beta + degree) *
         dealii::Polynomials::jacobi_polynomial_value(degree - 1,
                                                      alpha + 1,
                                                      beta + 1,
                                                      x);
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


template <int dim, typename Number = double>
class QuadratureCollapsed
{
public:
  QuadratureCollapsed(const unsigned int n_points_1D)
  {
    std::vector<Number> points_x;

    if (dim == 2)
      {
        const dealii::QGauss<1> quad_x(n_points_1D);
        this->points_cube_x = quad_x.get_points();

        for (unsigned int i = 0; i < n_points_1D; ++i)
          {
            points_x.emplace_back(quad_x.get_points()[i][0]);
            this->weights_x.emplace_back(quad_x.get_weights()[i]);
          }

        // Gives back points in interval [0,1]
        const auto points_y =
          dealii::Polynomials::jacobi_polynomial_roots<Number>(n_points_1D,
                                                               1,
                                                               0);

        for (const auto &p : points_y)
          this->points_cube_y.emplace_back(p);

        for (unsigned int i = 0; i < n_points_1D; ++i)
          {
            const Number y = points_y[i];
            // here we need to rescale y to 2*y-1
            const Number factor = 4.0 / (1.0 - std::pow(2. * y - 1., 2));
            const Number deriv =
              jacobi_polynomial_derivative(n_points_1D, 1, 0, y);
            this->weights_y.emplace_back(factor / (std::pow(deriv, 2)));
          }

        for (unsigned int i = 0; i < n_points_1D; ++i)
          for (unsigned int j = 0; j < n_points_1D; ++j)
            {
              dealii::Point<dim> p(points_x[i], points_y[j]);
              this->points.emplace_back(p);
              this->weights.emplace_back(this->weights_x[i] *
                                         this->weights_y[j]);

              dealii::Point<dim> p_triangle(points_x[i] * (1. - points_y[j]),
                                            points_y[j]);
              this->points_triangle.emplace_back(p_triangle);
            }
      }

    else if (dim == 3)
      {
        const dealii::QGauss<1> quad_x(n_points_1D);
        this->points_cube_x = quad_x.get_points();

        for (unsigned int i = 0; i < n_points_1D; ++i)
          {
            points_x.emplace_back(quad_x.get_points()[i][0]);
            this->weights_x.emplace_back(quad_x.get_weights()[i]);
          }

        // Gives back points in interval [0,1]
        const auto points_y =
          dealii::Polynomials::jacobi_polynomial_roots<Number>(n_points_1D,
                                                               1,
                                                               0);
        const auto points_z =
          dealii::Polynomials::jacobi_polynomial_roots<Number>(n_points_1D,
                                                               2,
                                                               0);

        for (const auto &p : points_y)
          this->points_cube_y.emplace_back(p);
        for (const auto &p : points_z)
          this->points_cube_z.emplace_back(p);

        for (unsigned int i = 0; i < n_points_1D; ++i)
          {
            const Number y = points_y[i];
            // here we need to rescale y to 2*y-1
            const Number factor = 4.0 / (1.0 - std::pow(2. * y - 1., 2));
            const Number deriv =
              jacobi_polynomial_derivative(n_points_1D, 1, 0, y);
            this->weights_y.emplace_back(factor / (std::pow(deriv, 2)));

            const Number z = points_z[i];
            // here we need to rescale z to 2*z-1
            const Number factor_z =
              0.5 * 8.0 / (1.0 - std::pow(2. * z - 1., 2));
            const Number deriv_z =
              jacobi_polynomial_derivative(n_points_1D, 2, 0, z);
            this->weights_z.emplace_back(factor_z / (std::pow(deriv_z, 2)));
          }

        for (unsigned int i = 0; i < n_points_1D; ++i)
          for (unsigned int j = 0; j < n_points_1D; ++j)
            for (unsigned int k = 0; k < n_points_1D; ++k)
              {
                dealii::Point<dim> p(points_x[i], points_y[j], points_z[k]);
                this->points.emplace_back(p);
                this->weights.emplace_back(
                  this->weights_x[i] * this->weights_y[j] * this->weights_z[k]);

                dealii::Point<dim> p_tet(points_x[i] *
                                           (1. - points_y[j] - points_z[k]),
                                         points_y[j] * (1. - points_z[k]),
                                         points_z[k]);
                this->points_tet.emplace_back(p_tet);
              }
      }
  }

  std::vector<Number> weights;
  std::vector<Number> weights_x;
  std::vector<Number> weights_y;
  std::vector<Number> weights_z;

  std::vector<dealii::Point<dim>> points;
  std::vector<dealii::Point<dim>> points_triangle;
  std::vector<dealii::Point<dim>> points_tet;

  std::vector<dealii::Point<1>> points_cube_x;
  std::vector<dealii::Point<1>> points_cube_y;
  std::vector<dealii::Point<1>> points_cube_z;
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

  void reinit(const Mapping<dim>              &mapping,
              const DoFHandler<dim>           &dof_handler,
              const Quadrature<dim>           &quad,
              const AffineConstraints<number> &constraints,
              const unsigned int mg_level = numbers::invalid_unsigned_int,
              const bool         ones_on_diagonal = false)
  {
    this->constraints.copy_from(constraints);

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags = update_gradients;
    data.mg_level             = mg_level;

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
                << dof_handler.get_triangulation().n_global_active_cells()
                << " " << dof_handler.n_dofs() << " "
                << static_cast<double>(dof_handler.n_dofs()) /
                     dof_handler.get_triangulation().n_global_active_cells()
                << " " << std::endl;

    constrained_indices.clear();

    if (ones_on_diagonal)
      for (auto i : this->matrix_free.get_constrained_dofs())
        constrained_indices.push_back(i);

    constexpr unsigned int n_lanes = VectorizedArray<number>::size();

    const auto &fe_p = dof_handler.get_fe();
    this->degree     = fe_p.degree;

    const FE_Q<dim>                 fe_q(degree);
    const std::vector<unsigned int> lex_to_standard =
      fe_q.get_poly_space_numbering_inverse();

    const std::vector<unsigned int> &standard_to_lex =
      fe_q.get_poly_space_numbering();

    const unsigned int n_dofs_per_cell_q = fe_q.n_dofs_per_cell();
    const unsigned int n_dofs_per_cell_p = fe_p.n_dofs_per_cell();

    const unsigned int n_vertices_p = fe_p.reference_cell().n_vertices();
    const unsigned int n_vertices_q = fe_q.reference_cell().n_vertices();

    const unsigned int n_dofs_per_edge = degree - 1;

    const unsigned int first_interior_index_q =
      dim == 2 ? fe_q.get_first_quad_index() : fe_q.get_first_hex_index();

    // first build a look up table which gives the triangle index for each
    // quad index
    std::vector<unsigned int> quad_to_triangle_index(
      n_dofs_per_cell_q, numbers::invalid_unsigned_int);

    // get the "standard" ordering, translate to lexicographic
    // ordering later
    unsigned int dof_counter_p = 0;
    unsigned int dof_counter_q = 0;

    if constexpr (dim == 2)
      {
        // go over all vertices
        for (; dof_counter_p < n_vertices_p; ++dof_counter_p, ++dof_counter_q)
          quad_to_triangle_index[dof_counter_q] = dof_counter_p;

        // fill remaining q vertices with info from the last p vertex
        const unsigned int last_vertex_p = n_vertices_p - 1;

        for (; dof_counter_q < n_vertices_q; ++dof_counter_q)
          quad_to_triangle_index[dof_counter_q] = last_vertex_p;

        // edge 0 simplex is edge 2 hypercube
        for (unsigned int i = 0; i < n_dofs_per_edge;
             ++i, ++dof_counter_q, ++dof_counter_p)
          quad_to_triangle_index[dof_counter_q + 2 * n_dofs_per_edge] =
            dof_counter_p;

        // edge 1 simplex is edge 1 hypercube
        for (unsigned int i = 0; i < n_dofs_per_edge;
             ++i, ++dof_counter_q, ++dof_counter_p)
          quad_to_triangle_index[dof_counter_q] = dof_counter_p;

        // edge 2 simplex is edge 0 hypercube but in reverse direction
        for (unsigned int i = 0; i < n_dofs_per_edge;
             ++i, ++dof_counter_q, ++dof_counter_p)
          quad_to_triangle_index[n_vertices_q + n_dofs_per_edge - i - 1] =
            dof_counter_p;

        // collapsed edge, maps to the top vertex
        for (unsigned int i = 0; i < n_dofs_per_edge; ++i, ++dof_counter_q)
          quad_to_triangle_index[dof_counter_q] = last_vertex_p;

        AssertDimension(dof_counter_q, first_interior_index_q);
      }
    else
      {
        DEAL_II_NOT_IMPLEMENTED();
      }

    const unsigned int first_interior_index_p =
      dim == 2 ? fe_p.get_first_quad_index() : fe_p.get_first_hex_index();
    std::vector<bool> used_q_index(n_dofs_per_cell_q, false);
    // now do the interior DoFs
    for (unsigned int i = first_interior_index_p; i < n_dofs_per_cell_p; ++i)
      {
        const auto   p_tri       = fe_p.unit_support_point(i);
        unsigned int min_index   = numbers::invalid_unsigned_int;
        double       min_distane = std::numeric_limits<double>::max();

        for (unsigned int j = first_interior_index_q; j < n_dofs_per_cell_q;
             ++j)
          if (used_q_index[j] == false)
            {
              const auto p_transformed =
                duffy_transform(fe_q.unit_support_point(j));
              if (p_tri.distance(p_transformed) < min_distane)
                {
                  min_index   = j;
                  min_distane = p_tri.distance(p_transformed);
                }
            }
        quad_to_triangle_index[min_index] = i;
        used_q_index[min_index]           = true;
      }

    // check that there are a few non interpolated DoFs
    unsigned int n_invalid_entries = 0;
    for (const auto &i : quad_to_triangle_index)
      {
        if (i == numbers::invalid_unsigned_int)
          ++n_invalid_entries;
      }
    Assert(n_invalid_entries ==
             n_dofs_per_cell_q - n_dofs_per_cell_p - n_dofs_per_edge - 1,
           ExcInternalError());

    // check that all simplex indices appear
    std::vector<unsigned int> simplex_usage_count(n_dofs_per_cell_p, 0);
    for (const unsigned int p_index : quad_to_triangle_index)
      if (p_index != numbers::invalid_unsigned_int)
        ++simplex_usage_count[p_index];
    Assert(std::all_of(simplex_usage_count.begin(),
                       simplex_usage_count.end(),
                       [](const unsigned int count) { return count > 0; }),
           ExcMessage(
             "The quadrilateral to triangle DoF lookup does not reference "
             "every simplex DoF."));

    // we also need the other direction, but this is not unique
    std::vector<unsigned int> triangle_to_quad_index(
      n_dofs_per_cell_p, numbers::invalid_unsigned_int);
    for (unsigned int i = 0; i < triangle_to_quad_index.size(); ++i)
      {
        bool not_found_entry = true;
        for (unsigned int j = 0; j < n_dofs_per_cell_q && not_found_entry; ++j)
          if (quad_to_triangle_index[j] == i)
            {
              triangle_to_quad_index[i] = j;
              not_found_entry           = false;
            }
      }
    Assert(std::none_of(triangle_to_quad_index.begin(),
                        triangle_to_quad_index.end(),
                        [](const unsigned int index) {
                          return index == numbers::invalid_unsigned_int;
                        }),
           ExcMessage(
             "The triangle to quadrilateral DoF lookup is incomplete."));

    // now get the interpolation for the interior dofs
    // do it first to identify further one to one relations
    const unsigned int n_interior_nodes_q =
      dim == 2 ? fe_q.n_dofs_per_quad() : fe_q.n_dofs_per_hex();

    const unsigned int n_interior_nodes_p =
      dim == 2 ? fe_p.n_dofs_per_quad() : fe_p.n_dofs_per_hex();

    // lexiographic indices of all interior nodes
    interior_indices_q.resize(n_interior_nodes_q - n_interior_nodes_p,
                              numbers::invalid_unsigned_int);

    // indices and interpolation factors for all lexiogrpahic numbered interior
    // nodes
    interpolation_factors_for_indices.resize(n_interior_nodes_q -
                                             n_interior_nodes_p);

    // go over all interior nodes in "standard" ordering
    for (unsigned int interior_node_counter = 0, i = 0; i < n_interior_nodes_q;
         ++i)
      {
        // get standard node index
        const unsigned int q_index = first_interior_index_q + i;

        // only need to handle nodes we did not handle before
        if (quad_to_triangle_index[q_index] == numbers::invalid_unsigned_int)
          {
            // get lexiographic index
            const unsigned int q_index_lexiographic = standard_to_lex[q_index];

            // get point on tri
            const Point<dim> point_q_on_tri =
              duffy_transform(fe_q.unit_support_point(q_index));

            // now evaluate all shape functions at the point
            for (unsigned int j = 0; j < n_dofs_per_cell_p; ++j)
              {
                const Number interpolation_factor =
                  fe_p.shape_value(j, point_q_on_tri);

                if (std::abs(interpolation_factor) > 1e-12)
                  {
                    Assert(interior_indices_q[interior_node_counter] ==
                               numbers::invalid_unsigned_int ||
                             interior_indices_q[interior_node_counter] ==
                               q_index_lexiographic,
                           ExcInternalError());

                    interior_indices_q[interior_node_counter] =
                      q_index_lexiographic;

                    // quad index for the current shape function
                    unsigned int shape_function_index_on_q =
                      triangle_to_quad_index[j];

                    interpolation_factors_for_indices[interior_node_counter]
                      .emplace_back(standard_to_lex[shape_function_index_on_q],
                                    interpolation_factor);
                  }
              }
            ++interior_node_counter;
          }
      }

    for (unsigned int i = 0; i < n_dofs_per_cell_q; ++i)
      {
        // check that all entries are either in quad_to_triangle or in
        // interior_indices
        bool is_in_interior = false;
        for (const auto idx : interior_indices_q)
          if (lex_to_standard[idx] == i)
            is_in_interior = true;

        Assert(is_in_interior ||
                 quad_to_triangle_index[i] != numbers::invalid_unsigned_int,
               ExcInternalError());
      }

    // TODO: all elements to be deleted stuff can be removed
    std::vector<unsigned int> elements_to_be_delted;
    for (unsigned int i = 0; i < interpolation_factors_for_indices.size(); ++i)
      {
        Assert(interpolation_factors_for_indices[i].empty() == false,
               ExcNotImplemented());

        // check for one to one mappings, we can skip the interpolation
        if (interpolation_factors_for_indices[i].size() == 1)
          {
            // interior index in "standard" numbering
            const unsigned int interior_index_q =
              lex_to_standard[interior_indices_q[i]];

            // index from which is interpolated in standard numbering
            const unsigned int index_for_interpolation =
              lex_to_standard[interpolation_factors_for_indices[i][0].first];

            quad_to_triangle_index[interior_index_q] =
              quad_to_triangle_index[index_for_interpolation];

            // mark entries for deletion
            elements_to_be_delted.push_back(i);
          }
      }

    Assert(elements_to_be_delted.empty(), ExcInternalError());
    for (unsigned int i = 0; i < elements_to_be_delted.size(); ++i)
      {
        // revese order of deletion
        const unsigned int element_to_be_delted =
          elements_to_be_delted[elements_to_be_delted.size() - 1 - i];

        interpolation_factors_for_indices.erase(
          interpolation_factors_for_indices.begin() + element_to_be_delted);

        interior_indices_q.erase(interior_indices_q.begin() +
                                 element_to_be_delted);
      }

    // now get the new dof indices
    manual_dof_indices.reinit(matrix_free.n_cell_batches(),
                              n_dofs_per_cell_q * n_lanes,
                              true);
    manual_dof_indices.fill(numbers::invalid_unsigned_int);
    std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell_p);

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
            // get simplex indices
            matrix_free.get_cell_iterator(c, v)->get_dof_indices(dof_indices);

            // go over all dofs in lexiographic ordering
            for (unsigned int i = 0; i < n_dofs_per_cell_q; ++i)
              {
                // corresponding index on tri. need to convert from
                // lexiographice to "standard" ordering
                const unsigned int fe_p_index =
                  quad_to_triangle_index[lex_to_standard[i]];

                if (fe_p_index != numbers::invalid_unsigned_int)
                  {
                    // global dof index
                    const types::global_dof_index dof_index =
                      dof_indices[fe_p_index];

                    if (constraints.is_constrained(dof_index))
                      has_constraints = true;
                    else
                      manual_dof_indices(c, i * n_lanes + v) =
                        matrix_free.get_dof_info()
                          .vector_partitioner->global_to_local(dof_index);
                  }
                else
                  {
                    has_constraints = true;
                  }
              }
          }
        dof_indices_have_constraints[c] = has_constraints;
      }

    // Set up collapsed data
    this->n_DoF_1D = degree + 1;
    const QuadratureCollapsed<dim>  quad_collapsed(degree + 1);
    const FE_Q<1>                   fe_1D = FE_Q<1>(degree);
    const std::vector<unsigned int> lexiographic_numbering_1D =
      fe_1D.get_poly_space_numbering_inverse();

    for (unsigned int q = 0; q < quad_collapsed.weights_x.size(); ++q)
      for (unsigned int dof_idx = 0; dof_idx < n_DoF_1D; ++dof_idx)
        {
          const unsigned int dof_idx_lexio = lexiographic_numbering_1D[dof_idx];

          this->S_x.emplace_back(
            fe_1D.shape_value(dof_idx_lexio, quad_collapsed.points_cube_x[q]));
          this->S_y.emplace_back(
            fe_1D.shape_value(dof_idx_lexio, quad_collapsed.points_cube_y[q]));
          if (dim == 3)
            this->S_z.emplace_back(
              fe_1D.shape_value(dof_idx_lexio,
                                quad_collapsed.points_cube_z[q]));

          this->D_x.emplace_back(
            fe_1D.shape_grad(dof_idx_lexio,
                             quad_collapsed.points_cube_x[q])[0]);
          this->D_y.emplace_back(
            fe_1D.shape_grad(dof_idx_lexio,
                             quad_collapsed.points_cube_y[q])[0]);
          if (dim == 3)
            this->D_z.emplace_back(
              fe_1D.shape_grad(dof_idx_lexio,
                               quad_collapsed.points_cube_z[q])[0]);
        }

    this->n_q_1D            = quad_collapsed.weights_x.size();
    this->quadrature_points = quad_collapsed.points;
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

  virtual void vmult_collapsed(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_collapsed, this, dst, src, true);

    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
  }

  void Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
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

  void do_cell_integral_collapsed(
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
    const unsigned int fe_degree     = this->degree;
    const unsigned int n_dofs_per_cell_q = std::pow(fe_degree + 1, dim);
    constexpr unsigned int batch_size    = 1;
    const unsigned int     n_q_points    = shape_info.n_q_points;
    constexpr unsigned int n_lanes       = VectorizedArray<number>::size();

    const auto   &mapping_data = matrix_free.get_mapping_info().cell_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();

    const unsigned int size_imideate_arrays =
      dim == 2 ? n_q_1D * n_DoF_1D : 0; // TODO
    scratch_data->resize_fast(
      batch_size *
      (n_dofs_per_cell_q + size_imideate_arrays + dim * std::pow(n_q_1D, dim)));
    VectorizedArray<number> *values_dofs = scratch_data->begin();
    VectorizedArray<number> *temp_x =
      scratch_data->begin() + batch_size * n_dofs_per_cell_q;
    VectorizedArray<number> *gradients_quad = scratch_data->begin() +
                                              batch_size * n_dofs_per_cell_q +
                                              batch_size * size_imideate_arrays;

    const number *src_ptr = src.begin();

    for (unsigned int cell = range.first; cell < range.second;
         cell += batch_size)
      {
        // read dof values
        const unsigned int my_batch_size =
          cell + batch_size <= range.second ? batch_size : range.second - cell;
        const unsigned int *dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < n_dofs_per_cell_q;
                     ++i, dof_indices += n_lanes)
                  {
                    values_dofs[batch * n_dofs_per_cell_q + i] = {};
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        values_dofs[batch * n_dofs_per_cell_q + i][v] =
                          src_ptr[dof_indices[v]];
                  }
              }
            else
              for (unsigned int i = 0; i < n_dofs_per_cell_q;
                   ++i, dof_indices += n_lanes)
                {
                  values_dofs[batch * n_dofs_per_cell_q + i] = {};
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    values_dofs[batch * n_dofs_per_cell_q + i][v] =
                      src_ptr[dof_indices[v]];
                }
          }

        // after reading the dof values interpolate the interior dofs
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          for (unsigned int i = 0; i < interior_indices_q.size(); ++i)
            {
              const unsigned int interior_index = interior_indices_q[i];
              const std::vector<std::pair<unsigned int, number>>
                &interpolation_index_factors =
                  interpolation_factors_for_indices[i];

              values_dofs[batch * n_dofs_per_cell_q + interior_index] = {};

              for (const auto &[index, factor] : interpolation_index_factors)
                {
                  values_dofs[batch * n_dofs_per_cell_q + interior_index] +=
                    factor * values_dofs[batch * n_dofs_per_cell_q + index];
                }
            }

        if constexpr (dim == 2)
          {
            // interpolate
            for (unsigned int j = 0; j < n_DoF_1D; ++j)
              dealii::internal::apply_matrix_vector_product<
                dealii::internal::EvaluatorVariant::evaluate_general,
                dealii::internal::EvaluatorQuantity::value,
                /*transpose_matrix*/ false,
                /*add*/ false,
                /*consider_strides*/ false>(D_x.data(),
                                            values_dofs + j * n_DoF_1D,
                                            temp_x + j * n_q_1D,
                                            n_q_1D,
                                            n_DoF_1D,
                                            1,
                                            1);

            // interpolate
            for (unsigned int s = 0; s < n_q_1D; ++s)
              dealii::internal::apply_matrix_vector_product<
                dealii::internal::EvaluatorVariant::evaluate_general,
                dealii::internal::EvaluatorQuantity::value,
                /*transpose_matrix*/ false,
                /*add*/ false,
                /*consider_strides*/ true>(S_y.data(),
                                           temp_x + s,
                                           gradients_quad + 2 * s * n_q_1D,
                                           n_q_1D,
                                           n_DoF_1D,
                                           n_q_1D,
                                           2);

            // interpolate
            for (unsigned int j = 0; j < n_DoF_1D; ++j)
              dealii::internal::apply_matrix_vector_product<
                dealii::internal::EvaluatorVariant::evaluate_general,
                dealii::internal::EvaluatorQuantity::value,
                /*transpose_matrix*/ false,
                /*add*/ false,
                /*consider_strides*/ false>(S_x.data(),
                                            values_dofs + j * n_DoF_1D,
                                            temp_x + j * n_q_1D,
                                            n_q_1D,
                                            n_DoF_1D,
                                            1,
                                            1);


            // interpolate
            for (unsigned int s = 0; s < n_q_1D; ++s)
              dealii::internal::apply_matrix_vector_product<
                dealii::internal::EvaluatorVariant::evaluate_general,
                dealii::internal::EvaluatorQuantity::value,
                /*transpose_matrix*/ false,
                /*add*/ false,
                /*consider_strides*/ true>(D_y.data(),
                                           temp_x + s,
                                           gradients_quad + 1 + 2 * s * n_q_1D,
                                           n_q_1D,
                                           n_DoF_1D,
                                           n_q_1D,
                                           2);
          }
        else
          {
            DEAL_II_NOT_IMPLEMENTED();
            // evaluate
            apply_matrix_vector_product<true>(
              shape_info.data[0].shape_gradients.data(),
              values_dofs,
              gradients_quad,
              dofs_per_cell,
              n_q_points * dim);
          }

        // quadrature point operation
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int offsets =
              mapping_data.data_index_offsets[cell + batch];
            const Tensor<2, dim, VectorizedArray<number>> *jac =
              mapping_data.jacobians[0].data() + offsets;
            const VectorizedArray<number> j_value =
              mapping_data.JxW_values[offsets];
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
                    const number nu1 = quadrature_points[q][0];
                    const number nu2 = quadrature_points[q][1];

                    Tensor<1, dim, VectorizedArray<number>> grad;
                    grad[0] = 1. / (1. - nu2) * grad_ptr[0];
                    grad[1] = nu1 / (1. - nu2) * grad_ptr[0] + grad_ptr[1];

                    Tensor<1, dim, VectorizedArray<number>> result =
                      my_metric * grad;
                    const number weight = quadrature_weights[q];

                    Tensor<1, dim, VectorizedArray<number>> result2;
                    for (unsigned int d = 0; d < dim; ++d)
                      result2[d] = weight * result[d];

                    grad_ptr[0] = 1. / (1. - nu2) * result2[0] +
                                  nu1 / (1. - nu2) * result2[1];
                    grad_ptr[1] = result2[1];
                  }
              }
            else
              {
                for (unsigned int q = 0; q < n_q_points; ++q, grad_ptr += dim)
                  {
                    const number nu1 = quadrature_points[q][0];
                    const number nu2 = quadrature_points[q][1];
                    Tensor<1, dim, VectorizedArray<number>> grad;
                    // for (unsigned int d = 0; d < dim; ++d)
                    //   grad[d] = grad_ptr[d];
                    grad[0] = 1. / (1. - nu2) * grad_ptr[0];
                    grad[1] = nu1 / (1. - nu2) * grad_ptr[0] + grad_ptr[1];

                    Tensor<1, dim, VectorizedArray<number>> result =
                      j_value[q] * (transpose(jac[q]) * (jac[q] * grad));
                    // for (unsigned int d = 0; d < dim; ++d)
                    //   grad_ptr[d] = result[d];
                    grad_ptr[0] = 1. / (1. - nu2) * result[0] +
                                  nu1 / (1. - nu2) * result[1];
                    grad_ptr[1] = result[1];
                  }
              }
          }


        if constexpr (dim == 2)
          {
            for (unsigned int s = 0; s < n_q_1D; ++s)
              dealii::internal::apply_matrix_vector_product<
                dealii::internal::EvaluatorVariant::evaluate_general,
                dealii::internal::EvaluatorQuantity::value,
                /*transpose_matrix*/ true,
                /*add*/ false,
                /*consider_strides*/ true>(D_x.data(),
                                           gradients_quad + 2 * s,
                                           temp_x + s * n_DoF_1D,
                                           n_q_1D,
                                           n_DoF_1D,
                                           2 * n_q_1D,
                                           1);


            for (unsigned int i = 0; i < n_DoF_1D; ++i)
              dealii::internal::apply_matrix_vector_product<
                dealii::internal::EvaluatorVariant::evaluate_general,
                dealii::internal::EvaluatorQuantity::value,
                /*transpose_matrix*/ true,
                /*add*/ false,
                /*consider_strides*/ true>(S_y.data(),
                                           temp_x + i,
                                           values_dofs + i,
                                           n_q_1D,
                                           n_DoF_1D,
                                           n_DoF_1D,
                                           n_DoF_1D);


            for (unsigned int s = 0; s < n_q_1D; ++s)
              dealii::internal::apply_matrix_vector_product<
                dealii::internal::EvaluatorVariant::evaluate_general,
                dealii::internal::EvaluatorQuantity::value,
                /*transpose_matrix*/ true,
                /*add*/ false,
                /*consider_strides*/ true>(S_x.data(),
                                           gradients_quad + 2 * s + 1,
                                           temp_x + s * n_DoF_1D,
                                           n_q_1D,
                                           n_DoF_1D,
                                           2 * n_q_1D,
                                           1);

            for (unsigned int i = 0; i < n_DoF_1D; ++i)
              dealii::internal::apply_matrix_vector_product<
                dealii::internal::EvaluatorVariant::evaluate_general,
                dealii::internal::EvaluatorQuantity::value,
                /*transpose_matrix*/ true,
                /*add*/ true,
                /*consider_strides*/ true>(D_y.data(),
                                           temp_x + i,
                                           values_dofs + i,
                                           n_q_1D,
                                           n_DoF_1D,
                                           n_DoF_1D,
                                           n_DoF_1D);
          }
        else
          {
            DEAL_II_NOT_IMPLEMENTED();
            // integrate
            apply_matrix_vector_product<false>(
              shape_info.data[0].shape_gradients.data(),
              gradients_quad,
              values_dofs,
              dofs_per_cell,
              n_q_points * dim);
          }

        // distribute interior dofs
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          for (unsigned int i = 0; i < interior_indices_q.size(); ++i)
            {
              const unsigned int interior_index = interior_indices_q[i];
              const std::vector<std::pair<unsigned int, number>>
                &interpolation_index_factors =
                  interpolation_factors_for_indices[i];

              for (const auto &[index, factor] : interpolation_index_factors)
                {
                  values_dofs[batch * n_dofs_per_cell_q + index] +=
                    factor *
                    values_dofs[batch * n_dofs_per_cell_q + interior_index];
                }
            }
        // for (unsigned int batch = 0; batch < my_batch_size; ++batch)
        //   for (const unsigned int d : interior_indices_q)
        //     values_dofs[batch * n_dofs_per_cell_q + d] = {};

        // distribute local to global
        dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < n_dofs_per_cell_q;
                     ++i, dof_indices += n_lanes)
                  {
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        dst.local_element(dof_indices[v]) +=
                          values_dofs[batch * n_dofs_per_cell_q + i][v];
                  }
              }
            else
              for (unsigned int i = 0; i < n_dofs_per_cell_q;
                   ++i, dof_indices += n_lanes)
                {
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    dst.local_element(dof_indices[v]) +=
                      values_dofs[batch * n_dofs_per_cell_q + i][v];
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

  std::vector<number> S_x;
  std::vector<number> S_y;
  std::vector<number> S_z;

  std::vector<number> D_x;
  std::vector<number> D_y;
  std::vector<number> D_z;


  std::vector<Point<dim>> quadrature_points;

  unsigned int degree;
  unsigned int n_DoF_1D;
  unsigned int n_q_1D;

  std::vector<unsigned int> interior_indices_q;
  std::vector<std::vector<std::pair<unsigned int, number>>>
    interpolation_factors_for_indices;
};

template <int dim, typename Number>
void do_test(const unsigned int fe_degree, const unsigned int refine_max)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  pcout << "Running in " << dim << "D with degree " << fe_degree << std::endl;

  const std::vector<Point<dim>> fe_p_support_points =
    adopted_fe_p_support_points<dim>(FE_Q<dim>(fe_degree),
                                     FE_SimplexP<dim>(fe_degree, false));
  FE_SimplexP<dim>    fe(fe_degree, fe_p_support_points);
  MappingFE<dim>      mapping(FE_SimplexP<dim>(1, false));
  QGaussSimplex<dim>  quad_simplex(fe_degree + 1);
  QStroudSimplex<dim> quad(fe_degree + 1);


  AffineConstraints<double>                      constraint;
  parallel::fullydistributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  for (unsigned int refinements = 1; refinements < refine_max; ++refinements)
    {
      const auto serial_grid_generator =
        [&refinements](dealii::Triangulation<dim, dim> &tria_serial) {
          // set up triangulation
          GridGenerator::subdivided_hyper_cube_with_simplices(tria_serial, 2);
          if (refinements > 0)
            tria_serial.refine_global(refinements);
        };
      const auto serial_grid_partitioner =
        [&](dealii::Triangulation<dim, dim> &tria_serial,
            const MPI_Comm                   comm,
            const unsigned int) {
          dealii::GridTools::partition_triangulation_zorder(
            dealii::Utilities::MPI::n_mpi_processes(comm), tria_serial);
        };

      const unsigned int group_size = 40;

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

      tria.clear();
      tria.create_triangulation(description);

      DoFHandler<dim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);

      // set up constraints, then renumber dofs, and set up constraints again
      if (dim == 3)
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

      Operator<dim, 1, Number> op;
      // set up operator
      op.reinit(mapping,
                dof_handler,
                quad,
                constraint,
                numbers::invalid_unsigned_int,
                false);
      Operator<dim, 1, Number> op2;
      // set up operator
      op2.reinit(mapping,
                 dof_handler,
                 quad_simplex,
                 constraint,
                 numbers::invalid_unsigned_int,
                 false);

      LinearAlgebra::distributed::Vector<Number> vec1, vec2, vec3;
      op.initialize_dof_vector(vec1);
      op.initialize_dof_vector(vec2);
      op.initialize_dof_vector(vec3);
      for (Number &a : vec1)
        a = static_cast<double>(rand()) / RAND_MAX;

      for (unsigned int r = 0; r < 5; ++r)
        {
          Timer time;
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("matvec_p" + std::to_string(fe_degree) + "_s" +
                               std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          for (unsigned int t = 0; t < 100; ++t)
            op2.vmult(vec2, vec1);
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(("matvec_p" + std::to_string(fe_degree) + "_s" +
                              std::to_string(dof_handler.n_dofs()))
                               .c_str());
#endif
          const double run_time = time.wall_time();
          pcout << "n_dofs mf basic  " << dof_handler.n_dofs() << "  time "
                << run_time / 100 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
        }
      for (unsigned int r = 0; r < 5; ++r)
        {
          Timer time;
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("matvec_gather_p" + std::to_string(fe_degree) +
                               "_s" + std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          for (unsigned int t = 0; t < 100; ++t)
            op.vmult_collapsed(vec3, vec1);
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(("matvec_gather_p" + std::to_string(fe_degree) +
                              "_s" + std::to_string(dof_handler.n_dofs()))
                               .c_str());
#endif
          const double run_time = time.wall_time();
          pcout << "n_dofs mf gather " << dof_handler.n_dofs() << "  time "
                << run_time / 100 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
        }
      pcout << std::endl;

      vec3 -= vec2;
      pcout << "   Error MF variants: " << vec3.l2_norm() / vec2.l2_norm()
            << std::endl;
    }
}

int main(int argc, char **argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  int degree     = 2;
  int dim        = 3;
  int refine_max = 4;
  if (argc > 1)
    dim = std::atoi(argv[1]);
  if (argc > 2)
    degree = std::atoi(argv[2]);
  if (argc > 3)
    refine_max = std::atoi(argv[3]);

  if (dim == 2)
    do_test<2, double>(degree, refine_max);
  else
    do_test<3, double>(degree, refine_max);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
