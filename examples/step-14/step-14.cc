
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
Point<dim> duffy_transform_quad_to_tri(const Point<dim> p)
{
  if constexpr (dim == 1)
    return p;
  else if constexpr (dim == 2)
    return Point<dim>(p[0] * (1.0 - p[1]), p[1]);
  else if constexpr (dim == 3)
    return Point<dim>(p[0] * (1.0 - p[1]) * (1.0 - p[2]),
                      p[1] * (1.0 - p[2]),
                      p[2]);

  DEAL_II_NOT_IMPLEMENTED();
  return Point<dim>();
}


template <int dim>
Point<dim> duffy_transform_tri_to_quad(const Point<dim> p)
{
  if constexpr (dim == 1)
    return p;
  else if constexpr (dim == 2)
    return Point<dim>(p[0] / (1.0 - p[1]), p[1]);
  else if constexpr (dim == 3)
    return Point<dim>(p[0] / (1.0 - p[1] - p[2]), p[1] / (1.0 - p[2]), p[2]);

  DEAL_II_NOT_IMPLEMENTED();
  return Point<dim>();
}

template <int dim, int degree>
constexpr int n_dof_simplex()
{
  if constexpr (dim == 1)
    {
      return degree + 1;
    }
  else if constexpr (dim == 2)
    {
      return (degree + 1) * (degree + 2) / 2;
    }
  else if constexpr (dim == 3)
    {
      return (degree + 1) * (degree + 2) * (degree + 3) / 6;
    }

  DEAL_II_NOT_IMPLEMENTED();
  return numbers::invalid_unsigned_int;
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

template <int dim_,
          int fe_degree,
          int n_q_1D,
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

  std::vector<unsigned int>
  create_hypercube_to_simplex_index_map(const FiniteElement<dim> &fe_p,
                                        const FE_Q<dim>          &fe_q)
  {
    const unsigned int n_dofs_per_cell_q = fe_q.n_dofs_per_cell();
    const unsigned int n_dofs_per_cell_p = fe_p.n_dofs_per_cell();

    const unsigned int n_vertices_p = fe_p.reference_cell().n_vertices();
    const unsigned int n_vertices_q = fe_q.reference_cell().n_vertices();

    const unsigned int n_dofs_per_edge = fe_degree - 1;

    const unsigned int n_dofs_per_face_q = n_dofs_per_edge * n_dofs_per_edge;
    const unsigned int n_dofs_per_face_p =
      n_dof_simplex<dim - 1, fe_degree - 3>();

    const unsigned int first_quad_index_p = fe_p.get_first_quad_index();
    const unsigned int first_quad_index_q = fe_q.get_first_quad_index();

    // first build a look up table which gives the triangle index for each
    // quad index
    std::vector<unsigned int> hypercube_to_simplex_index(
      n_dofs_per_cell_q, numbers::invalid_unsigned_int);

    // get the "standard" ordering, translate to lexicographic
    // ordering later
    unsigned int dof_counter_p = 0;
    unsigned int dof_counter_q = 0;

    if constexpr (dim == 2)
      {
        // go over all vertices
        for (; dof_counter_p < n_vertices_p; ++dof_counter_p, ++dof_counter_q)
          hypercube_to_simplex_index[dof_counter_q] = dof_counter_p;

        // fill remaining q vertices with info from the last p vertex
        const unsigned int last_vertex_p = n_vertices_p - 1;

        for (; dof_counter_q < n_vertices_q; ++dof_counter_q)
          hypercube_to_simplex_index[dof_counter_q] = last_vertex_p;

        // edge 0 simplex is edge 2 hypercube
        for (unsigned int i = 0; i < n_dofs_per_edge;
             ++i, ++dof_counter_q, ++dof_counter_p)
          hypercube_to_simplex_index[dof_counter_q + 2 * n_dofs_per_edge] =
            dof_counter_p;

        // edge 1 simplex is edge 1 hypercube
        for (unsigned int i = 0; i < n_dofs_per_edge;
             ++i, ++dof_counter_q, ++dof_counter_p)
          hypercube_to_simplex_index[dof_counter_q] = dof_counter_p;

        // edge 2 simplex is edge 0 hypercube but in reverse direction
        for (unsigned int i = 0; i < n_dofs_per_edge;
             ++i, ++dof_counter_q, ++dof_counter_p)
          hypercube_to_simplex_index[n_vertices_q + n_dofs_per_edge - i - 1] =
            dof_counter_p;

        // collapsed edge, maps to the top vertex
        for (unsigned int i = 0; i < n_dofs_per_edge; ++i, ++dof_counter_q)
          hypercube_to_simplex_index[dof_counter_q] = last_vertex_p;

        AssertDimension(dof_counter_q, first_quad_index_q);
      }
    else if constexpr (dim == 3)
      {
        // vertices
        hypercube_to_simplex_index[0] = 0;
        hypercube_to_simplex_index[1] = 1;
        hypercube_to_simplex_index[2] = 2;
        hypercube_to_simplex_index[3] = 2;
        hypercube_to_simplex_index[4] = 3;
        hypercube_to_simplex_index[5] = 3;
        hypercube_to_simplex_index[6] = 3;
        hypercube_to_simplex_index[7] = 3;

        dof_counter_q = 8;
        dof_counter_p = 4;

        // edge 0 simplex is edge 2 hypercube
        for (unsigned int i = 0; i < n_dofs_per_edge;
             ++i, ++dof_counter_q, ++dof_counter_p)
          hypercube_to_simplex_index[dof_counter_q + 2 * n_dofs_per_edge] =
            dof_counter_p;

        // edge 1 simplex is edge 1 hypercube
        for (unsigned int i = 0; i < n_dofs_per_edge;
             ++i, ++dof_counter_q, ++dof_counter_p)
          hypercube_to_simplex_index[dof_counter_q] = dof_counter_p;

        // edge 2 simplex is edge 0 hypercube but in reverse direction
        for (unsigned int i = 0; i < n_dofs_per_edge;
             ++i, ++dof_counter_q, ++dof_counter_p)
          hypercube_to_simplex_index[n_vertices_q + n_dofs_per_edge - i - 1] =
            dof_counter_p;

        // edge 3 simplex is edge 8 hypercube
        for (unsigned int i = 0; i < n_dofs_per_edge;
             ++i, ++dof_counter_q, ++dof_counter_p)
          hypercube_to_simplex_index[5 * n_dofs_per_edge + dof_counter_q] =
            dof_counter_p;

        // edge 4 simplex is edge 9 hypercube
        for (unsigned int i = 0; i < n_dofs_per_edge;
             ++i, ++dof_counter_q, ++dof_counter_p)
          hypercube_to_simplex_index[5 * n_dofs_per_edge + dof_counter_q] =
            dof_counter_p;

        // edge 5 simplex is edge 10 hypercube
        for (unsigned int i = 0; i < n_dofs_per_edge;
             ++i, ++dof_counter_q, ++dof_counter_p)
          hypercube_to_simplex_index[5 * n_dofs_per_edge + dof_counter_q] =
            dof_counter_p;

        // collapsed edge 3 simplex is vertex 2
        for (unsigned int i = 0; i < n_dofs_per_edge; ++i, ++dof_counter_q)
          hypercube_to_simplex_index[n_vertices_q + 3 * n_dofs_per_edge + i] =
            2;

        // collapsed edge 4 - 7 simplex is vertex 3
        for (unsigned int i = 0; i < 4 * n_dofs_per_edge; ++i, ++dof_counter_q)
          hypercube_to_simplex_index[n_vertices_q + 4 * n_dofs_per_edge + i] =
            3;

        // collapsed edge 11 also maps to simplex edge 5 which is also edge 10
        for (unsigned int i = 0; i < n_dofs_per_edge; ++i, ++dof_counter_q)
          hypercube_to_simplex_index[dof_counter_q] =
            hypercube_to_simplex_index[dof_counter_q - n_dofs_per_edge];

        AssertDimension(dof_counter_q, first_quad_index_q);
        AssertDimension(dof_counter_p, first_quad_index_p);

        // also handle the last face as it also collapses to simplex vertex 3
        for (unsigned int i = 0; i < n_dofs_per_face_q; ++i)
          hypercube_to_simplex_index[first_quad_index_q +
                                     5 * n_dofs_per_face_q + i] = 3;
      }

    // now do the face and interior dofs
    // first construct one to one correspondence between
    // face and interior dofs
    // then interpolate the rest
    if constexpr (dim == 2)
      {
        std::vector<bool> used_q_index(n_dofs_per_cell_q, false);

        for (unsigned int i = first_quad_index_p; i < n_dofs_per_cell_p; ++i)
          {
            const auto   p_simplex   = fe_p.unit_support_point(i);
            unsigned int min_index   = numbers::invalid_unsigned_int;
            double       min_distane = std::numeric_limits<double>::max();

            for (unsigned int j = first_quad_index_q; j < n_dofs_per_cell_q;
                 ++j)
              if (used_q_index[j] == false)
                {
                  const auto p_transformed =
                    duffy_transform_quad_to_tri(fe_q.unit_support_point(j));
                  if (p_simplex.distance(p_transformed) < min_distane)
                    {
                      min_index   = j;
                      min_distane = p_simplex.distance(p_transformed);
                    }
                }
            if (min_distane < 1e-12)
              {
                hypercube_to_simplex_index[min_index] = i;
                used_q_index[min_index]               = true;
              }
          }
      }
    else if constexpr (dim == 3)
      {
        const unsigned int first_interior_index_p = fe_p.get_first_hex_index();
        const unsigned int first_interior_index_q = fe_q.get_first_hex_index();

        // go over all hex faces, face number 5 is already done as it just
        // collapses to the top vertex
        for (unsigned int f = 0; f < 5; ++f)
          {
            // there are a maximum of n_dofs_per_face_p which can be mapped
            // uniquely, only on face 3 which collapses to an edge all DoFs can
            // be mapped
            const unsigned int n_entities_on_face =
              f == 3 ? n_dofs_per_face_q : n_dofs_per_face_p;
            for (unsigned int i = first_quad_index_q + f * n_dofs_per_face_q,
                              found_identities_in_face = 0;
                 i < first_quad_index_q + (f + 1) * n_dofs_per_face_q &&
                 found_identities_in_face < n_entities_on_face;
                 ++i)
              {
                // only do the ones not done yet
                if (hypercube_to_simplex_index[i] ==
                    numbers::invalid_unsigned_int)
                  {
                    const auto p_transformed =
                      duffy_transform_quad_to_tri(fe_q.unit_support_point(i));
                    unsigned int min_index = numbers::invalid_unsigned_int;

                    unsigned int search_index_start = first_quad_index_p;
                    unsigned int search_index_end   = first_interior_index_p;
                    if (f == 0)
                      {
                        // simplex face 2
                        search_index_start =
                          first_quad_index_p + 2 * n_dofs_per_face_p;
                        search_index_end =
                          first_quad_index_p + 3 * n_dofs_per_face_p;
                      }
                    else if (f == 1)
                      {
                        // simplex face 3
                        search_index_start =
                          first_quad_index_p + 3 * n_dofs_per_face_p;
                        search_index_end =
                          first_quad_index_p + 4 * n_dofs_per_face_p;
                      }
                    else if (f == 2)
                      {
                        // simplex face 1
                        search_index_start =
                          first_quad_index_p + 1 * n_dofs_per_face_p;
                        search_index_end =
                          first_quad_index_p + 2 * n_dofs_per_face_p;
                      }
                    else if (f == 3)
                      {
                        // collapses to edge 5
                        search_index_start =
                          fe_p.get_first_line_index() + 5 * n_dofs_per_edge;
                        search_index_end = first_quad_index_p;
                      }
                    else if (f == 4)
                      {
                        // simplex face 0
                        search_index_start =
                          first_quad_index_p + 0 * n_dofs_per_face_p;
                        search_index_end =
                          first_quad_index_p + 1 * n_dofs_per_face_p;
                      }
                    else
                      {
                        DEAL_II_NOT_IMPLEMENTED();
                      }
                    bool found_identity = false;
                    for (unsigned int j = search_index_start;
                         j < search_index_end && found_identity == false;
                         ++j)
                      {
                        const auto p_simplex = fe_p.unit_support_point(j);

                        if (p_simplex.distance(p_transformed) < 1e-12)
                          {
                            min_index = j;
                            ++found_identities_in_face;
                            found_identity = true;
                          }
                      }
                    hypercube_to_simplex_index[i] = min_index;
                  }
              }
          }

        // go over interior nodes
        for (unsigned int i = first_interior_index_q, found_point_couter = 0;
             i < n_dofs_per_cell_q &&
             found_point_couter < fe_p.n_dofs_per_hex();
             ++i)
          {
            // again we can only map the number of simplex interior points one
            // to one
            const auto p_transformed =
              duffy_transform_quad_to_tri(fe_q.unit_support_point(i));
            unsigned int min_index = numbers::invalid_unsigned_int;

            bool found_point = false;
            for (unsigned int j = first_interior_index_p;
                 j < n_dofs_per_cell_p && found_point == false;
                 ++j)
              {
                const auto p_simplex = fe_p.unit_support_point(j);

                if (p_simplex.distance(p_transformed) < 1e-12)
                  {
                    min_index   = j;
                    found_point = true;
                    ++found_point_couter;
                  }
              }
            hypercube_to_simplex_index[i] = min_index;
          }
      }
    // check that there are a few non interpolated DoFs
    unsigned int n_invalid_entries = 0;
    for (const auto &i : hypercube_to_simplex_index)
      if (i == numbers::invalid_unsigned_int)
        ++n_invalid_entries;
    Assert(n_invalid_entries == n_dofs_per_cell_q - n_dofs_to_interpolate,
           ExcInternalError());

    DEAL_II_NOT_IMPLEMENTED();
    // TODO: this does not work currently
    // the duffy transformed face dofs are non symmetric so we cannot use them
    // easily but now we do not load all simplex dofs, so we also cannot
    // interpolate
    //  check that all simplex indices appear
    std::vector<unsigned int> simplex_usage_count(n_dofs_per_cell_p, 0);
    for (const unsigned int p_index : hypercube_to_simplex_index)
      if (p_index != numbers::invalid_unsigned_int)
        ++simplex_usage_count[p_index];
    Assert(std::all_of(simplex_usage_count.begin(),
                       simplex_usage_count.end(),
                       [](const unsigned int count) { return count > 0; }),
           ExcMessage(
             "The quadrilateral to triangle DoF lookup does not reference "
             "every simplex DoF."));

    return hypercube_to_simplex_index;
  }

  std::vector<unsigned int> create_simplex_to_hypercube_map(
    std::vector<unsigned int> &hypercube_to_simplex_index,
    const unsigned int         n_dofs_per_cell_p)
  {
    // we also need the other direction, but this is not unique
    std::vector<unsigned int> simplex_to_hypercube_index(
      n_dofs_per_cell_p, numbers::invalid_unsigned_int);
    for (unsigned int i = 0; i < simplex_to_hypercube_index.size(); ++i)
      {
        bool not_found_entry = true;
        for (unsigned int j = 0;
             j < hypercube_to_simplex_index.size() && not_found_entry;
             ++j)
          if (hypercube_to_simplex_index[j] == i)
            {
              simplex_to_hypercube_index[i] = j;
              not_found_entry               = false;
            }
      }
    Assert(std::none_of(simplex_to_hypercube_index.begin(),
                        simplex_to_hypercube_index.end(),
                        [](const unsigned int index) {
                          return index == numbers::invalid_unsigned_int;
                        }),
           ExcMessage(
             "The triangle to quadrilateral DoF lookup is incomplete."));

    return simplex_to_hypercube_index;
  }

  void compute_interior_interpolation(
    const std::vector<unsigned int> &hypercube_to_simplex_index,
    const std::vector<unsigned int> &simplex_to_hypercube_index,
    const FiniteElement<dim>        &fe_p,
    const FE_Q<dim>                 &fe_q)
  {
    // clear data fields
    interpolation_coefficients.clear();
    interpolation_source.clear();

    // lexiographic indices of all unmapped face and interior nodes
    for (auto &i : interior_indices_q)
      i = numbers::invalid_unsigned_int;

    // indices and interpolation factors for all lexiogrpahic numbered interior
    // nodes
    for (auto &i : interpolation_row_start)
      i = numbers::invalid_unsigned_int;
    interpolation_row_start[0] = 0;

    // get some constants
    const std::vector<unsigned int> lex_to_standard =
      fe_q.get_poly_space_numbering_inverse();

    const std::vector<unsigned int> &standard_to_lex =
      fe_q.get_poly_space_numbering();

    const unsigned int n_dofs_per_cell_p  = fe_p.n_dofs_per_cell();
    const unsigned int first_quad_index_q = fe_q.get_first_quad_index();

    const unsigned int n_face_and_interior_nodes_q =
      dim == 2 ? fe_q.n_dofs_per_quad() :
                 6 * fe_q.n_dofs_per_quad() + fe_q.n_dofs_per_hex();
    // go over all interior nodes in "standard" ordering
    for (unsigned int face_or_interior_node_counter = 0, i = 0;
         i < n_face_and_interior_nodes_q;
         ++i)
      {
        // get standard node index
        const unsigned int q_index = first_quad_index_q + i;

        // only need to handle nodes we did not handle before
        if (hypercube_to_simplex_index[q_index] ==
            numbers::invalid_unsigned_int)
          {
            // get lexiographic index
            const unsigned int q_index_lexiographic = standard_to_lex[q_index];

            // get point on simplex
            const Point<dim> point_q_on_simplex =
              duffy_transform_quad_to_tri(fe_q.unit_support_point(q_index));

            // now evaluate all shape functions at the point
            for (unsigned int j = 0; j < n_dofs_per_cell_p; ++j)
              {
                const Number interpolation_factor =
                  fe_p.shape_value(j, point_q_on_simplex);

                if (std::abs(interpolation_factor) > 1e-12)
                  {
                    Assert(
                      interior_indices_q[face_or_interior_node_counter] ==
                          numbers::invalid_unsigned_int ||
                        interior_indices_q[face_or_interior_node_counter] ==
                          q_index_lexiographic,
                      ExcInternalError());

                    interior_indices_q[face_or_interior_node_counter] =
                      q_index_lexiographic;

                    // quad index for the current shape function
                    const unsigned int shape_function_index_on_q =
                      simplex_to_hypercube_index[j];

                    interpolation_coefficients.emplace_back(
                      interpolation_factor);
                    interpolation_source.emplace_back(
                      standard_to_lex[shape_function_index_on_q]);
                  }
              }
            interpolation_row_start[face_or_interior_node_counter + 1] =
              interpolation_coefficients.size();
            ++face_or_interior_node_counter;
          }
      }

    for (unsigned int i = 0; i < hypercube_to_simplex_index.size(); ++i)
      {
        // check that all entries are either in quad_to_triangle or in
        // interior_indices
        bool is_in_interior = false;
        for (const auto idx : interior_indices_q)
          if (lex_to_standard[idx] == i)
            is_in_interior = true;

        Assert(is_in_interior ||
                 hypercube_to_simplex_index[i] != numbers::invalid_unsigned_int,
               ExcInternalError());
      }
  }

  void setup_collapsed_data()
  {
    // Set up collapsed data
    constexpr int                   n_DoF_1D = fe_degree + 1;
    const QGauss<1>                 quad(n_q_1D);
    const FE_Q<1>                   fe_1D = FE_Q<1>(fe_degree);
    const std::vector<unsigned int> lexiographic_numbering =
      fe_1D.get_poly_space_numbering_inverse();

    std::vector<double> values_array(n_DoF_1D * n_q_1D);
    std::vector<double> gradients_array(n_DoF_1D * n_q_1D);

    for (unsigned int dof_idx = 0; dof_idx < n_DoF_1D; ++dof_idx)
      for (unsigned int q = 0; q < n_q_1D; ++q)
        {
          const unsigned int dof_idx_lexio = lexiographic_numbering[dof_idx];
          values_array[dof_idx * n_q_1D + q] =
            fe_1D.shape_value(dof_idx_lexio, quad.point(q));
          gradients_array[q + dof_idx * n_q_1D] =
            fe_1D.shape_grad(dof_idx_lexio, quad.point(q))[0];
        }

    // setup data for even-odd decomposition
    auto convert_to_eo = [](const std::vector<double> &array,
                            const unsigned             n_rows,
                            const unsigned             n_cols) {
      const unsigned int    stride = (n_cols + 1) / 2;
      AlignedVector<Number> array_eo(n_rows * stride);
      for (unsigned int i = 0; i < n_rows / 2; ++i)
        for (unsigned int q = 0; q < stride; ++q)
          {
            array_eo[i * stride + q] =
              0.5 *
              (array[i * n_cols + q] + array[i * n_cols + n_cols - 1 - q]);
            array_eo[(n_rows - 1 - i) * stride + q] =
              0.5 *
              (array[i * n_cols + q] - array[i * n_cols + n_cols - 1 - q]);
          }
      if ((n_rows - 1) % 2 == 0)
        for (unsigned int q = 0; q < stride; ++q)
          {
            array_eo[(n_rows - 1) / 2 * stride + q] =
              array[((n_rows - 1) / 2) * n_cols + q];
          }

      return array_eo;
    };

    this->values_eo    = convert_to_eo(values_array, fe_degree + 1, n_q_1D);
    this->gradients_eo = convert_to_eo(gradients_array, fe_degree + 1, n_q_1D);

    // quadrature data
    for (unsigned int k = 0, counter = 0; k < n_q_1D; ++k)
      for (unsigned int j = 0; j < n_q_1D; ++j)
        for (unsigned int i = 0; i < n_q_1D; ++i, ++counter)
          {
            const dealii::Point<dim> p(quad.point(i)[0],
                                       quad.point(j)[0],
                                       quad.point(k)[0]);

            const number factor = 1 / ((1. - p[1]) * (1. - p[2]));

            g_11_q[counter] = factor;
            g_21_q[counter] = p[0] * factor;
            g_22_q[counter] = 1. / (1. - p[2]);
            g_32_q[counter] = p[1] / (1. - p[2]);

            quadrature_weights_symmetric[counter] =
              (1. - p[1]) * (1. - p[2]) * (1. - p[2]) * quad.weight(i) *
              quad.weight(j) * quad.weight(k);
            // TODO: optimize with 1D values only
          }

    Assert(dim == 3, ExcNotImplemented());
    // tensorproduct not optimized with even odd in 2D yet
  }

  void reinit_sum_factorisation(const DoFHandler<dim>           &dof_handler,
                                const AffineConstraints<number> &constraints)
  {
    // build the dof index map
    // first get the one to one mappings on the vertices and edges, then
    // determine the one to one mappings on the faces in 3D and the interior
    // points, lastly compute the interpolation factors for the unmapped dofs
    constexpr unsigned int n_lanes = VectorizedArray<number>::size();

    const auto &fe_p = dof_handler.get_fe();

    const FE_Q<dim>                 fe_q(fe_degree);
    const std::vector<unsigned int> lex_to_standard =
      fe_q.get_poly_space_numbering_inverse();

    const unsigned int n_dofs_per_cell_q = fe_q.n_dofs_per_cell();
    const unsigned int n_dofs_per_cell_p = fe_p.n_dofs_per_cell();

    std::vector<unsigned int> hypercube_to_simplex_index =
      create_hypercube_to_simplex_index_map(fe_p, fe_q);

    const std::vector<unsigned int> simplex_to_hypercube_index =
      create_simplex_to_hypercube_map(hypercube_to_simplex_index,
                                      fe_p.n_dofs_per_cell());

    compute_interior_interpolation(hypercube_to_simplex_index,
                                   simplex_to_hypercube_index,
                                   fe_p,
                                   fe_q);

    constexpr unsigned int n_dofs_to_load =
      Utilities::pow(fe_degree + 1, dim) - n_dofs_to_interpolate;

    // order after simplex indices, such that the same simplex index can be read
    // from cache
    std::vector<unsigned int> fe_p_indices_loaded(
      n_dofs_per_cell_q, numbers::invalid_unsigned_int);
    for (unsigned int i = 0; i < n_dofs_per_cell_q; ++i)
      fe_p_indices_loaded[i] = hypercube_to_simplex_index[lex_to_standard[i]];

    for (auto &i : values_to_load_q_index)
      i = numbers::invalid_unsigned_int;

    for (unsigned int i = 0, counter = 0; i < n_dofs_per_cell_p; ++i)
      for (unsigned int j = 0; j < n_dofs_per_cell_q; ++j)
        if (fe_p_indices_loaded[j] == i)
          values_to_load_q_index[counter++] = j;

    for (auto &i : values_to_load_q_index)
      Assert(i != numbers::invalid_unsigned_int, ExcInternalError());

    // now get the new dof indices
    manual_dof_indices.reinit(matrix_free.n_cell_batches(),
                              n_dofs_to_load * n_lanes,
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
            // for (unsigned int i = 0, counter = 0; i < n_dofs_per_cell_q; ++i)
            //   {
            //     // corresponding index on tri. need to convert from
            //     // lexiographice to "standard" ordering
            //     const unsigned int fe_p_index =
            //       hypercube_to_simplex_index[lex_to_standard[i]];

            //     if (fe_p_index != numbers::invalid_unsigned_int)
            //       {
            //         const types::global_dof_index dof_index =
            //           dof_indices[fe_p_index];
            //         values_to_load_q_index[counter] = i;

            //         if (constraints.is_constrained(dof_index))
            //           has_constraints = true;
            //         else
            //           manual_dof_indices(c, counter * n_lanes + v) =
            //             matrix_free.get_dof_info()
            //               .vector_partitioner->global_to_local(dof_index);
            //         ++counter;
            //       }
            //   }

            for (unsigned int i = 0; i < n_dofs_to_load; ++i)
              {
                // corresponding index on tri. need to convert from
                // lexiographice to "standard" ordering
                const unsigned int fe_p_index = hypercube_to_simplex_index
                  [lex_to_standard[values_to_load_q_index[i]]];

                const types::global_dof_index dof_index =
                  dof_indices[fe_p_index];

                if (constraints.is_constrained(dof_index))
                  has_constraints = true;
                else
                  manual_dof_indices(c, i * n_lanes + v) =
                    matrix_free.get_dof_info()
                      .vector_partitioner->global_to_local(dof_index);
              }
          }
        dof_indices_have_constraints[c] = has_constraints;
      }

    // Set up collapsed data
    setup_collapsed_data();
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

    for (const auto &constrained_index : constrained_indices)
      dst.local_element(constrained_index) =
        src.local_element(constrained_index);
  }

  virtual void vmult_sum_factorisation(VectorType       &dst,
                                       const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_sum_factorisation, this, dst, src, true);

    for (const auto &constrained_index : constrained_indices)
      dst.local_element(constrained_index) =
        src.local_element(constrained_index);
  }

  virtual void vmult_masked_gather(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_masked_gather, this, dst, src, true);

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

  void do_cell_integral_sum_factorisation(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    AlignedVector<VectorizedArray<number>> *scratch_data =
      matrix_free.acquire_scratch_data();

    constexpr unsigned int n_dofs_per_cell_q =
      Utilities::pow(fe_degree + 1, dim);
    constexpr unsigned int n_DoF_1D   = fe_degree + 1;
    constexpr unsigned int n_q_points = Utilities::pow(n_q_1D, dim);
    constexpr unsigned int n_lanes    = VectorizedArray<number>::size();

    const auto &mapping_data = matrix_free.get_mapping_info().cell_data[0];

    constexpr unsigned int size_imideate_arrays =
      dim == 2 ? n_q_1D * n_DoF_1D :
                 n_q_1D * n_DoF_1D * n_DoF_1D + n_q_1D * n_q_1D * n_DoF_1D;

    constexpr unsigned int values_quad_size =
      n_dofs_per_cell_q + n_q_points * dim;

    scratch_data->resize_fast(values_quad_size + size_imideate_arrays);
    VectorizedArray<number> *values_dofs = scratch_data->begin();
    VectorizedArray<number> *gradients_quad =
      scratch_data->begin() + n_dofs_per_cell_q;

    VectorizedArray<number> *temp_q = scratch_data->begin() + values_quad_size;
    VectorizedArray<number> *temp_qq =
      dim == 3 ? scratch_data->begin() + values_quad_size +
                   n_q_1D * n_DoF_1D * n_DoF_1D :
                 scratch_data->begin() + values_quad_size;

    const number *src_ptr = src.begin();

    constexpr unsigned int n_dofs_to_load =
      n_dofs_per_cell_q - n_dofs_to_interpolate;

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        // read dof values
        const unsigned int *dof_indices = &manual_dof_indices(cell, 0);
        if (dof_indices_have_constraints[cell])
          {
            for (unsigned int i = 0; i < n_dofs_to_load;
                 ++i, dof_indices += n_lanes)
              {
                values_dofs[values_to_load_q_index[i]] = {};
                for (unsigned int v = 0; v < n_lanes; ++v)
                  if (dof_indices[v] != numbers::invalid_unsigned_int)
                    values_dofs[values_to_load_q_index[i]][v] =
                      src_ptr[dof_indices[v]];
              }
          }
        else
          for (unsigned int i = 0; i < n_dofs_to_load;
               ++i, dof_indices += n_lanes)
            {
              // values_dofs[values_to_load_q_index[i]] = {};
              for (unsigned int v = 0; v < n_lanes; ++v)
                values_dofs[values_to_load_q_index[i]][v] =
                  src_ptr[dof_indices[v]];
            }

        // after reading the dof values interpolate the interior dofs
        for (unsigned int i = 0; i < n_dofs_to_interpolate; ++i)
          {
            VectorizedArray<number> value = {};
            for (unsigned int k = interpolation_row_start[i];
                 k < interpolation_row_start[i + 1];
                 ++k)
              value += interpolation_coefficients[k] *
                       values_dofs[interpolation_source[k]];

            values_dofs[interior_indices_q[i]] = value;
          }

        // do the interpolation now
        for (unsigned int k = 0; k < n_DoF_1D; ++k)
          for (unsigned int j = 0; j < n_DoF_1D; ++j)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::gradient,
              n_q_1D,
              n_DoF_1D,
              1,
              1,
              /*transpose_matrix*/ true,
              /*add*/ false>(gradients_eo.data(),
                             values_dofs + k * n_DoF_1D * n_DoF_1D +
                               j * n_DoF_1D,
                             temp_q + k * n_q_1D * n_DoF_1D + j * n_q_1D);

        // interpolate SyDx
        for (unsigned int k = 0; k < n_DoF_1D; ++k)
          for (unsigned int r = 0; r < n_q_1D; ++r)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::value,
              n_q_1D,
              n_DoF_1D,
              n_q_1D,
              n_q_1D,
              /*transpose_matrix*/ true,
              /*add*/ false>(values_eo.data(),
                             temp_q + k * n_DoF_1D * n_q_1D + r,
                             temp_qq + k * n_q_1D * n_q_1D + r);

        // interpolate SzSyDx
        for (unsigned int s = 0; s < n_q_1D; ++s)
          for (unsigned int r = 0; r < n_q_1D; ++r)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::value,
              n_q_1D,
              n_DoF_1D,
              n_q_1D * n_q_1D,
              3 * n_q_1D * n_q_1D,
              /*transpose_matrix*/ true,
              /*add*/ false>(values_eo.data(),
                             temp_qq + s * n_q_1D + r,
                             gradients_quad + 3 * (s * n_q_1D + r));

        // interpolate Sx
        for (unsigned int k = 0; k < n_DoF_1D; ++k)
          for (unsigned int j = 0; j < n_DoF_1D; ++j)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::value,
              n_q_1D,
              n_DoF_1D,
              1,
              1,
              /*transpose_matrix*/ true,
              /*add*/ false>(values_eo.data(),
                             values_dofs + k * n_DoF_1D * n_DoF_1D +
                               j * n_DoF_1D,
                             temp_q + k * n_q_1D * n_DoF_1D + j * n_q_1D);

        // interpolate DySx
        for (unsigned int k = 0; k < n_DoF_1D; ++k)
          for (unsigned int r = 0; r < n_q_1D; ++r)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::gradient,
              n_q_1D,
              n_DoF_1D,
              n_q_1D,
              n_q_1D,
              /*transpose_matrix*/ true,
              /*add*/ false>(gradients_eo.data(),
                             temp_q + k * n_DoF_1D * n_q_1D + r,
                             temp_qq + k * n_q_1D * n_q_1D + r);

        // interpolate SzDySx
        for (unsigned int s = 0; s < n_q_1D; ++s)
          for (unsigned int r = 0; r < n_q_1D; ++r)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::value,
              n_q_1D,
              n_DoF_1D,
              n_q_1D * n_q_1D,
              3 * n_q_1D * n_q_1D,
              /*transpose_matrix*/ true,
              /*add*/ false>(values_eo.data(),
                             temp_qq + s * n_q_1D + r,
                             gradients_quad + 3 * (s * n_q_1D + r) + 1);

        // interpolate SySx
        for (unsigned int k = 0; k < n_DoF_1D; ++k)
          for (unsigned int r = 0; r < n_q_1D; ++r)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::value,
              n_q_1D,
              n_DoF_1D,
              n_q_1D,
              n_q_1D,
              /*transpose_matrix*/ true,
              /*add*/ false>(values_eo.data(),
                             temp_q + k * n_DoF_1D * n_q_1D + r,
                             temp_qq + k * n_q_1D * n_q_1D + r);

        // interpolate DzSySx
        for (unsigned int s = 0; s < n_q_1D; ++s)
          for (unsigned int r = 0; r < n_q_1D; ++r)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::gradient,
              n_q_1D,
              n_DoF_1D,
              n_q_1D * n_q_1D,
              3 * n_q_1D * n_q_1D,
              /*transpose_matrix*/ true,
              /*add*/ false>(gradients_eo.data(),
                             temp_qq + s * n_q_1D + r,
                             gradients_quad + 3 * (s * n_q_1D + r) + 2);

        // quadrature point operation
        const unsigned int offsets = mapping_data.data_index_offsets[cell];
        const Tensor<2, dim, VectorizedArray<number>> *jac =
          mapping_data.jacobians[0].data() + offsets;
        const VectorizedArray<number> j_value =
          mapping_data.JxW_values[offsets];
        VectorizedArray<number> *grad_ptr = gradients_quad;

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
            const number g_11 = g_11_q[q];
            const number g_21 = g_21_q[q];
            const number g_22 = g_22_q[q];
            const number g_32 = g_32_q[q];

            Tensor<1, dim, VectorizedArray<number>> grad;
            grad[0] = g_11 * grad_ptr[0];
            grad[1] = g_21 * grad_ptr[0] + g_22 * grad_ptr[1];
            grad[2] = g_21 * grad_ptr[0] + g_32 * grad_ptr[1] + grad_ptr[2];

            const number weight = quadrature_weights_symmetric[q];
            // TODO: optimize with 1D values only

            Tensor<1, dim, VectorizedArray<number>> result =
              my_metric * weight * grad;

            grad_ptr[0] =
              g_11 * result[0] + g_21 * result[1] + g_21 * result[2];
            grad_ptr[1] = g_22 * result[1] + g_32 * result[2];
            grad_ptr[2] = result[2];
          }

        // integrate Sz
        for (unsigned int s = 0; s < n_q_1D; ++s)
          for (unsigned int i = 0; i < n_q_1D; ++i)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::value,
              n_q_1D,
              n_DoF_1D,
              dim * n_q_1D * n_q_1D,
              n_q_1D * n_q_1D,
              /*transpose_matrix*/ false,
              /*add*/ false>(values_eo.data(),
                             gradients_quad + dim * (s * n_q_1D + i),
                             temp_qq + s * n_q_1D + i);

        // integrate SySz
        for (unsigned int t = 0; t < n_DoF_1D; ++t)
          for (unsigned int i = 0; i < n_q_1D; ++i)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::value,
              n_q_1D,
              n_DoF_1D,
              n_q_1D,
              n_q_1D,
              /*transpose_matrix*/ false,
              /*add*/ false>(values_eo.data(),
                             temp_qq + t * n_q_1D * n_q_1D + i,
                             temp_q + t * n_DoF_1D * n_q_1D + i);

        // integrate DxSzSy
        for (unsigned int j = 0; j < n_DoF_1D; ++j)
          for (unsigned int i = 0; i < n_DoF_1D; ++i)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::gradient,
              n_q_1D,
              n_DoF_1D,
              1,
              1,
              /*transpose_matrix*/ false,
              /*add*/ false>(gradients_eo.data(),
                             temp_q + j * n_DoF_1D * n_q_1D + i * n_q_1D,
                             values_dofs + j * n_DoF_1D * n_DoF_1D +
                               i * n_DoF_1D);

        // integrate Sz
        for (unsigned int s = 0; s < n_q_1D; ++s)
          for (unsigned int i = 0; i < n_q_1D; ++i)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::value,
              n_q_1D,
              n_DoF_1D,
              dim * n_q_1D * n_q_1D,
              n_q_1D * n_q_1D,
              /*transpose_matrix*/ false,
              /*add*/ false>(values_eo.data(),
                             gradients_quad + dim * (s * n_q_1D + i) + 1,
                             temp_qq + s * n_q_1D + i);

        // integrate DySz
        for (unsigned int t = 0; t < n_DoF_1D; ++t)
          for (unsigned int i = 0; i < n_q_1D; ++i)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::gradient,
              n_q_1D,
              n_DoF_1D,
              n_q_1D,
              n_q_1D,
              /*transpose_matrix*/ false,
              /*add*/ false>(gradients_eo.data(),
                             temp_qq + t * n_q_1D * n_q_1D + i,
                             temp_q + t * n_DoF_1D * n_q_1D + i);

        // integrate Dz
        for (unsigned int s = 0; s < n_q_1D; ++s)
          for (unsigned int i = 0; i < n_q_1D; ++i)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::gradient,
              n_q_1D,
              n_DoF_1D,
              dim * n_q_1D * n_q_1D,
              n_q_1D * n_q_1D,
              /*transpose_matrix*/ false,
              /*add*/ false>(gradients_eo.data(),
                             gradients_quad + dim * (s * n_q_1D + i) + 2,
                             temp_qq + s * n_q_1D + i);

        // integrate SyDz
        for (unsigned int j = 0; j < n_DoF_1D; ++j)
          for (unsigned int i = 0; i < n_q_1D; ++i)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::value,
              n_q_1D,
              n_DoF_1D,
              n_q_1D,
              n_q_1D,
              /*transpose_matrix*/ false,
              /*add*/ true>(values_eo.data(),
                            temp_qq + j * n_q_1D * n_q_1D + i,
                            temp_q + j * n_DoF_1D * n_q_1D + i);

        // integrate Sx(DySz + SyDz)
        for (unsigned int j = 0; j < n_DoF_1D; ++j)
          for (unsigned int i = 0; i < n_DoF_1D; ++i)
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_evenodd,
              dealii::internal::EvaluatorQuantity::value,
              n_q_1D,
              n_DoF_1D,
              1,
              1,
              /*transpose_matrix*/ false,
              /*add*/ true>(values_eo.data(),
                            temp_q + j * n_DoF_1D * n_q_1D + i * n_q_1D,
                            values_dofs + j * n_DoF_1D * n_DoF_1D +
                              i * n_DoF_1D);

        // distribute interior dofs
        for (unsigned int i = 0; i < n_dofs_to_interpolate; ++i)
          {
            const VectorizedArray<number> value =
              values_dofs[interior_indices_q[i]];

            for (unsigned int k = interpolation_row_start[i];
                 k < interpolation_row_start[i + 1];
                 ++k)
              values_dofs[interpolation_source[k]] +=
                interpolation_coefficients[k] * value;
          }

        // distribute local to global
        dof_indices = &manual_dof_indices(cell, 0);
        if (dof_indices_have_constraints[cell])
          {
            for (unsigned int i = 0; i < n_dofs_to_load;
                 ++i, dof_indices += n_lanes)
              {
                for (unsigned int v = 0; v < n_lanes; ++v)
                  if (dof_indices[v] != numbers::invalid_unsigned_int)
                    dst.local_element(dof_indices[v]) +=
                      values_dofs[values_to_load_q_index[i]][v];
              }
          }
        else
          for (unsigned int i = 0; i < n_dofs_to_load;
               ++i, dof_indices += n_lanes)
            {
              for (unsigned int v = 0; v < n_lanes; ++v)
                dst.local_element(dof_indices[v]) +=
                  values_dofs[values_to_load_q_index[i]][v];
            }
      }

    matrix_free.release_scratch_data(scratch_data);
  }


  void do_cell_integral_masked_gather(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    // get data on the referece cell
    AlignedVector<VectorizedArray<number>> *scratch_data =
      matrix_free.acquire_scratch_data();
    const internal::MatrixFreeFunctions::ShapeInfo<number> &shape_info =
      matrix_free.get_shape_info();
    const unsigned int dofs_per_cell = shape_info.dofs_per_component_on_cell;
    // set the batch size
    constexpr unsigned int batch_size = 4;
    const unsigned int     n_q_points = shape_info.n_q_points;
    // get the SIMD width
    constexpr unsigned int n_lanes = VectorizedArray<number>::size();

    // get mapping data and quadrature weights
    const auto   &mapping_data = matrix_free.get_mapping_info().cell_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();

    // reserve space for the gradients interpolated to the quadrature points
    scratch_data->resize_fast(batch_size * (dim * n_q_points + dofs_per_cell));
    // and set pointers
    VectorizedArray<number> *values_dofs = scratch_data->begin();
    VectorizedArray<number> *gradients_quad =
      scratch_data->begin() + batch_size * dofs_per_cell;

    // loop over the local cells
    for (unsigned int cell = range.first; cell < range.second;
         cell += batch_size)
      {
        // read dof values manually to prevent deal.ii overhead
        // step G_e u
        const unsigned int my_batch_size =
          cell + batch_size <= range.second ? batch_size : range.second - cell;
        // get DoF indices of each cell in the batch
        const unsigned int *dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const number *src_ptr = src.begin();
            // if the cell has constrained DoFs, read only the non constrained
            // DoFs
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  {
                    values_dofs[batch * dofs_per_cell + i] = {};
                    // loop over all SIMD lanes
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        values_dofs[batch * dofs_per_cell + i][v] =
                          src_ptr[dof_indices[v]];
                  }
              }
            else
              // no constraints on this cell
              // read all DoFs on the cell
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  values_dofs[batch * dofs_per_cell + i] = {};
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    values_dofs[batch * dofs_per_cell + i][v] =
                      src_ptr[dof_indices[v]];
                }
          }

        // interpolation step E_e u_e
        // apply the matrix-matrix product to the cells batched together above
        // (called MM-Mult in the paper)
        // note that in shape_info.data[0].shape_gradients.data()
        // the gradient of all shape functions are evaluated at all quadrature
        // points on the reference element (as described in the data structures)
        // section
        apply_matrix_vector_product<true>(
          shape_info.data[0].shape_gradients.data(),
          values_dofs,
          gradients_quad,
          dofs_per_cell,
          n_q_points * dim);

        // quadrature point operation D_e
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            // Get Jacobians
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
                // Compute metric tensor $J^{-1}J^{-T}$ as it is constant on one
                // cell over all quadrature points
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
                    // get transform gradient to the real space and back
                    // also apply JxW
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
                // curvilinear case
                // compute the gradient in real space and transform back
                // apply JxW at every quadrature point
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

        // integrate E_e^T
        // use same optimizations as before
        apply_matrix_vector_product<false>(
          shape_info.data[0].shape_gradients.data(),
          gradients_quad,
          values_dofs,
          dofs_per_cell,
          n_q_points * dim);

        // distribute local to global G_e^T
        // skip constrained indices
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
                          values_dofs[batch * dofs_per_cell + i][v];
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

  AlignedVector<number> values_eo;
  AlignedVector<number> gradients_eo;

  std::array<number, Utilities::pow(n_q_1D, dim)> g_11_q;
  std::array<number, Utilities::pow(n_q_1D, dim)> g_21_q;
  std::array<number, Utilities::pow(n_q_1D, dim)> g_22_q;
  std::array<number, Utilities::pow(n_q_1D, dim)> g_32_q;
  std::array<number, Utilities::pow(n_q_1D, dim)> quadrature_weights_symmetric;

  static constexpr unsigned int n_dofs_to_interpolate =
    dim == 2 ? Utilities::pow(fe_degree + 1, dim) -
                 n_dof_simplex<dim, fe_degree>() - (fe_degree - 1) - 1 :
               Utilities::pow(fe_degree + 1, dim) -
                 (n_dof_simplex<dim, fe_degree>() + 4 + 6 * (fe_degree - 1) +
                  2 * (fe_degree - 1) * (fe_degree - 1));

  std::array<unsigned int, n_dofs_to_interpolate>     interior_indices_q;
  std::array<unsigned int, n_dofs_to_interpolate + 1> interpolation_row_start;
  std::vector<unsigned int>                           interpolation_source;
  std::vector<number> interpolation_coefficients;

  std::array<unsigned int,
             Utilities::pow(fe_degree + 1, dim) - n_dofs_to_interpolate>
    values_to_load_q_index;
};

template <int dim, int fe_degree, typename Number>
void do_test(const unsigned int refine_max)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  pcout << "Running in " << dim << "D with degree " << fe_degree << std::endl;

  const std::vector<Point<dim>> fe_p_support_points =
    adopted_fe_p_support_points<dim>(FE_Q<dim>(fe_degree),
                                     FE_SimplexP<dim>(fe_degree, false));

  if (false)
    {
      for (const auto &p : fe_p_support_points)
        std::cout << p << ", ";
      std::cout << std::endl;
    }

  FE_SimplexP<dim>   fe(fe_degree, fe_p_support_points);
  MappingFE<dim>     mapping(FE_SimplexP<dim>(1, false));
  QGaussSimplex<dim> quad(fe_degree + 1);

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

      Operator<dim, fe_degree, fe_degree + 1, 1, Number> op;
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

      for (unsigned int r = 0; r < 5; ++r)
        {
          Timer time;
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("matvec_p" + std::to_string(fe_degree) + "_s" +
                               std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          for (unsigned int t = 0; t < 100; ++t)
            op.vmult(vec2, vec1);
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
          pcout << "n_dofs mf gather " << dof_handler.n_dofs() << "  time "
                << run_time / 100 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
        }

      op.reinit_sum_factorisation(dof_handler, constraint);
      for (unsigned int r = 0; r < 5; ++r)
        {
          Timer time;
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("matvec_sum_factorisation_p" +
                               std::to_string(fe_degree) + "_s" +
                               std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          for (unsigned int t = 0; t < 100; ++t)
            op.vmult_sum_factorisation(vec4, vec1);
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(("matvec_sum_factorisation_p" +
                              std::to_string(fe_degree) + "_s" +
                              std::to_string(dof_handler.n_dofs()))
                               .c_str());
#endif
          const double run_time = time.wall_time();
          pcout << "n_dofs mf sum-factorisation " << dof_handler.n_dofs()
                << "  time " << run_time / 100 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
        }
      pcout << std::endl;

      vec3 -= vec2;
      vec4 -= vec2;
      pcout << "   Error MF variants: " << vec3.l2_norm() / vec2.l2_norm()
            << " " << vec4.l2_norm() / vec2.l2_norm() << std::endl
            << std::endl
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
    {
      Assert(false, ExcNotImplemented());
    }
  else
    {
      if (degree == 1)
        do_test<3, 1, double>(refine_max);
      if (degree == 2)
        do_test<3, 2, double>(refine_max);
      if (degree == 3)
        do_test<3, 3, double>(refine_max);
      if (degree == 4)
        do_test<3, 4, double>(refine_max);
      if (degree == 5)
        do_test<3, 5, double>(refine_max);
      if (degree == 6)
        do_test<3, 6, double>(refine_max);
      if (degree == 7)
        do_test<3, 7, double>(refine_max);
      if (degree == 8)
        do_test<3, 8, double>(refine_max);
      if (degree == 9)
        do_test<3, 9, double>(refine_max);
      if (degree == 10)
        do_test<3, 10, double>(refine_max);
    }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
