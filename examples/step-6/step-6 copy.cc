
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include "./../../../tests/simplex/simplex_grids.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

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
#include <deal.II/lac/sparse_matrix.h>


#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif


using namespace dealii;


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

  VectorizedArray<double>
  value_array(const Point<dim, VectorizedArray<double>> &p)
  {
    Point<dim>              point;
    VectorizedArray<double> results;

    for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
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

    for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
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


template <int dim, typename Number = double>
class PoissonProblem
{
  using value_type = Number;
  using number     = Number;
  using VectorType = Vector<Number>;

public:
  PoissonProblem(const Mapping<dim>              &mapping,
                 const DoFHandler<dim>           &dof_handler,
                 const Quadrature<dim>           &quad,
                 const AffineConstraints<number> &constraints)
  {
    system_rhs.reinit(dof_handler.n_dofs());
    system_rhs = 0.0;

    system_matrix.clear();
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, constraints);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    const FiniteElement<dim> *fe = &dof_handler.get_fe();

    FEValues<dim> fe_values(mapping,
                            *fe,
                            quad,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const QGauss<dim - 1>        quadrature_quad(fe->degree + 1);
    const QGaussSimplex<dim - 1> quadrature_tri(fe->degree + 1);

    FEFaceValues<dim> fe_face_values_quad(mapping,
                                          *fe,
                                          quadrature_quad,
                                          update_values | update_gradients |
                                            update_quadrature_points |
                                            update_normal_vectors |
                                            update_JxW_values);

    FEFaceValues<dim> fe_face_values_tri(mapping,
                                         *fe,
                                         quadrature_tri,
                                         update_values | update_gradients |
                                           update_quadrature_points |
                                           update_normal_vectors |
                                           update_JxW_values);


    FEFaceValues<dim> fe_neighbor_face_values_quad(mapping,
                                                   *fe,
                                                   quadrature_quad,
                                                   update_values |
                                                     update_gradients |
                                                     update_quadrature_points |
                                                     update_normal_vectors |
                                                     update_JxW_values);

    FEFaceValues<dim> fe_neighbor_face_values_tri(mapping,
                                                  *fe,
                                                  quadrature_tri,
                                                  update_values |
                                                    update_gradients |
                                                    update_quadrature_points |
                                                    update_normal_vectors |
                                                    update_JxW_values);

    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
    const unsigned int n_q_points    = fe_values.n_quadrature_points;

    FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<Number> cell_neighbor_matrix(dofs_per_cell, dofs_per_cell);
    Vector<Number>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> neighbor_dof_indices(dofs_per_cell);

    RightHandSide<dim> rhs;
    Solution<dim>      exact_solution;

    const Number penalty_factor =
      1.0 * (fe->degree + 1) * (fe->degree + dim) / Number(dim);
    Number              penalty_parameter_minus = 0.0;
    std::vector<Number> penalty_parameter_plus(fe->reference_cell().n_faces(),
                                               0.0);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          // compute the penalty parameter for the interior
          {
            fe_values.reinit(cell);
            // calculate cell volume
            Number volume = 0.0;
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                volume += fe_values.JxW(q);
              }

            // calculate surface area
            Number surface_area = 0.0;
            for (const unsigned int f : cell->face_indices())
              {
                auto &fe_face_values =
                  cell->face(f)->reference_cell().is_hyper_cube() ?
                    fe_face_values_quad :
                    fe_face_values_tri;

                fe_face_values.reinit(cell, f);
                const Number factor = (cell->at_boundary(f) and
                                       not(cell->has_periodic_neighbor(f))) ?
                                        1. :
                                        0.5;
                for (unsigned int q = 0; q < fe_face_values.n_quadrature_points;
                     ++q)
                  {
                    surface_area += fe_face_values.JxW(q) * factor;
                  }
              }
            penalty_parameter_minus = surface_area / volume * penalty_factor;
          }
          // compute the penalty parameter for the neighbor
          {
            for (const unsigned int face_no : cell->face_indices())
              {
                if (cell->face(face_no)->at_boundary())
                  {
                    // nothing to do
                  }
                else
                  {
                    const auto neighbor = cell->neighbor(face_no);
                    if (neighbor->is_artificial())
                      continue;

                    fe_values.reinit(neighbor);
                    // calculate cell volume
                    Number volume = 0.0;
                    for (unsigned int q = 0; q < n_q_points; ++q)
                      {
                        volume += fe_values.JxW(q);
                      }

                    // calculate surface area
                    Number surface_area = 0.0;
                    for (const unsigned int f : neighbor->face_indices())
                      {
                        auto &fe_face_values =
                          neighbor->face(f)->reference_cell().is_hyper_cube() ?
                            fe_face_values_quad :
                            fe_face_values_tri;

                        fe_face_values.reinit(neighbor, f);
                        const Number factor =
                          (neighbor->at_boundary(f) and
                           not(neighbor->has_periodic_neighbor(f))) ?
                            1. :
                            0.5;
                        for (unsigned int q = 0;
                             q < fe_face_values.n_quadrature_points;
                             ++q)
                          {
                            surface_area += fe_face_values.JxW(q) * factor;
                          }
                      }
                    penalty_parameter_plus[face_no] =
                      surface_area / volume * penalty_factor;
                  }
              }
          }

          fe_values.reinit(cell);

          cell_matrix = 0.;
          cell_rhs    = 0.;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const Number rhs_value =
                  rhs.value(fe_values.quadrature_point(q_point), 0);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) += (fe_values.shape_grad(i, q_point) *
                                        fe_values.shape_grad(j, q_point) *
                                        fe_values.JxW(q_point)); // dx

                cell_rhs(i) +=
                  (fe_values.shape_value(i, q_point) * // phi_i(x_q)
                   rhs_value *                         // f(x_q)
                   fe_values.JxW(q_point));            // dx
              }

          cell->get_dof_indices(local_dof_indices);

          // now loop over all faces
          for (const unsigned int face_no : cell->face_indices())
            {
              auto &fe_face_values =
                cell->face(face_no)->reference_cell().is_hyper_cube() ?
                  fe_face_values_quad :
                  fe_face_values_tri;

              auto &fe_neighbor_face_values =
                cell->face(face_no)->reference_cell().is_hyper_cube() ?
                  fe_neighbor_face_values_quad :
                  fe_neighbor_face_values_tri;

              const unsigned int n_face_q = fe_face_values.n_quadrature_points;

              fe_face_values.reinit(cell, face_no);

              if (cell->at_boundary(face_no))
                {
                  {
                    const Number penalty = penalty_parameter_minus;
                    for (unsigned int q = 0; q < n_face_q; ++q)
                      {
                        const Tensor<1, dim> normal =
                          fe_face_values.normal_vector(q);

                        const double g = exact_solution.value(
                          fe_face_values.quadrature_point(q), 0);

                        //   std::cout << "Face number " << face_no << "
                        //   quadrature point " << q << " at location " <<
                        //   fe_face_values.quadrature_point(q) << ", normal "
                        //   << fe_face_values.normal_vector(q) <<
                        //   ", JxW " << fe_face_values.JxW(q);

                        // std::cout << "Face number " << face_no << " face
                        // penalty " << penalty << std::endl;
                        //   for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        //    std::cout << "i " << i << ": " <<
                        //    fe_face_values.shape_value(i, q) << " " <<
                        //    fe_face_values.shape_grad(i, q) << std::endl;
                        // std::cout << std::endl;


                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            const double phi_i =
                              fe_face_values.shape_value(i, q);
                            const Tensor<1, dim> grad_phi_i =
                              fe_face_values.shape_grad(i, q);
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                              {
                                const double phi_j =
                                  fe_face_values.shape_value(j, q);
                                const Tensor<1, dim> grad_phi_j =
                                  fe_face_values.shape_grad(j, q);

                                cell_matrix(i, j) +=
                                  (-(grad_phi_j * normal) * phi_i -
                                   (grad_phi_i * normal) * phi_j +
                                   2.0 * penalty * phi_j * phi_i) *
                                  fe_face_values.JxW(q);
                              }

                            cell_rhs(i) += ((-(grad_phi_i * normal) * g) +
                                            2.0 * penalty * g * phi_i) *
                                           fe_face_values.JxW(q);
                          }
                      }
                  }
                }
              else
                {
                  const auto neighbor = cell->neighbor(face_no);
                  if (neighbor->is_artificial())
                    continue;

                  const unsigned int neighbor_face_no =
                    cell->neighbor_of_neighbor(face_no);

                  fe_neighbor_face_values.reinit(neighbor, neighbor_face_no);
                  neighbor->get_dof_indices(neighbor_dof_indices);

                  cell_neighbor_matrix = 0.0;

                  const double penalty =
                    std::max(penalty_parameter_plus[face_no],
                             penalty_parameter_minus);

                  for (unsigned int q = 0; q < n_face_q; ++q)
                    {
                      Assert(fe_face_values.quadrature_point(q).distance(
                               fe_neighbor_face_values.quadrature_point(q)) <
                               1e-14,
                             ExcInternalError());

                      const Tensor<1, dim> normal =
                        fe_face_values.normal_vector(q);

                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          const double phi_i_minus =
                            fe_face_values.shape_value(i, q);
                          const Tensor<1, dim> grad_phi_i_minus =
                            fe_face_values.shape_grad(i, q);

                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              const double phi_j_minus =
                                fe_face_values.shape_value(j, q);
                              const Tensor<1, dim> grad_phi_j_minus =
                                fe_face_values.shape_grad(j, q);

                              cell_matrix(i, j) +=
                                (-0.5 * (grad_phi_j_minus * normal) *
                                   phi_i_minus -
                                 0.5 * (grad_phi_i_minus * normal) *
                                   phi_j_minus +
                                 penalty * phi_j_minus * phi_i_minus) *
                                fe_face_values.JxW(q);

                              const double phi_j_plus =
                                fe_neighbor_face_values.shape_value(j, q);
                              const Tensor<1, dim> grad_phi_j_plus =
                                fe_neighbor_face_values.shape_grad(j, q);

                              cell_neighbor_matrix(i, j) +=
                                (-0.5 * (grad_phi_j_plus * normal) *
                                   phi_i_minus +
                                 0.5 * (grad_phi_i_minus * normal) *
                                   phi_j_plus -
                                 penalty * phi_j_plus * phi_i_minus) *
                                fe_face_values.JxW(q);
                            }
                        }
                    }
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      system_matrix.add(local_dof_indices[i],
                                        neighbor_dof_indices[j],
                                        cell_neighbor_matrix(i, j));
                }
            }

          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }
  }

  void vmult(VectorType &dst, const VectorType &src) const
  {
    system_matrix.vmult(dst, src);
  }

  MPI_Comm             mpi_communicator;
  SparseMatrix<Number> system_matrix;
  VectorType           system_rhs;

  SparsityPattern sparsity_pattern;
};

template <int dim_, int n_components = dim_, typename Number = double>
class Operator : public Subscriptor
{
public:
  using value_type = Number;
  using number     = Number;
  using VectorType = Vector<Number>;

  static const int dim = dim_;

  using FECellIntegrator = FEEvaluation<dim, -1, 0, n_components, Number>;
  using FEFaceIntegrator = FEFaceEvaluation<dim, -1, 0, n_components, Number>;

  void reinit(const Mapping<dim>              &mapping,
              const DoFHandler<dim>           &dof_handler,
              const Quadrature<dim>           &quad,
              const AffineConstraints<number> &constraints,
              const unsigned int mg_level = numbers::invalid_unsigned_int)
  {
    this->constraints.copy_from(constraints);

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points);
    data.mapping_update_flags_inner_faces =
      (update_gradients | update_JxW_values | update_normal_vectors);
    data.mapping_update_flags_boundary_faces =
      (update_gradients | update_JxW_values | update_normal_vectors |
       update_quadrature_points);
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


    FEEvaluation<dim, -1, 0, dim, number>     eval_cell(matrix_free);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_face(matrix_free, true);
    const double penalty_factor = 1.0 * (dof_handler.get_fe().degree + dim) /
                                  double(dim) *
                                  (dof_handler.get_fe().degree + 1);
    {
      unsigned int n_cells =
        matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
      array_penalty_parameter.resize(n_cells);

      const dealii::FiniteElement<dim> &fe = dof_handler.get_fe();
      // const auto reference_cells = dof_handler.get_fe().reference_cell();

      dealii::FEValues<dim> fe_values(mapping,
                                      fe,
                                      quad,
                                      dealii::update_JxW_values);

      const auto face_quadrature_quad =
        ReferenceCells::Quadrilateral.get_gauss_type_quadrature(fe.degree + 1);
      const auto face_quadrature_tri =
        ReferenceCells::Triangle.get_gauss_type_quadrature(fe.degree + 1);
      dealii::FEFaceValues<dim> fe_face_values_quad(mapping,
                                                    fe,
                                                    face_quadrature_quad,
                                                    dealii::update_JxW_values);
      dealii::FEFaceValues<dim> fe_face_values_tri(mapping,
                                                   fe,
                                                   face_quadrature_tri,
                                                   dealii::update_JxW_values);

      for (unsigned int i = 0; i < n_cells; ++i)
        {
          for (unsigned int v = 0;
               v < matrix_free.n_active_entries_per_cell_batch(i);
               ++v)
            {
              typename dealii::DoFHandler<dim>::cell_iterator cell =
                matrix_free.get_cell_iterator(i, v);
              fe_values.reinit(cell);

              // calculate cell volume
              number volume = 0;
              for (unsigned int q = 0; q < quad.size(); ++q)
                {
                  volume += fe_values.JxW(q);
                }

              // calculate surface area
              number surface_area = 0;
              for (const unsigned int f : cell->face_indices())
                {
                  auto &fe_face_values =
                    cell->face(f)->reference_cell().is_hyper_cube() ?
                      fe_face_values_quad :
                      fe_face_values_tri;

                  fe_face_values.reinit(cell, f);
                  const number factor = (cell->at_boundary(f) and
                                         not(cell->has_periodic_neighbor(f))) ?
                                          1. :
                                          0.5;
                  for (unsigned int q = 0;
                       q < fe_face_values.n_quadrature_points;
                       ++q)
                    {
                      surface_area += fe_face_values.JxW(q) * factor;
                    }
                }

              array_penalty_parameter[i][v] =
                surface_area / volume * penalty_factor;
            }
        }
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
    this->matrix_free.loop(
      &Operator::do_cell_integral_range,
      &Operator::do_face_integral_range,
      &Operator::do_boundary,
      this,
      dst,
      src,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }

  void Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  void rhs(VectorType &rhs) const
  {
    VectorType dummy;
    initialize_dof_vector(dummy);

    this->matrix_free.loop(
      &Operator::do_rhs_cell_range,
      &Operator::do_rhs_face_range,
      &Operator::do_rhs_boundary_range,
      this,
      rhs,
      dummy,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }

private:
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
        integrator.gather_evaluate(src, EvaluationFlags::gradients);

        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          integrator.submit_gradient(integrator.get_gradient(q), q);

        integrator.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  void do_face_integral_range(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceIntegrator integrator_inner(matrix_free, range, true);
    FEFaceIntegrator integrator_outer(matrix_free, range, false);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        integrator_inner.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);
        integrator_outer.reinit(face);
        integrator_outer.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);

        const VectorizedArray<number> sigma =
          std::max(integrator_inner.read_cell_data(array_penalty_parameter),
                   integrator_outer.read_cell_data(array_penalty_parameter));

        for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
          {
            {
              const auto normal = integrator_inner.normal_vector(q);

            const auto grad_inner = integrator_inner.get_gradient(q);
            const auto grad_outer = integrator_outer.get_gradient(q);

            const auto grad_inner_normal = grad_inner * normal;
            const auto grad_outer_normal = grad_outer * normal;

            const auto normal_derivative_inner =
              integrator_inner.get_normal_derivative(q);
            const auto normal_derivative_outer =
              integrator_outer.get_normal_derivative(q);
            
            bool different = false;
            for (unsigned int v = 0;  v < grad_inner_normal.size(); ++v)
            if(std::abs(grad_inner_normal[v] - normal_derivative_inner[v]) > 1e-14 || std::abs(grad_outer_normal[v] - normal_derivative_outer[v]) > 1e-14)
              different = true;            
            
            if(different)
            std::cout << "face " << face << " quadrature point " << q << " inner: "
                      << grad_inner_normal - normal_derivative_inner
                      << ", outer: "
                      << grad_outer_normal - normal_derivative_outer
                      << std::endl;
            }
            const auto normal = integrator_inner.normal_vector(q);
            const VectorizedArray<number> solution_jump =
              (integrator_inner.get_value(q) - integrator_outer.get_value(q));
            const VectorizedArray<number> averaged_normal_derivative =
              (integrator_inner.get_gradient(q) * normal +
               integrator_outer.get_gradient(q) * normal) *
              number(0.5);
            const VectorizedArray<number> test_by_value =
              solution_jump * sigma - averaged_normal_derivative;

            integrator_inner.submit_value(test_by_value, q);
            integrator_outer.submit_value(-test_by_value, q);

            integrator_inner.submit_gradient(-solution_jump * normal *
                                                        number(0.5),
                                                      q);
            integrator_outer.submit_gradient(-solution_jump * normal *
                                                        number(0.5),
                                                      q);
          }

        integrator_inner.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
        integrator_outer.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
      }
  }

  void do_boundary(const MatrixFree<dim, number>               &matrix_free,
                   VectorType                                  &dst,
                   const VectorType                            &src,
                   const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceIntegrator integrator_inner(matrix_free, range, true);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        // const auto dof_numbering =
        // integrator_inner.get_internal_dof_numbering(); for(auto & d:
        // dof_numbering) std::cout << d << std::endl;


        integrator_inner.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);

        const VectorizedArray<number> sigma =
          integrator_inner.read_cell_data(array_penalty_parameter);

        for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
          {
            {
              const auto normal = integrator_inner.normal_vector(q);

            const auto grad_inner = integrator_inner.get_gradient(q);

            const auto grad_inner_normal = grad_inner * normal;

            const auto normal_derivative_inner =
              integrator_inner.get_normal_derivative(q);
           
            
            bool different = false;
            for (unsigned int v = 0;  v < grad_inner_normal.size(); ++v)
            if(std::abs(grad_inner_normal[v] - normal_derivative_inner[v]) > 1e-14)
              different = true;            
            
            if(different)
            std::cout << "boundary face " << face << " quadrature point " << q << " inner: "
                      << grad_inner_normal - normal_derivative_inner
                      << std::endl;
            }
            const auto normal_derivative_one =
              integrator_inner.get_normal_derivative(q)[0];
            const auto normal_derivative_two =
              (integrator_inner.get_gradient(q) *
               integrator_inner.normal_vector(q))[0];

            if (false)
              if (std::abs(normal_derivative_one - normal_derivative_two) >
                  1e-15)
                {
                  const auto quad_point  = integrator_inner.quadrature_point(q);
                  const auto quad_normal = integrator_inner.normal_vector(q);
                  std::cout
                    << "MF: Face number " << face << " quadrature point " << q
                    << " at location " << quad_point[0][0] << " "
                    << quad_point[1][0] << " " << quad_point[2][0] << " "
                    << ", normal " << quad_normal[0][0] << " "
                    << quad_normal[1][0] << " " << quad_normal[2][0] << ", JxW "
                    << integrator_inner.JxW(q)[0] << std::endl;

                  std::cout << "Face penalty: " << sigma[0] << std::endl;
                  std::cout
                    << "value, grad: " << integrator_inner.get_value(q)[0]
                    << ", " << integrator_inner.get_gradient(q)[0][0] << " "
                    << integrator_inner.get_gradient(q)[1][0] << " "
                    << integrator_inner.get_gradient(q)[2][0] << ", "
                    << integrator_inner.get_normal_derivative(q)[0] << ", "
                    << (integrator_inner.get_gradient(q) * quad_normal)[0]
                    << std::endl;
                  std::cout << std::endl;
                }
            const auto normal = integrator_inner.normal_vector(q);

            const VectorizedArray<number> u_inner =
              integrator_inner.get_value(q);
            const VectorizedArray<number> u_outer = -u_inner;
            const VectorizedArray<number> normal_derivative_inner =
              integrator_inner.get_gradient(q) * normal;
            const VectorizedArray<number> normal_derivative_outer =
              normal_derivative_inner;
            const VectorizedArray<number> solution_jump = (u_inner - u_outer);
            const VectorizedArray<number> average_normal_derivative =
              (normal_derivative_inner + normal_derivative_outer) * number(0.5);
            const VectorizedArray<number> test_by_value =
              solution_jump * sigma - average_normal_derivative;

            integrator_inner.submit_gradient(-solution_jump * normal *
                                               number(0.5),
                                             q);
            integrator_inner.submit_value(test_by_value, q);
          }

        integrator_inner.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
      }
  }


  void
  do_rhs_cell_range(const MatrixFree<dim, number> &matrix_free,
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

  void do_rhs_face_range(const MatrixFree<dim, number> &,
                         VectorType &,
                         const VectorType &,
                         const std::pair<unsigned int, unsigned int> &) const
  {}

  void do_rhs_boundary_range(
    const MatrixFree<dim, number> &matrix_free,
    VectorType                    &dst,
    const VectorType &,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceIntegrator integrator_inner(matrix_free, range, true);
    Solution<dim>    solution;

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);

        const VectorizedArray<number> sigma =
          integrator_inner.read_cell_data(array_penalty_parameter);

        for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
          {
            const auto g =
              solution.value_array(integrator_inner.quadrature_point(q));

            integrator_inner.submit_gradient(
              -g * integrator_inner.normal_vector(q), q);
            integrator_inner.submit_value(2.0 * sigma * g, q);
          }

        integrator_inner.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
      }
  }

  MatrixFree<dim, number> matrix_free;

  AffineConstraints<number> constraints;

  std::vector<unsigned int> constrained_indices;

  dealii::AlignedVector<dealii::VectorizedArray<number>>
    array_penalty_parameter;
};



template <int dim, typename Number>
void do_test(const unsigned int        min_degree,
             const unsigned int        max_degree,
             const unsigned int        n_cycles_max,
             const ReferenceCell<dim> &ref_cell)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);
  pcout << "Running in " << dim << "D with degrees between " << min_degree
        << " and " << max_degree << " on " << ref_cell.to_string()
        << " elements" << std::endl;

  std::vector<ConvergenceTable> convergence_tables;
  convergence_tables.resize(2 * (max_degree - min_degree + 1));


  const FiniteElement<dim> *fe;
  const Quadrature<dim>    *quad;
  const MappingFE<dim>     *mapping;

  FE_PyramidP<dim> mapping_fe_pyramid(1, true);
  FE_WedgeP<dim>   mapping_fe_wedge(1, true);
  FE_SimplexP<dim> mapping_fe_simplex(1, true);
  FE_Q<dim>        mapping_fe_hypercube(1);

  MappingFE<dim> mapping_pyramid(mapping_fe_pyramid);
  MappingFE<dim> mapping_wedge(mapping_fe_wedge);
  MappingFE<dim> mapping_simplex(mapping_fe_simplex);
  MappingFE<dim> mapping_hypercube(mapping_fe_hypercube);

  if (ref_cell == ReferenceCells::Pyramid)
    mapping = &mapping_pyramid;
  else if (ref_cell == ReferenceCells::Wedge)
    mapping = &mapping_wedge;
  else if (ref_cell.is_simplex())
    mapping = &mapping_simplex;
  else if (ref_cell.is_hyper_cube())
    mapping = &mapping_hypercube;
  else
    DEAL_II_NOT_IMPLEMENTED();

  AffineConstraints<double> constraint;
  const unsigned int        n_cells_max = 200000000;
  unsigned int              n_cells     = 1;

  const unsigned int n_dofs_max = 33000000;
  unsigned int       n_dofs     = 1;

  for (unsigned int cycle = 0; cycle < n_cycles_max && n_cells < n_cells_max;
       ++cycle)
    {
      const auto serial_grid_generator = [&cycle, &ref_cell](
                                           dealii::Triangulation<dim, dim>
                                             &tria_serial) {
        // set up triangulation
        if (ref_cell == ReferenceCells::Pyramid)
          GridGenerator::subdivided_hyper_cube_with_pyramids(tria_serial,
                                                             std::pow(2,
                                                                      cycle));
        else if (ref_cell == ReferenceCells::Wedge)
          {
            dealii::Triangulation<dim, dim> temp;
            // GridGenerator::subdivided_hyper_cube_with_wedges(tria_serial, 2);
            //  GridGenerator::subdivided_hyper_cube_with_wedges(tria_serial,
            //                                                 std::pow(2,
            //                                                 cycle));
            if (true)
              {
                std::vector<Point<dim>>    vertices;
                std::vector<CellData<dim>> cells;
                vertices.emplace_back(0.0, 0.0, 0.0);
                vertices.emplace_back(1.0, 0.0, 0.0);
                vertices.emplace_back(0.0, 1.0, 0.0);
                vertices.emplace_back(0.0, 0.0, 1.0);
                vertices.emplace_back(1.0, 0.0, 1.0);
                vertices.emplace_back(0.0, 1.0, 1.0);

                vertices.emplace_back(1.0, 1.0, 0.0);
                vertices.emplace_back(1.0, 1.0, 1.0);

                {
                  CellData<dim> wedge;
                  wedge.vertices = {0, 1, 2, 3, 4, 5};
                  cells.push_back(wedge);
                }
                if (false)
                  {
                    CellData<dim> wedge;
                    wedge.vertices = {1, 6, 2, 4, 7, 5};
                    cells.push_back(wedge);
                  }

                temp.create_triangulation(vertices, cells, SubCellData());

                if (cycle > 0)
                  temp.refine_global(cycle);

                const auto                &new_vertices = temp.get_vertices();
                std::vector<CellData<dim>> new_cells;
                for (auto &cell : temp.active_cell_iterators())
                  {
                    const auto         reference_cell = cell->reference_cell();
                    const unsigned int n_vertices = reference_cell.n_vertices();

                    CellData<dim> wedge;
                    wedge.vertices.resize(n_vertices);

                    for (unsigned int i = 0; i < n_vertices; ++i)
                      {
                        wedge.vertices[i] = cell->vertex_index(i);
                      }
                    new_cells.push_back(wedge);
                  }
                tria_serial.create_triangulation(new_vertices,
                                                 new_cells,
                                                 SubCellData());
              }
          }
        else if (ref_cell.is_simplex())
          GridGenerator::subdivided_hyper_cube_with_simplices(tria_serial, 2);
        else if (ref_cell.is_hyper_cube())
          GridGenerator::subdivided_hyper_cube(tria_serial, 2);
        else
          DEAL_II_NOT_IMPLEMENTED();

        if (ref_cell != ReferenceCells::Pyramid &&
            ref_cell != ReferenceCells::Wedge)
          tria_serial.refine_global(cycle);


        {
          for (const auto &cell : tria_serial.active_cell_iterators())
            {
              for (const auto f : cell->face_indices())
                {
                  if (cell->face(f)->at_boundary())
                    {
                      const auto face_orientation =
                        cell->combined_face_orientation(f);

                      if (face_orientation !=
                          numbers::default_geometric_orientation)
                        std::cout
                          << "boundary face in non default orientation in cycle "
                          << cycle << " with orientation "
                          << int(face_orientation) << std::endl;
                      //else
                      //  std::cout << "boundary face in standard orientation "
                       //           << int(face_orientation) << std::endl;
                    }
                  else
                    {
                      const auto face_orientation =
                        cell->combined_face_orientation(f);

                      const auto neighbor = cell->neighbor(f);
                      const auto neighbor_face_number =
                        cell->neighbor_face_no(f);

                      const auto face_orientation_neighbor =
                        neighbor->combined_face_orientation(
                          neighbor_face_number);

                      if (face_orientation ==
                            numbers::default_geometric_orientation ||
                          face_orientation_neighbor ==
                            numbers::default_geometric_orientation)
                        {
                          // std::cout << "Face with orientations: "
                          //           << int(face_orientation) << " "
                          //           << int(face_orientation_neighbor)
                          //           << std::endl;
                        }
                      else
                        {
                          std::cout
                            << "face with 2 non standard sides in cycle "
                            << cycle << " with orientations "
                            << int(face_orientation) << " and "
                            << int(face_orientation_neighbor) << std::endl;
                        }
                    }
                }
            }
        }
      };
      const auto serial_grid_partitioner =
        [&](dealii::Triangulation<dim, dim> &tria_serial,
            const MPI_Comm                   comm,
            const unsigned int) {
          dealii::GridTools::partition_triangulation_zorder(
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

      bool continue_iterating = true;
      for (unsigned int fe_degree = min_degree;
           fe_degree <= max_degree && n_dofs < n_dofs_max && continue_iterating;
           ++fe_degree)
        {
          for (const bool use_equidistant_points :
               std::vector<bool>{{true, false}})
            {
              DoFHandler<dim> dof_handler(tria);

              FE_PyramidDGP<dim> fe_pyramidp(fe_degree, use_equidistant_points);
              FE_WedgeDGP<dim>   fe_wedgep(fe_degree, use_equidistant_points);
              FE_SimplexDGP<dim> fe_simplexp(fe_degree, use_equidistant_points);
              FE_DGQ<dim>        fe_q = use_equidistant_points ?
                                          FE_DGQArbitraryNodes<dim>(
                                     QIterated<1>(QTrapezoid<1>(), fe_degree)) :
                                          FE_DGQ<dim>(fe_degree);

              if (ref_cell == ReferenceCells::Pyramid)
                fe = &fe_pyramidp;
              else if (ref_cell == ReferenceCells::Wedge)
                fe = &fe_wedgep;
              else if (ref_cell.is_simplex())
                fe = &fe_simplexp;
              else if (ref_cell.is_hyper_cube())
                fe = &fe_q;
              else
                DEAL_II_NOT_IMPLEMENTED();

              dof_handler.distribute_dofs(*fe);

              // set up constraints
              const IndexSet locally_relevant_dofs =
                DoFTools::extract_locally_relevant_dofs(dof_handler);
              constraint.reinit(dof_handler.locally_owned_dofs(),
                                locally_relevant_dofs);
              constraint.close();

              QGaussPyramid<dim> quad_pyramid(fe_degree + 1);
              QGaussWedge<dim>   quad_wedge(fe_degree + 1);
              QGaussSimplex<dim> quad_simplex(fe_degree + 1);
              QGauss<dim>        quad_hypercube(fe_degree + 1);

              if (ref_cell == ReferenceCells::Pyramid)
                quad = &quad_pyramid;
              else if (ref_cell == ReferenceCells::Wedge)
                quad = &quad_wedge;
              else if (ref_cell.is_simplex())
                quad = &quad_simplex;
              else if (ref_cell.is_hyper_cube())
                quad = &quad_hypercube;
              else
                DEAL_II_NOT_IMPLEMENTED();

              pcout << "Set up operator of degree " << fe_degree << std::endl;
              Operator<dim, 1, Number> op;
              // set up operator
              op.reinit(*mapping,
                        dof_handler,
                        *quad,
                        constraint,
                        numbers::invalid_unsigned_int);


              pcout << "Set up system matrix" << std::endl;
              PoissonProblem<dim, Number> matrix(*mapping,
                                                 dof_handler,
                                                 *quad,
                                                 constraint);

              FullMatrix<Number> full_system_matrix;
              full_system_matrix.copy_from(matrix.system_matrix);

              const unsigned int n_current_dofs = dof_handler.n_dofs();
              Assert(n_current_dofs == full_system_matrix.m() &&
                       n_current_dofs == full_system_matrix.n(),
                     ExcInternalError());

              // std::cout << "System matrix" << std::endl;
              // full_system_matrix.print(std::cout, 10, 3);

              FullMatrix<Number> full_system_matrix_mf(n_current_dofs,
                                                       n_current_dofs);
              {
                Vector<Number> e(n_current_dofs);
                Vector<Number> r(n_current_dofs);
                for (unsigned int i = 0; i < n_current_dofs; ++i)
                  {
                    e    = 0.0;
                    e[i] = 1.0;

                    r = 0.;
                    //std::cout << "basis vector " << i << std::endl;
                    op.vmult(r, e);

                    for (unsigned int j = 0; j < n_current_dofs; ++j)
                      {
                        full_system_matrix_mf[j][i] = r[j];
                      }
                  }
              }
              //std::cout << std::endl;

              //std::cout << "System matrix (mf)" << std::endl;
              //full_system_matrix_mf.print(std::cout, 10, 3);

              FullMatrix<Number> full_system_matrix_diff(n_current_dofs,
                                                         n_current_dofs);
              for (unsigned int i = 0; i < full_system_matrix_diff.m(); ++i)
                for (unsigned int j = 0; j < full_system_matrix_diff.n(); ++j)
                  full_system_matrix_diff[i][j] =
                    full_system_matrix[i][j] - full_system_matrix_mf[i][j];

              std::cout << "Difference" << std::endl;
              for (unsigned int i = 0; i < full_system_matrix_diff.m(); ++i)
                for (unsigned int j = 0; j < full_system_matrix_diff.n(); ++j)
                 if (std::abs(full_system_matrix_diff[i][j])>1e-14)
                  std::cout << i << " " << j << ": " << full_system_matrix_diff[i][j] << ", ";
                std::cout << std::endl;
              //full_system_matrix_diff.print(std::cout, 10, 3);

              Vector<Number> x, rhs, x_mb, rhs_mb, random_vector;
              op.initialize_dof_vector(x);
              op.initialize_dof_vector(rhs);
              op.initialize_dof_vector(random_vector);
              op.rhs(rhs);

              for (Number &a : random_vector)
                a = static_cast<double>(rand()) / RAND_MAX;

              rhs_mb = matrix.system_rhs;
              x_mb.reinit(dof_handler.n_dofs());

              rhs_mb -= rhs;
              pcout << "Check rhs: " << rhs_mb.l2_norm() << std::endl;
              rhs_mb = matrix.system_rhs;

              x = 0.;
              op.vmult(x, random_vector);

              x_mb = 0.;
              matrix.vmult(x_mb, random_vector);
              x_mb -= x;

              pcout << "Check vmult: " << x_mb.l2_norm() << std::endl;

              x    = 0.;
              x_mb = 0.;

              ReductionControl         reduction_control(dof_handler.n_dofs(),
                                                 1e-12,
                                                 1e-12);
              SolverCG<Vector<Number>> solver(reduction_control);
              PreconditionIdentity     preconditioner;

              solver.solve(op, x, rhs, preconditioner);

              pcout << "Solved MF in " << reduction_control.last_step()
                    << " iterations with final residual "
                    << std::setprecision(16) << reduction_control.last_value()
                    << std::endl;


              solver.solve(matrix, x_mb, rhs_mb, preconditioner);

              pcout << "Solved SpMV in " << reduction_control.last_step()
                    << " iterations with final residual "
                    << std::setprecision(16) << reduction_control.last_value()
                    << std::endl;

              x_mb -= x;
              pcout << "Difference in solution: " << x_mb.l2_norm()
                    << std::endl;
              x.update_ghost_values();
              Vector<double> difference_per_cell;
              VectorTools::integrate_difference(
                *mapping,
                dof_handler,
                x,
                Solution<dim>(),
                difference_per_cell,
                fe->reference_cell().get_gauss_type_quadrature(
                  // std::max(int(1.5 * fe->degree) + 3, int(fe->degree + 5))),
                  fe->degree + 3),
                VectorTools::L2_norm);

              const double L2_error =
                VectorTools::compute_global_error(tria,
                                                  difference_per_cell,
                                                  VectorTools::L2_norm);

              if (L2_error < 1e-10)
                continue_iterating = false;

              const unsigned int n_active_cells = tria.n_global_active_cells();
              n_dofs                            = dof_handler.n_dofs();

              if (use_equidistant_points)
                pcout << "Cycle " << cycle << ':' << std::endl
                      << fe->get_name() << " equidistant" << std::endl
                      << "   Number of active cells:       " << n_active_cells
                      << std::endl
                      << "   Number of degrees of freedom: " << n_dofs
                      << std::endl
                      << "   L2 error:                     " << L2_error
                      << std::endl;
              else
                pcout << "Cycle " << cycle << ':' << std::endl
                      << fe->get_name() << " blend and warp" << std::endl
                      << "   Number of active cells:       " << n_active_cells
                      << std::endl
                      << "   Number of degrees of freedom: " << n_dofs
                      << std::endl
                      << "   L2 error:                     " << L2_error
                      << std::endl;

              if (fe->degree > 1)
                {
                  const unsigned int index =
                    fe->reference_cell().is_hyper_cube() ?
                      1 :
                      fe->reference_cell().n_vertices();
                  pcout << "First line support point "
                        << fe->unit_support_point(index) << std::endl;
                }


              unsigned int offset = fe->degree - min_degree;
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
  if (false)
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

            std::string error_filename = "error_DG_MF_";
            error_filename +=
              ref_cell.to_string() + "_p_" + std::to_string(degree_counter);
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

  int min_degree     = 1;
  int max_degree     = 7;
  int n_cycles_max   = 7;
  int reference_cell = 0;
  if (argc > 1)
    min_degree = std::atoi(argv[1]);
  if (argc > 2)
    max_degree = std::atoi(argv[2]);
  if (argc > 3)
    n_cycles_max = std::atoi(argv[3]);
  if (argc > 4)
    reference_cell = std::atoi(argv[4]);


  if (false)
    {
      for (unsigned int cycle = 0; cycle < 6; ++cycle)
        {
          std::cout << "Running cycle " << cycle << std::endl;
          // set up triangulation
          dealii::Triangulation<dim, dim> tria;
          GridGenerator::subdivided_hyper_cube_with_wedges(tria,
                                                           std::pow(2, cycle));
          // GridGenerator::subdivided_hyper_cube_with_pyramids(tria, 2);
          if (cycle > 0)
            tria.refine_global(cycle);

          for (const auto &cell : tria.active_cell_iterators())
            if (cell->reference_cell().is_simplex() == false)
              {
                for (const auto f : cell->face_indices())
                  {
                    if (cell->face(f)->at_boundary())
                      {
                        const auto face_orientation =
                          cell->combined_face_orientation(f);

                        if (face_orientation !=
                            numbers::default_geometric_orientation)
                          std::cout
                            << "boundary face in non default orientation in cycle "
                            << cycle << " with orientation "
                            << int(face_orientation) << std::endl;
                      }
                    else if (false)
                      {
                        const auto face_orientation =
                          cell->combined_face_orientation(f);

                        const auto neighbor = cell->neighbor(f);
                        const auto neighbor_face_number =
                          cell->neighbor_face_no(f);

                        const auto face_orientation_neighbor =
                          neighbor->combined_face_orientation(
                            neighbor_face_number);

                        if (face_orientation ==
                              numbers::default_geometric_orientation ||
                            face_orientation_neighbor ==
                              numbers::default_geometric_orientation)
                          {
                          }
                        else
                          {
                            std::cout
                              << "face with 2 non standard sides in cycle "
                              << cycle << " with orientations "
                              << int(face_orientation) << " and "
                              << int(face_orientation_neighbor) << std::endl;
                          }
                      }
                  }
              }
          std::cout << "else all correct" << std::endl;
        }
      return 1;
    }

  {
    if (reference_cell == 0)
      {
        do_test<dim, double>(min_degree,
                             max_degree,
                             n_cycles_max,
                             ReferenceCells::Pyramid);
        std::cout << std::endl;
      }

    if (reference_cell == 1)

      {
        do_test<dim, double>(min_degree,
                             max_degree,
                             n_cycles_max,
                             ReferenceCells::Wedge);
        std::cout << std::endl;
      }

    if (reference_cell == 2)

      {
        do_test<dim, double>(min_degree,
                             max_degree,
                             n_cycles_max,
                             ReferenceCells::Tetrahedron);
        std::cout << std::endl;
      }

    if (reference_cell == 3)

      {
        do_test<dim, double>(min_degree,
                             max_degree,
                             n_cycles_max,
                             ReferenceCells::Hexahedron);
        std::cout << std::endl;
      }
  }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
