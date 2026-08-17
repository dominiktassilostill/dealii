#include "./../../../tests/simplex/simplex_grids.h"

#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/numerics/data_out.h>

#define FORCE_USE_OF_TRILINOS
namespace LA
{
#if defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/base/conditional_ostream.h>
#include <array>
#include <fstream>
#include <iostream>

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
};

template <int dim>
class PoissonProblem
{
public:
  PoissonProblem(const unsigned int        min_degree,
                 const unsigned int        max_degree,
                 const ReferenceCell<dim> &ref_cell);

  void run(const unsigned int n_cycles_max,
           const double       error_threshold,
           const bool         only_run_blend_and_warp);

private:
  void   setup_system();
  void   assemble_system();
  void   solve();
  bool   process_solution(const unsigned int cycle,
                          const bool         use_equidistant_points,
                          const bool         only_run_blend_and_warp,
                          const double       error_threshold);
  double compute_l2_error();
  void   compute_difference_at_nodes();

  MPI_Comm                                       mpi_communicator;
  parallel::fullydistributed::Triangulation<dim> triangulation;


  DoFHandler<dim> dof_handler;

  const FiniteElement<dim> *fe;
  const MappingFE<dim>     *mapping;

  AffineConstraints<double> constraints;

  LA::MPI::SparseMatrix system_matrix;
  LA::MPI::Vector       solution;
  LA::MPI::Vector       system_rhs;

  ConditionalOStream pcout;

  std::vector<ConvergenceTable> convergence_tables;

  unsigned int       min_degree;
  unsigned int       max_degree;
  ReferenceCell<dim> ref_cell;
};


template <int dim>
PoissonProblem<dim>::PoissonProblem(const unsigned int        min_degree,
                                    const unsigned int        max_degree,
                                    const ReferenceCell<dim> &ref_cell)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , dof_handler(triangulation)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , min_degree(min_degree)
  , max_degree(max_degree)
  , ref_cell(ref_cell)
{
  pcout << "Running with " << ref_cell.to_string() << " elements" << std::endl;

  convergence_tables.resize(2 * (max_degree - min_degree + 1));
}


template <int dim>
void PoissonProblem<dim>::setup_system()
{
  dof_handler.clear();
  system_matrix.clear();
  solution.clear();
  system_rhs.clear();

  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(*fe);

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       mpi_communicator);
}


template <int dim>
void PoissonProblem<dim>::assemble_system()
{
  FEValues<dim> fe_values(*mapping,
                          *fe,
                          fe->reference_cell().get_gauss_type_quadrature(
                            fe->degree + 1),
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);


  const QGauss<dim - 1>        quadrature_quad(fe->degree + 1);
  const QGaussSimplex<dim - 1> quadrature_tri(fe->degree + 1);

  FEFaceValues<dim> fe_face_values_quad(*mapping,
                                        *fe,
                                        quadrature_quad,
                                        update_values | update_gradients |
                                          update_quadrature_points |
                                          update_normal_vectors |
                                          update_JxW_values);

  FEFaceValues<dim> fe_face_values_tri(*mapping,
                                       *fe,
                                       quadrature_tri,
                                       update_values | update_gradients |
                                         update_quadrature_points |
                                         update_normal_vectors |
                                         update_JxW_values);


  FEFaceValues<dim> fe_neighbor_face_values_quad(*mapping,
                                                 *fe,
                                                 quadrature_quad,
                                                 update_values |
                                                   update_gradients |
                                                   update_quadrature_points |
                                                   update_normal_vectors |
                                                   update_JxW_values);

  FEFaceValues<dim> fe_neighbor_face_values_tri(*mapping,
                                                *fe,
                                                quadrature_tri,
                                                update_values |
                                                  update_gradients |
                                                  update_quadrature_points |
                                                  update_normal_vectors |
                                                  update_JxW_values);

  const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
  const unsigned int n_q_points    = fe_values.n_quadrature_points;

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_neighbor_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> neighbor_dof_indices(dofs_per_cell);

  RightHandSide<dim> rhs;
  Solution<dim>      exact_solution;

  const double penalty_factor =
    1.0 * (fe->degree + 1) * (fe->degree + dim) / double(dim);
  double              penalty_parameter_minus = 0.0;
  std::vector<double> penalty_parameter_plus(fe->reference_cell().n_faces(),
                                             0.0);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        // compute the penalty parameter for the interior
        {
          fe_values.reinit(cell);
          // calculate cell volume
          double volume = 0.0;
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              volume += fe_values.JxW(q);
            }

          // calculate surface area
          double surface_area = 0.0;
          for (const unsigned int f : cell->face_indices())
            {
              auto &fe_face_values =
                cell->face(f)->reference_cell().is_hyper_cube() ?
                  fe_face_values_quad :
                  fe_face_values_tri;

              fe_face_values.reinit(cell, f);
              const double factor =
                (cell->at_boundary(f) and not(cell->has_periodic_neighbor(f))) ?
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
                  double volume = 0.0;
                  for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                      volume += fe_values.JxW(q);
                    }

                  // calculate surface area
                  double surface_area = 0.0;
                  for (const unsigned int f : neighbor->face_indices())
                    {
                      auto &fe_face_values =
                        neighbor->face(f)->reference_cell().is_hyper_cube() ?
                          fe_face_values_quad :
                          fe_face_values_tri;

                      fe_face_values.reinit(neighbor, f);
                      const double factor =
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
              const double rhs_value =
                rhs.value(fe_values.quadrature_point(q_point), 0);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) += (fe_values.shape_grad(i, q_point) *
                                      fe_values.shape_grad(j, q_point) *
                                      fe_values.JxW(q_point)); // dx

              cell_rhs(i) += (fe_values.shape_value(i, q_point) * // phi_i(x_q)
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
                const double penalty = penalty_parameter_minus;
                for (unsigned int q = 0; q < n_face_q; ++q)
                  {
                    const Tensor<1, dim> normal =
                      fe_face_values.normal_vector(q);

                    const double g =
                      exact_solution.value(fe_face_values.quadrature_point(q),
                                           0);

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        const double phi_i = fe_face_values.shape_value(i, q);
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
                               penalty * phi_j * phi_i) *
                              fe_face_values.JxW(q);
                          }

                        cell_rhs(i) +=
                          ((-(grad_phi_i * normal) * g) + penalty * g * phi_i) *
                          fe_face_values.JxW(q);
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

                const double penalty = std::max(penalty_parameter_plus[face_no],
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
                               0.5 * (grad_phi_i_minus * normal) * phi_j_minus +
                               penalty * phi_j_minus * phi_i_minus) *
                              fe_face_values.JxW(q);

                            const double phi_j_plus =
                              fe_neighbor_face_values.shape_value(j, q);
                            const Tensor<1, dim> grad_phi_j_plus =
                              fe_neighbor_face_values.shape_grad(j, q);

                            cell_neighbor_matrix(i, j) +=
                              (-0.5 * (grad_phi_j_plus * normal) * phi_i_minus +
                               0.5 * (grad_phi_i_minus * normal) * phi_j_plus -
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

        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}


template <int dim>
void PoissonProblem<dim>::solve()
{
  // SolverControl solver_control(dof_handler.n_dofs(),
  //                              1e-6 * system_rhs.l2_norm());

  ReductionControl solver_control(dof_handler.n_dofs(), 1e-12, 1e-12);
  LA::SolverCG     solver(solver_control);

  LA::MPI::Vector completely_distributed_solution(
    dof_handler.locally_owned_dofs(), mpi_communicator);


  LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
  data.symmetric_operator = true;
#else
/* Trilinos defaults are good */
#endif
  //  LA::MPI::PreconditionAMG preconditioner;
  // preconditioner.initialize(system_matrix, data);

  TrilinosWrappers::PreconditionIdentity preconditioner;
  preconditioner.initialize(system_matrix);

  solver.solve(system_matrix,
               completely_distributed_solution,
               system_rhs,
               preconditioner);

  pcout << "Solved in " << solver_control.last_step()
        << " iterations with final residual " << std::setprecision(16)
        << solver_control.last_value() << std::endl;

  constraints.distribute(completely_distributed_solution);
  solution = completely_distributed_solution;
}


template <int dim>
bool PoissonProblem<dim>::process_solution(const unsigned int cycle,
                                           const bool   use_equidistant_points,
                                           const bool   only_run_blend_and_warp,
                                           const double error_threshold)
{
  Vector<double> difference_per_cell;

  Assert(dof_handler.get_fe().get_name() == fe->get_name(), ExcInternalError());

  if (fe->degree > 1)
    {
      const unsigned int index = fe->reference_cell().is_hyper_cube() ?
                                   1 :
                                   fe->reference_cell().n_vertices();
      Assert(dof_handler.get_fe().unit_support_point(index) ==
               fe->unit_support_point(index),
             ExcInternalError());
    }
  VectorTools::integrate_difference(
    *mapping,
    dof_handler,
    solution,
    Solution<dim>(),
    difference_per_cell,
    fe->reference_cell().get_gauss_type_quadrature(
      // std::max(int(1.5 * fe->degree) + 3, int(fe->degree + 5))),
      fe->degree + 3),
    VectorTools::L2_norm);

  const double L2_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);

  const unsigned int n_active_cells = triangulation.n_global_active_cells();
  const unsigned int n_dofs         = dof_handler.n_dofs();

  if (use_equidistant_points)
    pcout << "Cycle " << cycle << ':' << std::endl
          << fe->get_name() << " equidistant" << std::endl
          << "   Number of active cells:       " << n_active_cells << std::endl
          << "   Number of degrees of freedom: " << n_dofs << std::endl
          << "   L2 error:                     " << L2_error << std::endl;
  else
    pcout << "Cycle " << cycle << ':' << std::endl
          << fe->get_name() << " blend and warp" << std::endl
          << "   Number of active cells:       " << n_active_cells << std::endl
          << "   Number of degrees of freedom: " << n_dofs << std::endl
          << "   L2 error:                     " << L2_error << std::endl;
  // const double l2_manual = compute_l2_error();
  // pcout << "   Difference in L2 error: " << L2_error - l2_manual <<
  // std::endl;
  // compute_difference_at_nodes();

  if (fe->degree > 1)
    {
      const unsigned int index = fe->reference_cell().is_hyper_cube() ?
                                   1 :
                                   fe->reference_cell().n_vertices();
      pcout << "First line support point " << fe->unit_support_point(index)
            << std::endl;
    }


  unsigned int offset = fe->degree - min_degree;
  if (!use_equidistant_points)
    if (!only_run_blend_and_warp)
      offset += max_degree - min_degree + 1;

  convergence_tables[offset].add_value("cycle", cycle);
  convergence_tables[offset].add_value("cells", n_active_cells);
  convergence_tables[offset].add_value("dofs", n_dofs);
  convergence_tables[offset].add_value("L2", L2_error);

  return L2_error < error_threshold;
}


template <int dim>
double PoissonProblem<dim>::compute_l2_error()
{
  double local_error_squared = 0.0;

  FEValues<dim> fe_values(*mapping,
                          *fe,
                          fe->reference_cell().get_gauss_type_quadrature(
                            std::max<unsigned int>(3 * fe->degree + 5, 20)),
                          update_values | update_quadrature_points |
                            update_JxW_values);

  std::vector<double> numerical_values(fe_values.n_quadrature_points);

  Solution<dim> exact_solution;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(solution, numerical_values);

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          {
            const double difference =
              numerical_values[q] -
              exact_solution.value(fe_values.quadrature_point(q), 0);

            local_error_squared += difference * difference * fe_values.JxW(q);
          }
      }

  const double global_error_squared =
    Utilities::MPI::sum(local_error_squared, mpi_communicator);

  const double manual_L2 = std::sqrt(global_error_squared);

  pcout << "   manual L2       " << std::setprecision(16) << manual_L2
        << std::endl;

  return manual_L2;
}



template <int dim>
void PoissonProblem<dim>::compute_difference_at_nodes()
{
  double          local_difference_squared = 0.0;
  Quadrature<dim> quadrature(fe->get_unit_support_points());

  FEValues<dim> fe_values(*mapping,
                          *fe,
                          quadrature,
                          update_values | update_quadrature_points);

  std::vector<double> numerical_values(fe_values.n_quadrature_points);

  Solution<dim> exact_solution;

  std::vector<types::global_dof_index> local_dof_indices(fe->n_dofs_per_cell());


  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(solution, numerical_values);


        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
          {
            const double coeff = solution(local_dof_indices[i]);

            const double nodal_value = numerical_values[i];

            const double diff = coeff - nodal_value;
            local_difference_squared += diff * diff;
          }
      }

  const double global_difference_squared =
    Utilities::MPI::sum(local_difference_squared, mpi_communicator);

  const double global_difference = std::sqrt(global_difference_squared);

  pcout << "   nodal difference       " << std::setprecision(16)
        << global_difference << std::endl;
}



template <int dim>
void PoissonProblem<dim>::run(const unsigned int n_cycles_max,
                              const double       error_threshold,
                              const bool         only_run_blend_and_warp)
{
  if (only_run_blend_and_warp)
    {
      convergence_tables.clear();
      convergence_tables.resize(max_degree - min_degree + 1);
    }

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

  const unsigned int n_cells_max = 200000000;
  unsigned int       n_cells     = 1;

  const unsigned int n_dofs_max = 33000000;
  unsigned int       n_dofs     = 1;

  std::vector<bool> reached_error_threshold(max_degree - min_degree + 1, false);

  for (unsigned int cycle = 0; n_cells < n_cells_max && cycle < n_cycles_max;
       ++cycle)
    {
      {
        triangulation.clear();

        const auto serial_grid_generator =
          [&](dealii::Triangulation<dim, dim> &tria_serial) {
            // set up triangulation
            if (ref_cell == ReferenceCells::Pyramid)
              GridGenerator::subdivided_hyper_cube_with_pyramids(
                tria_serial, std::pow(2, cycle));
            else if (ref_cell == ReferenceCells::Wedge)
              GridGenerator::subdivided_hyper_cube_with_wedges(tria_serial, 2);
            else if (ref_cell.is_simplex())
              GridGenerator::subdivided_hyper_cube_with_simplices(tria_serial,
                                                                  2);
            else if (ref_cell.is_hyper_cube())
              GridGenerator::subdivided_hyper_cube(tria_serial, 2);
            else
              DEAL_II_NOT_IMPLEMENTED();

            if (ref_cell != ReferenceCells::Pyramid)
              tria_serial.refine_global(cycle);
          };
        const auto serial_grid_partitioner =
          [&](dealii::Triangulation<dim, dim> &tria_serial,
              const MPI_Comm                   comm,
              const unsigned int) {
            dealii::GridTools::partition_triangulation_zorder(
              dealii::Utilities::MPI::n_mpi_processes(comm), tria_serial);
          };

        const unsigned int group_size = 32;

        typename dealii::TriangulationDescription::Settings
          triangulation_description_setting =
            dealii::TriangulationDescription::default_setting;
        const auto description = dealii::TriangulationDescription::Utilities::
          create_description_from_triangulation_in_groups<dim, dim>(
            serial_grid_generator,
            serial_grid_partitioner,
            triangulation.get_mpi_communicator(),
            group_size,
            dealii::Triangulation<dim>::none,
            triangulation_description_setting);

        triangulation.create_triangulation(description);
        pcout << "Cycle " << cycle << " set up triangulation" << std::endl;
      }

      // we got the triangulation
      // now go over all polynomial degrees
      // and over equidistant and gl support points if needed
      const std::vector<bool> variants_vector =
        only_run_blend_and_warp ? std::vector<bool>{{false}} :
                                  std::vector<bool>{{true, false}};

      unsigned int max_degree_cycle = max_degree;
      if (ref_cell == ReferenceCells::Pyramid)
        {
          if (cycle > 3)
            max_degree_cycle = std::min(4U, max_degree);
          if (cycle > 4)
            max_degree_cycle = std::min(3U, max_degree);
        }
      else if (ref_cell == ReferenceCells::Wedge)
        {
          if (cycle > 3)
            max_degree_cycle = std::min(4U, max_degree);
          if (cycle > 4)
            max_degree_cycle = std::min(3U, max_degree);
        }
      else if (ref_cell.is_simplex())
        {
          if (cycle > 3)
            max_degree_cycle = std::min(4U, max_degree);
          if (cycle > 4)
            max_degree_cycle = std::min(3U, max_degree);
        }
      else if (ref_cell.is_hyper_cube())
        {
          if (cycle > 2)
            max_degree_cycle = std::min(5U, max_degree);
          if (cycle > 3)
            max_degree_cycle = std::min(4U, max_degree);
          if (cycle > 4)
            max_degree_cycle = std::min(3U, max_degree);
        }
      else
        DEAL_II_NOT_IMPLEMENTED();

      for (unsigned int degree = min_degree;
           degree <= max_degree_cycle && n_dofs < n_dofs_max &&
           !reached_error_threshold[degree - min_degree];
           ++degree)
        {
          for (const bool use_equidistant_points : variants_vector)
            {
              FE_PyramidDGP<dim> fe_pyramidp(degree, use_equidistant_points);
              FE_WedgeDGP<dim>   fe_wedgep(degree, use_equidistant_points);
              FE_SimplexDGP<dim> fe_simplexp(degree, use_equidistant_points);
              FE_DGQ<dim>        fe_q = use_equidistant_points ?
                                          FE_DGQArbitraryNodes<dim>(
                                     QIterated<1>(QTrapezoid<1>(), degree)) :
                                          FE_DGQ<dim>(degree);

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

              pcout << "Setup system" << std::endl;
              setup_system();
              pcout << "Assemble system" << std::endl;
              assemble_system();
              pcout << "Solve system" << std::endl;
              solve();
              reached_error_threshold[degree - min_degree] =
                process_solution(cycle,
                                 use_equidistant_points,
                                 only_run_blend_and_warp,
                                 error_threshold);
              pcout << std::endl;

              if (false)
                {
                  DataOut<dim> data_out;

                  DataOutBase::VtkFlags flags;
                  flags.write_higher_order_cells = false;
                  data_out.set_flags(flags);

                  data_out.add_data_vector(dof_handler, solution, "solution");
                  Vector<double> mpi_owner(triangulation.n_active_cells());
                  mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
                  data_out.add_data_vector(mpi_owner, "owner");
                  data_out.build_patches(*mapping,
                                         1,
                                         DataOut<dim>::curved_inner_cells);

                  const std::string filename = "solution-" + fe->get_name();
                  data_out.write_vtu_with_pvtu_record("",
                                                      filename,
                                                      cycle,
                                                      MPI_COMM_WORLD);
                }
              n_dofs = dof_handler.n_dofs();
            }
          pcout << std::endl;
        }
      pcout << std::endl;

      n_dofs  = 0;
      n_cells = triangulation.n_global_active_cells();
    }

  //    std::string vtk_filename;
  //  vtk_filename = "solution_";
  //  vtk_filename += fe->get_name() + "_cycle_" + std::to_string(cycle);
  //  vtk_filename += ".vtk";
  //  std::ofstream output(vtk_filename);

  //  DataOut<dim> data_out;
  //  data_out.attach_dof_handler(dof_handler);
  //  data_out.add_data_vector(solution, "solution");
  //  data_out.build_patches(1);
  //  data_out.write_vtk(output);

  unsigned int degree_counter  = min_degree;
  bool         use_equi_points = true;
  if (only_run_blend_and_warp)
    use_equi_points = false;
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

      pcout << std::endl;

      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          convergence_table.write_text(std::cout);

          std::string error_filename = "error_DG_SpMV_";
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
  const unsigned int                       dim = 3;
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);


  if (false)
    {
      for (unsigned int i = 1; i < 5; ++i)
        {
          // FE_SimplexP<dim> fe_equi(i, true);
          // FE_SimplexP<dim> fe_blend_and_warp(i, false);
          FE_DGQArbitraryNodes<dim> fe_equi(QIterated<1>(QTrapezoid<1>(), i));
          FE_DGQ<dim>               fe_blend_and_warp(i);

          const auto points_equi  = fe_equi.get_unit_support_points();
          const auto points_b_a_w = fe_blend_and_warp.get_unit_support_points();

          if (true)
            {
              std::cout << "Unit support points p=" << i << " (equidistant)"
                        << std::endl;
              for (const auto &p : points_equi)
                std::cout << p << std::endl;
              std::cout << std::endl;
            }

          if (true)
            {
              std::cout << "Unit support points p=" << i << " (blend and warp)"
                        << std::endl;
              for (const auto &p : points_b_a_w)
                std::cout << p << std::endl;
              std::cout << std::endl;
            }

          if (false)
            {
              const auto ref_cell = fe_blend_and_warp.reference_cell();
              for (const auto &p : points_b_a_w)
                {
                  if (!ref_cell.contains_point(p, 1e-16))
                    {
                      std::cout
                        << "!!!!!!!!!!!!!!!!!!!!!!!!!!!1 invalid point: ";
                      std::cout << p << std::endl;
                    }
                }
            }

          if (false)
            {
              std::cout << "Difference between support points at p=" << i
                        << std::endl;
              for (unsigned int i = 0; i < points_b_a_w.size(); ++i)
                std::cout << points_equi[i] - points_b_a_w[i] << std::endl;
            }
          std::cout << std::endl;
          std::cout << std::endl;
        }
      return 0;
    }


  try
    {
      const unsigned int min_degree      = 1;
      const unsigned int max_degree      = 7;
      const unsigned int n_cycles_max    = 7;
      const double       error_threshold = 1e-9;

      if (false)
        {
          PoissonProblem<dim> poisson(min_degree,
                                      max_degree,
                                      ReferenceCells::Pyramid);
          poisson.run(n_cycles_max, error_threshold, false);
        }

      if (false)
        {
          PoissonProblem<dim> poisson(min_degree,
                                      max_degree,
                                      ReferenceCells::Wedge);
          poisson.run(n_cycles_max, error_threshold, false);
        }

      if (false)
        {
          PoissonProblem<dim> poisson(min_degree,
                                      max_degree,
                                      ReferenceCells::Tetrahedron);
          poisson.run(n_cycles_max, error_threshold, false);
        }

      // if (false)
      {
        PoissonProblem<dim> poisson(min_degree,
                                    max_degree,
                                    ReferenceCells::Hexahedron);
        poisson.run(n_cycles_max, error_threshold, false);
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
