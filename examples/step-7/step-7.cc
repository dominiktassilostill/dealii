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

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/grid/grid_out.h>
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
  PoissonProblem(const unsigned int min_degree, const unsigned int max_degree);

  void run(const unsigned int n_cycles_max,
           const double       error_threshold,
           const bool         only_run_blend_and_warp);

private:
  void   setup_system(const hp::FECollection<dim> &fe_collection);
  void   assemble_system(const hp::FECollection<dim> &fe_collection);
  void   solve();
  bool   process_solution(const unsigned int cycle,
                          const bool         use_equidistant_points,
                          const bool         only_run_blend_and_warp,
                          const double       error_threshold,
                          const unsigned int fe_degree);
  double compute_l2_error();
  void   compute_difference_at_nodes();

  MPI_Comm                                       mpi_communicator;
  parallel::fullydistributed::Triangulation<dim> triangulation;


  DoFHandler<dim> dof_handler;

  hp::MappingCollection<dim> mapping_collection;

  AffineConstraints<double> constraints;

  LA::MPI::SparseMatrix system_matrix;
  LA::MPI::Vector       solution;
  LA::MPI::Vector       system_rhs;

  ConditionalOStream pcout;

  std::vector<ConvergenceTable> convergence_tables;

  unsigned int min_degree;
  unsigned int max_degree;
};


template <int dim>
PoissonProblem<dim>::PoissonProblem(const unsigned int min_degree,
                                    const unsigned int max_degree)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , dof_handler(triangulation)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , min_degree(min_degree)
  , max_degree(max_degree)
{
  convergence_tables.resize(2 * (max_degree - min_degree + 1));
}


template <int dim>
void PoissonProblem<dim>::setup_system(
  const hp::FECollection<dim> &fe_collection)
{
  dof_handler.clear();
  dof_handler.reinit(triangulation);
  pcout << "reinit triangulation done, there are "
        << triangulation.get_reference_cells().size()
        << " different types of cells and and fe collection of size "
        << fe_collection.size() << std::endl;

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
  dof_handler.distribute_dofs(fe_collection);

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  solution.clear();
  system_rhs.clear();
  solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  // DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);

  system_matrix.clear();
  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       mpi_communicator);
}


template <int dim>
void PoissonProblem<dim>::assemble_system(
  const hp::FECollection<dim> &fe_collection)
{
  const unsigned int   fe_degree = fe_collection[0].degree;
  hp::QCollection<dim> quadrature_collection(QGaussPyramid<3>(fe_degree + 1),
                                             QGaussWedge<3>(fe_degree + 1),
                                             QGaussSimplex<3>(fe_degree + 1),
                                             QGauss<3>(fe_degree + 1));

  hp::FEValues<dim> hp_fe_values(mapping_collection,
                                 fe_collection,
                                 quadrature_collection,
                                 update_values | update_gradients |
                                   update_quadrature_points |
                                   update_JxW_values);

  FullMatrix<double> cell_matrix;
  Vector<double>     cell_rhs;

  RightHandSide<dim> rhs;

  std::vector<types::global_dof_index> local_dof_indices;


  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_matrix = 0;

        cell_rhs.reinit(dofs_per_cell);
        cell_rhs = 0;

        hp_fe_values.reinit(cell);

        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
             ++q_point)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // compute cell matrix
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) += (fe_values.shape_grad(i, q_point) *
                                      fe_values.shape_grad(j, q_point) *
                                      fe_values.JxW(q_point)); // dx

              // compute rhs
              const double rhs_value =
                rhs.value(fe_values.quadrature_point(q_point), 0);
              cell_rhs(i) += (fe_values.shape_value(i, q_point) * // phi_i(x_q)
                              rhs_value *                         // f(x_q)
                              fe_values.JxW(q_point));            // dx
            }

        // distribute to DoFs
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

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
                                           const double error_threshold,
                                           const unsigned int fe_degree)
{
  Vector<double> difference_per_cell;

  hp::QCollection<dim> quadrature_collection_error(
    QGaussPyramid<3>(fe_degree + 3),
    QGaussWedge<3>(fe_degree + 3),
    QGaussSimplex<3>(fe_degree + 3),
    QGauss<3>(fe_degree + 3));


  VectorTools::integrate_difference(mapping_collection,
                                    dof_handler,
                                    solution,
                                    Solution<dim>(),
                                    difference_per_cell,
                                    quadrature_collection_error,
                                    VectorTools::L2_norm);

  const double L2_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);

  const unsigned int n_active_cells = triangulation.n_global_active_cells();
  const unsigned int n_dofs         = dof_handler.n_dofs();

  if (use_equidistant_points)
    pcout << "Cycle " << cycle << ':' << std::endl
          << " at degree " << fe_degree << " equidistant" << std::endl
          << "   Number of active cells:       " << n_active_cells << std::endl
          << "   Number of degrees of freedom: " << n_dofs << std::endl
          << "   L2 error:                     " << L2_error << std::endl;
  else
    pcout << "Cycle " << cycle << ':' << std::endl
          << " at degree " << fe_degree << " blend and warp" << std::endl
          << "   Number of active cells:       " << n_active_cells << std::endl
          << "   Number of degrees of freedom: " << n_dofs << std::endl
          << "   L2 error:                     " << L2_error << std::endl;
  // const double l2_manual = compute_l2_error();
  // pcout << "   Difference in L2 error: " << L2_error - l2_manual <<
  // std::endl;
  // compute_difference_at_nodes();

  unsigned int offset = fe_degree - min_degree;
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

  mapping_collection.push_back(mapping_pyramid);
  mapping_collection.push_back(mapping_wedge);
  mapping_collection.push_back(mapping_simplex);
  mapping_collection.push_back(mapping_hypercube);

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
            if (true)
              {
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

                tria_serial.create_triangulation(vertices,
                                                 cells,
                                                 SubCellData());
                tria_serial.refine_global(1);

                // std::ofstream out("grid-mixed.vtk");
                // GridOut       grid_out;
                // grid_out.write_vtk(tria_serial, out);
                // std::cout << "Grid written to grid-mixed.vtk" << std::endl;
              }
            else
              GridGenerator::subdivided_hyper_cube_with_pyramids(tria_serial,
                                                                 2);

            if (cycle > 0)
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
      }

      // we got the triangulation
      // now go over all polynomial degrees
      // and over equidistant and gl support points if needed
      const std::vector<bool> variants_vector =
        only_run_blend_and_warp ? std::vector<bool>{{false}} :
                                  std::vector<bool>{{true, false}};

      unsigned int max_degree_cycle = max_degree;
      if (cycle > 2)
        max_degree_cycle = std::min(6U, max_degree);
      if (cycle > 4)
        max_degree_cycle = std::min(5U, max_degree);
      if (cycle > 5)
        max_degree_cycle = std::min(3U, max_degree);


      for (unsigned int degree = min_degree;
           degree <= max_degree_cycle && n_dofs < n_dofs_max &&
           !reached_error_threshold[degree - min_degree];
           ++degree)
        {
          for (const bool use_equidistant_points : variants_vector)
            {
              FE_PyramidP<dim> fe_pyramidp(degree, use_equidistant_points);
              FE_WedgeP<dim>   fe_wedgep(degree, use_equidistant_points);
              FE_SimplexP<dim> fe_simplexp(degree, use_equidistant_points);
              FE_Q<dim>        fe_q =
                use_equidistant_points ?
                         FE_Q<dim>(QIterated<1>(QTrapezoid<1>(), degree)) :
                         FE_Q<dim>(degree);

              hp::FECollection<dim> new_fe_collection(fe_pyramidp,
                                                      fe_wedgep,
                                                      fe_simplexp,
                                                      fe_q);

              setup_system(new_fe_collection);

              assemble_system(new_fe_collection);
              solve();
              reached_error_threshold[degree - min_degree] =
                process_solution(cycle,
                                 use_equidistant_points,
                                 only_run_blend_and_warp,
                                 error_threshold,
                                 degree);
              pcout << std::endl;

              n_dofs = dof_handler.n_dofs();
            }
          pcout << std::endl;
        }
      pcout << std::endl;

      n_dofs  = 0;
      n_cells = triangulation.n_global_active_cells();
    }

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

          std::string error_filename = "error_CG_SpMV_";
          error_filename += "Mixed_p_" + std::to_string(degree_counter);
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

  try
    {
      const unsigned int min_degree      = 1;
      const unsigned int max_degree      = 4;
      const unsigned int n_cycles_max    = 7;
      const double       error_threshold = 1e-9;

      {
        PoissonProblem<dim> poisson(min_degree, max_degree);
        poisson.run(n_cycles_max, error_threshold, false);
        std::cout << std::endl;
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
