#include "./../../../tests/simplex/simplex_grids.h"

#include <deal.II/fe/fe_pyramid_p.h>
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


#define FORCE_USE_OF_TRILINOS
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
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
    PoissonProblem(const FiniteElement<dim> &fe);

    void run();

  private:
    void setup_system();
    void assemble_system();
    void solve();
    void process_solution(const unsigned int cycle);

    MPI_Comm                                       mpi_communicator;
    parallel::fullydistributed::Triangulation<dim> triangulation;


    DoFHandler<dim> dof_handler;

    ObserverPointer<const FiniteElement<dim>> fe;

    AffineConstraints<double> constraints;

    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector       solution;
    LA::MPI::Vector       system_rhs;

    ConditionalOStream pcout;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    ConvergenceTable convergence_table;
  };


  template <int dim>
  PoissonProblem<dim>::PoissonProblem(const FiniteElement<dim> &fe)
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator)
    , dof_handler(triangulation)
    , fe(&fe)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  {}


  template <int dim>
  void PoissonProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
    
    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
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
    const QGaussPyramid<dim> quadrature_formula(fe->degree + 1);
    MappingFE<dim>           mapping(FE_PyramidP<dim>(1));

    FEValues<dim> fe_values(mapping,
                            *fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    RightHandSide<dim>                   rhs;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
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

                cell_rhs(i) +=
                  (fe_values.shape_value(i, q_point) * // phi_i(x_q)
                   rhs_value *                         // f(x_q)
                   fe_values.JxW(q_point));            // dx
              }
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }


  template <int dim>
  void PoissonProblem<dim>::solve()
  {
    SolverControl solver_control(dof_handler.n_dofs(),
                                 1e-6 * system_rhs.l2_norm());
    LA::SolverCG  solver(solver_control);


    LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#else
/* Trilinos defaults are good */
#endif
    LA::MPI::PreconditionAMG preconditioner;
    preconditioner.initialize(system_matrix, data);

    solver.solve(system_matrix,
      solution,
                 system_rhs,
                 preconditioner);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(solution);
  }


  template <int dim>
  void PoissonProblem<dim>::process_solution(const unsigned int cycle)
  {
    MappingFE<dim> mapping(FE_PyramidP<dim>(1));
    Vector<double>  difference_per_cell;
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      QGaussPyramid<dim>(fe->degree + 3),
                                      VectorTools::L2_norm);
    const double L2_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    const unsigned int n_active_cells = triangulation.n_global_active_cells();
    const unsigned int n_dofs         = dof_handler.n_dofs();

    pcout << "Cycle " << cycle << ':' << std::endl
          << "   Number of active cells:       " << n_active_cells << std::endl
          << "   Number of degrees of freedom: " << n_dofs << std::endl
          << "   L2 error:                     " << L2_error << std::endl;

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
  }


  template <int dim>
  void PoissonProblem<dim>::run()
  {
    const unsigned int n_cells_max    = 20000000;
    unsigned int       n_cells        = 1;
    unsigned int       n_subdivisions = 1;
    unsigned int cycle = 0;
    while (n_cells < n_cells_max && cycle < 10)
      {
        triangulation.clear();

        const auto serial_grid_generator =
          [&n_subdivisions](dealii::Triangulation<dim, dim> &tria_serial) {
            // set up triangulation
            GridGenerator::subdivided_hyper_cube_with_pyramids(tria_serial,
                                                               n_subdivisions+5);
          };
        const auto serial_grid_partitioner =
          [&](dealii::Triangulation<dim, dim> &tria_serial,
              const MPI_Comm                   comm,
              const unsigned int) {
            dealii::GridTools::partition_triangulation_zorder(
              dealii::Utilities::MPI::n_mpi_processes(comm), tria_serial);
          };

        const unsigned int group_size = 20;

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



        setup_system();

        assemble_system();
        solve();
        process_solution(cycle);

        n_cells        = triangulation.n_global_active_cells();
        n_subdivisions = n_subdivisions * 2;
        ++cycle;
      }

    /*
     std::string vtk_filename;
     vtk_filename = "solution";
     vtk_filename += "-pyramidp" + std::to_string(fe->degree);
     vtk_filename += ".vtk";
     std::ofstream output(vtk_filename);

     DataOut<dim> data_out;
     data_out.attach_dof_handler(dof_handler);
     data_out.add_data_vector(solution, "solution");
     data_out.build_patches(1);
     data_out.write_vtk(output);
 */

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
    convergence_table.write_text(std::cout);

    std::string error_filename = "error";
    error_filename += "-pyramid" + std::to_string(fe->degree);
    error_filename += ".tex";
    std::ofstream error_table_file(error_filename);

    convergence_table.write_tex(error_table_file);
  }


int main(int argc, char **argv)
{
  const unsigned int                       dim = 3;
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  QGaussPyramid<dim> quadrature_formula_linear(1);
  QGaussPyramid<dim> quadrature_formula_quadratic(2);

  for(unsigned int i = 0; i < quadrature_formula_linear.size(); ++i)
    std::cout << quadrature_formula_linear.point(i) << " " << quadrature_formula_linear.weight(i) << std::endl;

  std::cout << std::endl;
  for(unsigned int i = 0; i < quadrature_formula_quadratic.size(); ++i)
    std::cout << quadrature_formula_quadratic.point(i) << " " << quadrature_formula_quadratic.weight(i) << std::endl;

  try
    {  
      for (unsigned int degree = 3; degree <= 3; ++degree)
        {
          std::cout << "Solving with Pyramid elements fo degree " << degree << std::endl
                    << "============================================="
                    << std::endl
                    << std::endl;

          const FE_PyramidP<dim> fe(degree);
          PoissonProblem<dim>    poisson(fe);

          poisson.run();

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

