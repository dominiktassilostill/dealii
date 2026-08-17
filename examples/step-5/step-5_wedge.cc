
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


    FEEvaluation<dim, -1, 0, dim, number>     eval_cell(matrix_free, 0, 0);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_face(matrix_free,
                                                        true,
                                                        0,
                                                        0);
    const double penalty_factor = 1.0 * (dof_handler.get_fe().degree + dim) /
                                  double(dim) *
                                  (dof_handler.get_fe().degree + 1);
    {
      unsigned int n_cells =
        matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
      array_penalty_parameter.resize(n_cells);

      const dealii::FiniteElement<dim> &fe = dof_handler.get_fe();
      const auto reference_cells = dof_handler.get_fe().reference_cell();

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
                  dealii::FEFaceValues<dim> *fe_face_values;
                  if (reference_cells.face_reference_cell(f).is_hyper_cube())
                    fe_face_values = &fe_face_values_quad;
                  else
                    fe_face_values = &fe_face_values_tri;

                  fe_face_values->reinit(cell, f);
                  const number factor = (cell->at_boundary(f) and
                                         not(cell->has_periodic_neighbor(f))) ?
                                          1. :
                                          0.5;
                  for (unsigned int q = 0;
                       q < fe_face_values->n_quadrature_points;
                       ++q)
                    {
                      surface_area += fe_face_values->JxW(q) * factor;
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
    const VectorType dummy;
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
    FEFaceIntegrator integrator_inner(matrix_free, range);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        integrator_inner.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);

        const VectorizedArray<number> sigma =
          integrator_inner.read_cell_data(array_penalty_parameter);

        for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
          {
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
    FEFaceIntegrator integrator_inner(matrix_free, range);
    Solution<dim>    solution;

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);

        const VectorizedArray<number> sigma =
          integrator_inner.read_cell_data(array_penalty_parameter);

        for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
          {
            const auto normal = integrator_inner.normal_vector(q);
            const auto g =
              solution.value_array(integrator_inner.quadrature_point(q));

            integrator_inner.submit_gradient(-g * normal, q);
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
      const auto serial_grid_generator =
        [&cycle, &ref_cell](dealii::Triangulation<dim, dim> &tria_serial) {
          // set up triangulation
          if (ref_cell == ReferenceCells::Pyramid)
            GridGenerator::subdivided_hyper_cube_with_pyramids(tria_serial,
                                                               std::pow(2,
                                                                        cycle));
          else if (ref_cell == ReferenceCells::Wedge)
            {
              dealii::Triangulation<dim, dim> temp;
              GridGenerator::subdivided_hyper_cube_with_wedges(temp, 2);
              // GridGenerator::subdivided_hyper_cube_with_wedges(tria_serial,
              // 2);
              // GridGenerator::subdivided_hyper_cube_with_wedges(tria_serial,
              //                                               std::pow(2,
              //                                                      cycle));
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
          else if (ref_cell.is_simplex())
            GridGenerator::subdivided_hyper_cube_with_simplices(tria_serial, 2);
          else if (ref_cell.is_hyper_cube())
            GridGenerator::subdivided_hyper_cube(tria_serial, 2);
          else
            DEAL_II_NOT_IMPLEMENTED();

          // if (ref_cell != ReferenceCells::Pyramid)
          if (ref_cell != ReferenceCells::Pyramid &&
              ref_cell != ReferenceCells::Wedge)
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
              LinearAlgebra::distributed::Vector<Number> x, rhs;
              op.initialize_dof_vector(x);
              op.initialize_dof_vector(rhs);
              op.rhs(rhs);

              ReductionControl reduction_control(dof_handler.n_dofs(),
                                                 1e-12,
                                                 1e-12);
              SolverCG<LinearAlgebra::distributed::Vector<Number>> solver(
                reduction_control);
              PreconditionIdentity preconditioner;

              solver.solve(op, x, rhs, preconditioner);

              pcout << "Solved in " << reduction_control.last_step()
                    << " iterations with final residual "
                    << std::setprecision(16) << reduction_control.last_value()
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

  int min_degree   = 1;
  int max_degree   = 7;
  int n_cycles_max = 7;
  if (argc > 1)
    min_degree = std::atoi(argv[1]);
  if (argc > 2)
    max_degree = std::atoi(argv[2]);
  if (argc > 3)
    n_cycles_max = std::atoi(argv[3]);

  {
    if (false)
      {
        do_test<dim, double>(min_degree,
                             max_degree,
                             n_cycles_max,
                             ReferenceCells::Pyramid);
        std::cout << std::endl;
      }

    // if (false)
    {
      do_test<dim, double>(min_degree,
                           max_degree,
                           n_cycles_max,
                           ReferenceCells::Wedge);
      std::cout << std::endl;
    }

    if (false)
      {
        do_test<dim, double>(min_degree,
                             max_degree,
                             n_cycles_max,
                             ReferenceCells::Tetrahedron);
        std::cout << std::endl;
      }

    if (false)
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
