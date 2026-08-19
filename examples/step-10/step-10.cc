
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


#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>


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

  void reinit(const hp::MappingCollection<dim> &mapping,
              const DoFHandler<dim>            &dof_handler,
              const hp::QCollection<dim>       &quad,
              const AffineConstraints<number>  &constraints,
              const unsigned int                mg_level,
              const unsigned int                fe_degree,
              const hp::FECollection<dim>       fe_collection)
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

    const double penalty_factor =
      1.0 * (fe_degree + dim) / double(dim) * (fe_degree + 1);
    {
      unsigned int n_cells =
        matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
      array_penalty_parameter.resize(n_cells);

      hp::FEValues<dim> hp_fe_values(mapping,
                                     fe_collection,
                                     quad,
                                     update_JxW_values);

      const QGauss<dim - 1>        quadrature_quad(fe_degree + 1);
      const QGaussSimplex<dim - 1> quadrature_tri(fe_degree + 1);
      const Quadrature<dim - 1>    quadrature_dummy(
        std::vector<Point<dim - 1>>{Point<dim - 1>()});

      hp::QCollection<dim - 1> face_quadratures_quad(quadrature_quad,
                                                     quadrature_quad,
                                                     quadrature_dummy,
                                                     quadrature_quad);

      hp::QCollection<dim - 1> face_quadratures_tri(quadrature_tri,
                                                    quadrature_tri,
                                                    quadrature_tri,
                                                    quadrature_dummy);

      hp::FEFaceValues<dim> hp_fe_face_values_quad(mapping,
                                                   fe_collection,
                                                   face_quadratures_quad,
                                                   update_JxW_values);

      hp::FEFaceValues<dim> hp_fe_face_values_tri(mapping,
                                                  fe_collection,
                                                  face_quadratures_tri,
                                                  update_JxW_values);

      for (unsigned int i = 0; i < n_cells; ++i)
        {
          for (unsigned int v = 0;
               v < matrix_free.n_active_entries_per_cell_batch(i);
               ++v)
            {
              typename dealii::DoFHandler<dim>::cell_iterator cell =
                matrix_free.get_cell_iterator(i, v);
              hp_fe_values.reinit(cell);
              const FEValues<dim> &fe_values =
                hp_fe_values.get_present_fe_values();

              // calculate cell volume
              number volume = 0;
              for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                {
                  volume += fe_values.JxW(q);
                }

              // calculate surface area
              number surface_area = 0;
              for (const unsigned int f : cell->face_indices())
                {
                  hp::FEFaceValues<dim> *hp_fe_face_values =
                    cell->face(f)->reference_cell().is_hyper_cube() ?
                      &hp_fe_face_values_quad :
                      &hp_fe_face_values_tri;

                  hp_fe_face_values->reinit(cell, f);
                  const FEFaceValues<dim> &fe_face_values =
                    hp_fe_face_values->get_present_fe_values();

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
            const VectorizedArray<number> solution_jump =
              (integrator_inner.get_value(q) - integrator_outer.get_value(q));
            const VectorizedArray<number> averaged_normal_derivative =
              (integrator_inner.get_normal_derivative(q) +
               integrator_outer.get_normal_derivative(q)) *
              number(0.5);
            const VectorizedArray<number> test_by_value =
              solution_jump * sigma - averaged_normal_derivative;

            integrator_inner.submit_value(test_by_value, q);
            integrator_outer.submit_value(-test_by_value, q);

            integrator_inner.submit_normal_derivative(-solution_jump *
                                                        number(0.5),
                                                      q);
            integrator_outer.submit_normal_derivative(-solution_jump *
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
    {
      for (unsigned int face = range.first; face < range.second; ++face)
        {
          const auto          face_info       = matrix_free.get_face_info(face);
          const unsigned char face_number_int = face_info.interior_face_no;
          const unsigned char face_orientation_int = face_info.face_orientation;
          if (face_orientation_int != 0)
            std::cout << "boundary face number and orientation: "
                      << int(face_number_int) << ", "
                      << int(face_orientation_int) << std::endl;
        }
    }
    FEFaceIntegrator integrator_inner(matrix_free, range, true);

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
            const VectorizedArray<number> u_inner =
              integrator_inner.get_value(q);
            const VectorizedArray<number> u_outer = -u_inner;
            const VectorizedArray<number> normal_derivative_inner =
              integrator_inner.get_normal_derivative(q);
            const VectorizedArray<number> normal_derivative_outer =
              normal_derivative_inner;
            const VectorizedArray<number> solution_jump = (u_inner - u_outer);
            const VectorizedArray<number> average_normal_derivative =
              (normal_derivative_inner + normal_derivative_outer) * number(0.5);
            const VectorizedArray<number> test_by_value =
              solution_jump * sigma - average_normal_derivative;

            integrator_inner.submit_normal_derivative(-solution_jump *
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
            const auto g =
              solution.value_array(integrator_inner.quadrature_point(q));

            integrator_inner.submit_normal_derivative(-g, q);
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
void do_test(const unsigned int min_degree,
             const unsigned int max_degree,
             const unsigned int n_cycles_max)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);
  pcout << "Running in " << dim << "D with degrees between " << min_degree
        << " and " << max_degree << " on mixed mesh" << std::endl;

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

  const hp::MappingCollection<dim> mapping_collection(mapping_pyramid,
                                                      mapping_wedge,
                                                      mapping_simplex,
                                                      mapping_hypercube);
  // mapping_collection.push_back(mapping_pyramid);
  // mapping_collection.push_back(mapping_wedge);
  // mapping_collection.push_back(mapping_simplex);
  // mapping_collection.push_back(mapping_hypercube);

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

            tria_serial.create_triangulation(vertices, cells, SubCellData());
            tria_serial.refine_global(1);

            // std::ofstream out("grid-mixed.vtk");
            // GridOut       grid_out;
            // grid_out.write_vtk(tria_serial, out);
            // std::cout << "Grid written to grid-mixed.vtk" << std::endl;
          }

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

              FE_PyramidDGP<dim> fe_pyramidp(fe_degree, use_equidistant_points);
              FE_WedgeDGP<dim>   fe_wedgep(fe_degree, use_equidistant_points);
              FE_SimplexDGP<dim> fe_simplexp(fe_degree, use_equidistant_points);
              FE_DGQ<dim>        fe_q = use_equidistant_points ?
                                          FE_DGQArbitraryNodes<dim>(
                                     QIterated<1>(QTrapezoid<1>(), fe_degree)) :
                                          FE_DGQ<dim>(fe_degree);

              const hp::FECollection<dim> fe_collection(fe_pyramidp,
                                                        fe_wedgep,
                                                        fe_simplexp,
                                                        fe_q);

              dof_handler.distribute_dofs(fe_collection);

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

              const hp::QCollection<dim> quadrature_collection(quad_pyramid,
                                                               quad_wedge,
                                                               quad_simplex,
                                                               quad_hypercube);

              pcout << "Set up operator of degree " << fe_degree << std::endl;
              Operator<dim, 1, Number> op;
              // set up operator
              op.reinit(mapping_collection,
                        dof_handler,
                        quadrature_collection,
                        constraint,
                        numbers::invalid_unsigned_int,
                        fe_degree,
                        fe_collection);

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


              const hp::QCollection<dim> quad_error(
                QGaussPyramid<dim>(fe_degree + 3),
                QGaussWedge<dim>(fe_degree + 3),
                QGaussSimplex<dim>(fe_degree + 3),
                QGauss<dim>(fe_degree + 3));

              x.update_ghost_values();
              Vector<double> difference_per_cell;
              VectorTools::integrate_difference(mapping_collection,
                                                dof_handler,
                                                x,
                                                Solution<dim>(),
                                                difference_per_cell,
                                                quad_error,
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
                      << "   Mixed equidistant of degree " << fe_degree
                      << std::endl
                      << "   Number of active cells:       " << n_active_cells
                      << std::endl
                      << "   Number of degrees of freedom: " << n_dofs
                      << std::endl
                      << "   L2 error:                     " << L2_error
                      << std::endl;
              else
                pcout << "Cycle " << cycle << ':' << std::endl
                      << "   Mixed blend and warp of degree " << fe_degree
                      << std::endl
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
  std::cout << std::endl;

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
