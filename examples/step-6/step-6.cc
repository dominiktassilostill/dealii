
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

using namespace dealii;

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

private:
  void
  do_cell_integral_range(const MatrixFree<dim, number> &,
                         VectorType &,
                         const VectorType &,
                         const std::pair<unsigned int, unsigned int> &) const
  {}

  void do_face_integral_range(
    const MatrixFree<dim, number> &matrix_free,
    VectorType &,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceIntegrator integrator_inner(matrix_free, true);
    FEFaceIntegrator integrator_outer(matrix_free, false);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        integrator_inner.gather_evaluate(src, EvaluationFlags::gradients);
        integrator_outer.reinit(face);
        integrator_outer.gather_evaluate(src, EvaluationFlags::gradients);

        for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
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
            for (unsigned int v = 0; v < grad_inner_normal.size(); ++v)
              if (std::abs(grad_inner_normal[v] - normal_derivative_inner[v]) >
                    1e-14 ||
                  std::abs(grad_outer_normal[v] - normal_derivative_outer[v]) >
                    1e-14)
                different = true;

            if (different)
              std::cout << "face " << face << " quadrature point " << q
                        << " inner: "
                        << grad_inner_normal - normal_derivative_inner
                        << ", outer: "
                        << grad_outer_normal - normal_derivative_outer
                        << std::endl;
            else
              std::cout << "face " << face << " quadrature point " << q
                        << " normal: " << normal << std::endl;
          }
      }
  }

  void do_boundary(const MatrixFree<dim, number> &matrix_free,
                   VectorType &,
                   const VectorType                            &src,
                   const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceIntegrator integrator_inner(matrix_free, true);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        integrator_inner.gather_evaluate(src, EvaluationFlags::gradients);

        for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
          {
            const auto normal = integrator_inner.normal_vector(q);

            const auto grad_inner = integrator_inner.get_gradient(q);

            const auto grad_inner_normal = grad_inner * normal;

            const auto normal_derivative_inner =
              integrator_inner.get_normal_derivative(q);

            bool different = false;
            for (unsigned int v = 0; v < grad_inner_normal.size(); ++v)
              if (std::abs(grad_inner_normal[v] - normal_derivative_inner[v]) >
                  1e-14)
                different = true;

            if (different)
              DEAL_II_ASSERT_UNREACHABLE();
            if (different)
              std::cout << "boundary face " << face << " quadrature point " << q
                        << ": " << grad_inner_normal - normal_derivative_inner
                        << " and jacobian x normal: "
                        << integrator_inner.inverse_jacobian(q) * normal
                        << std::endl;
            else
              std::cout << "boundary face " << face << " quadrature point " << q
                        << "jacobian x normal: "
                        << integrator_inner.inverse_jacobian(q) * normal
                        << std::endl;
          }
      }
  }

  MatrixFree<dim, number> matrix_free;

  AffineConstraints<number> constraints;
};



template <int dim, typename Number>
void do_test(const unsigned int fe_degree, const ReferenceCell<dim> &ref_cell)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);
  pcout << "Running in " << dim << "D with degree " << fe_degree << " on "
        << ref_cell.to_string() << " elements" << std::endl;

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

  for (unsigned int cycle = 0; cycle < 1; ++cycle)
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
                      // else
                      //   std::cout << "boundary face in standard orientation "
                      //            << int(face_orientation) << std::endl;
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

      {
        // for (const bool use_equidistant_points :
        // std::vector<bool>{{true, false}})
        const bool use_equidistant_points = false;
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

          LinearAlgebra::distributed::Vector<Number> x, b;
          op.initialize_dof_vector(x);
          op.initialize_dof_vector(b);
          for (Number &a : b)
            a = static_cast<double>(rand()) / RAND_MAX;

          op.vmult(x, b);
        }
      }
    }
}


int main(int argc, char **argv)
{
  constexpr int                    dim = 3;
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  int degree         = 1;
  int reference_cell = 1;
  if (argc > 1)
    degree = std::atoi(argv[1]);
  if (argc > 2)
    reference_cell = std::atoi(argv[2]);

  if (false)
    {
      dealii::Triangulation<2> tria;
      GridGenerator::reference_cell(tria, ReferenceCells::Triangle);
      tria.refine_global();
      for (const auto &cell : tria.active_cell_iterators())
        std::cout << "cell " << cell->active_cell_index()
                  << " vertices: " << cell->vertex(0) << ", " << cell->vertex(1)
                  << ", " << cell->vertex(2) << std::endl;
    }
  {
    for (unsigned int cycle = 1; cycle < 5; ++cycle)
      {
        dealii::Triangulation<dim, dim> tria1, tria2;
        if (reference_cell == 1)
          {
            std::cout << "Wedge" << std::endl;
            std::vector<Point<dim>>    vertices;
            std::vector<CellData<dim>> cells;
            vertices.emplace_back(0.0, 0.0, 0.0);
            vertices.emplace_back(1.0, 0.0, 0.0);
            vertices.emplace_back(0.0, 1.0, 0.0);
            vertices.emplace_back(0.0, 0.0, 1.0);
            vertices.emplace_back(1.0, 0.0, 1.0);
            vertices.emplace_back(0.0, 1.0, 1.0);
            {
              CellData<dim> wedge;
              wedge.vertices = {0, 1, 2, 3, 4, 5};
              cells.push_back(wedge);
            }
            // if (false)
            {
              vertices.emplace_back(1.0, 1.0, 0.0);
              vertices.emplace_back(1.0, 1.0, 1.0);

              CellData<dim> wedge;
              wedge.vertices = {1, 6, 2, 4, 7, 5};
              cells.push_back(wedge);
            }

            tria1.create_triangulation(vertices, cells, SubCellData());

            // tria1.clear();
            // GridGenerator::subdivided_hyper_cube_with_wedges(tria1, 5);

            if (cycle > 0)
              tria1.refine_global(cycle);

            // const auto                 new_vertices = tria1.get_vertices();
            // std::vector<CellData<dim>> new_cells;
            // for (const auto &cell : tria1.active_cell_iterators())
            //   {
            //     const auto         reference_cell = cell->reference_cell();
            //     const unsigned int n_vertices     =
            //     reference_cell.n_vertices();

            //     CellData<dim> wedge;
            //     wedge.vertices.resize(n_vertices);

            //     for (unsigned int i = 0; i < n_vertices; ++i)
            //       {
            //         wedge.vertices[i] = cell->vertex_index(i);
            //       }
            //     new_cells.push_back(wedge);
            //   }
            // tria2.create_triangulation(new_vertices, new_cells,
            // SubCellData());
          }
        else if (reference_cell == 2)
          {
            std::cout << "Tet" << std::endl;
            GridGenerator::subdivided_hyper_cube_with_simplices(tria1, 2);
            tria1.refine_global(cycle);
          }
        else if (reference_cell == 3)
          {
            std::cout << "Pyramid" << std::endl;
            GridGenerator::subdivided_hyper_cube_with_pyramids(tria1, 2);
            tria1.refine_global(cycle);
          }
        else
          {
            DEAL_II_NOT_IMPLEMENTED();
          }
        {
          dealii::Triangulation<dim, dim> *tria;
          tria = &tria1;

          for (const auto &cell : tria->active_cell_iterators())
            {
              // std::cout << "cell " << cell->active_cell_index() << std::endl;
              for (const auto f : cell->face_indices())
                {
                  if (cell->face(f)->at_boundary())
                    {
                      const auto face_orientation =
                        cell->combined_face_orientation(f);

                      if (face_orientation !=
                          numbers::default_geometric_orientation)
                        std::cout << "boundary face " << f << " of cell "
                                  << cell->active_cell_index()
                                  << " in non default orientation in cycle "
                                  << cycle << " with orientation "
                                  << int(face_orientation) << std::endl;
                      // else
                      //   std::cout << "boundary face in standard orientation
                      //   "
                      //            << int(face_orientation) << std::endl;
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
                          std::cout << "face " << f << " of cell "
                                    << cell->active_cell_index()
                                    << " with 2 non standard sides in cycle "
                                    << cycle << " with orientations "
                                    << int(face_orientation) << " and "
                                    << int(face_orientation_neighbor)
                                    << std::endl;
                        }
                    }
                }
            }
        }

        const std::vector<unsigned int> vertex_to_reference{
          {0, 1, 2, 3, 4, 5, 6, 12, 7, 13, 8, 14, 9, 10, 11, 15, 16, 17}};

        const auto new_isotropic_child_cell_vertices =
          ReferenceCells::Wedge.new_isotropic_child_cell_vertices(0);
        if (false)
          {
            for (const auto &cell : tria2.active_cell_iterators())
              {
                std::cout << "child " << cell->active_cell_index() << std::endl;
                std::cout << "Vertices ";
                for (const auto v : cell->vertex_indices())
                  {
                    if (new_isotropic_child_cell_vertices
                          [cell->active_cell_index()][v] ==
                        vertex_to_reference[cell->vertex_index(v)])
                      std::cout << vertex_to_reference[cell->vertex_index(v)]
                                << " ";
                    else
                      std::cout << vertex_to_reference[cell->vertex_index(v)]
                                << " vs "
                                << new_isotropic_child_cell_vertices
                                     [cell->active_cell_index()][v]
                                << " ";
                  }
                std::cout << std::endl;
              }
          }
        if (false)
          for (auto *tria :
               std::vector<Triangulation<dim, dim> *>{{&tria1, &tria2}})
            {
              for (const auto &cell : tria->active_cell_iterators())
                if (cell->active_cell_index() == 3)
                  {
                    if (cell->level() != 0)
                      {
                        const auto parent      = cell->parent();
                        const auto parent_face = parent->face(0);
                        std::cout
                          << "Orientation parent face: "
                          << int(parent->combined_face_orientation(0))
                          << " with vertices: " << parent_face->vertex_index(0)
                          << " " << parent_face->vertex_index(1) << " "
                          << parent_face->vertex_index(2) << ": "
                          << parent_face->vertex(0) << ", "
                          << parent_face->vertex(1) << ", "
                          << parent_face->vertex(2) << std::endl;

                        const auto parent_face_child = parent_face->child(3);
                        std::cout << "parent face child: "
                                  << parent_face_child->vertex(0) << ", "
                                  << parent_face_child->vertex(1) << ", "
                                  << parent_face_child->vertex(2) << std::endl;
                      }
                    std::cout << "child " << cell->active_cell_index()
                              << std::endl;
                    std::cout
                      << "Vertices: " << cell->vertex_index(0) << " "
                      << cell->vertex_index(1) << " " << cell->vertex_index(2)
                      << " " << cell->vertex_index(3) << " "
                      << cell->vertex_index(4) << " " << cell->vertex_index(5)
                      << std::endl;
                    const auto face = cell->face(0);
                    {
                      std::cout << "combined face orientation of face 0: "
                                << int(cell->combined_face_orientation(0))
                                << std::endl;
                      std::cout << "Face 0 with vertices" << std::endl;
                      std::cout << face->vertex_index(0) << " ";
                      std::cout << face->vertex_index(1) << " ";
                      std::cout << face->vertex_index(2) << ": "
                                << face->vertex(0) << ", " << face->vertex(1)
                                << ", " << face->vertex(2) << std::endl;

                      std::cout << "Line indices: " << face->line_index(0)
                                << " " << face->line_index(1) << " "
                                << face->line_index(2) << std::endl;
                    }
                    std::cout << std::endl;
                  }
            }
        std::cout << "Done with cycle " << cycle << std::endl;
      }
    return 1;
  }


  {
    if (reference_cell == 0)
      {
        do_test<dim, double>(degree, ReferenceCells::Pyramid);
      }

    if (reference_cell == 1)

      {
        do_test<dim, double>(degree, ReferenceCells::Wedge);
      }

    if (reference_cell == 2)

      {
        do_test<dim, double>(degree, ReferenceCells::Tetrahedron);
      }

    if (reference_cell == 3)

      {
        do_test<dim, double>(degree, ReferenceCells::Hexahedron);
      }
  }
}
