#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/fe_simplex_p.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/fully_distributed_tria.h>


// This is needed for C++ output:
#include <iostream>
#include <fstream>
// And this for the declarations of the `std::sqrt` and `std::fabs` functions:
#include <cmath>



void cube(std::uint8_t refinement_choice)
{
  dealii::ConditionalOStream pcout(
    std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);


  pcout << "Cube" << std::endl;
  using namespace dealii;
  constexpr unsigned int dim = 3;

  Triangulation<3, 3> tria;
  GridGenerator::subdivided_hyper_cube_with_simplices(tria, 2);
  {
    std::ofstream out("grid_cube.0.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(tria, out);
  }

  MappingFE<dim, dim> mapping(FE_SimplexP<dim, dim>(1));
  QGaussSimplex<dim>  quad(4);

  double max_aspect_ratio =
    GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);
  pcout << "Aspect ration at level 0 is: " << max_aspect_ratio << std::endl;

  for (const auto &cell : tria.active_cell_iterators())
    {
      cell->set_refine_flag(RefinementCase<dim>::isotropic_refinement);
      cell->set_refine_choice(static_cast<unsigned int>(refinement_choice));
    }
  tria.execute_coarsening_and_refinement();

  max_aspect_ratio =
    GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);

  pcout << "Aspect ration at level 1 is: " << max_aspect_ratio << std::endl;


  for (const auto &cell : tria.active_cell_iterators())
    {
      const auto parent = cell->parent();
      for (unsigned int i = 0; i < parent->n_children(); ++i)
        {
          auto child = parent->child(i);
          child->set_material_id(i);
        }
    }
  {
    std::ofstream out("grid_cube.1.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(tria, out);
  }

  for (int i = 2; i < 5; i++)
    {
      for (const auto &cell : tria.active_cell_iterators())
        {
          cell->set_refine_flag(RefinementCase<dim>::isotropic_refinement);
          cell->set_refine_choice(static_cast<unsigned int>(refinement_choice));
        }
      tria.execute_coarsening_and_refinement();
      for (const auto &cell : tria.active_cell_iterators())
        {
          const auto parent = cell->parent();
          for (unsigned int i = 0; i < parent->n_children(); ++i)
            {
              auto child = parent->child(i);
              child->set_material_id(i);
            }
        }
      std::ofstream out("grid_cube." + std::to_string(i) + ".vtk");
      GridOut       grid_out;
      grid_out.write_vtk(tria, out);

      max_aspect_ratio =
        GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);

      pcout << "Aspect ration at level " << i << " is: " << max_aspect_ratio
            << std::endl;
    }
}

void grid_in(std::uint8_t refinement_choice)
{
  dealii::ConditionalOStream pcout(
    std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << "Grid in" << std::endl;
  using namespace dealii;
  constexpr unsigned int dim = 3;


  for (unsigned int refinement = 0; refinement < 4; ++refinement)
    {
      const auto serial_grid_generator =
        [refinement,
         refinement_choice](dealii::Triangulation<dim, dim> &tria_serial) {
          // set up triangulation
          dealii::GridIn<dim> grid_in;
          grid_in.attach_triangulation(tria_serial);
          std::ifstream input_file("/home/still/ExaDG/lung.vtk");
          grid_in.read_vtk(input_file);

          for (unsigned int i = 0; i < refinement; ++i)
            {
              for (const auto &cell : tria_serial.active_cell_iterators())
                {
                  cell->set_refine_flag(
                    RefinementCase<dim>::isotropic_refinement);
                  cell->set_refine_choice(
                    static_cast<unsigned int>(refinement_choice));
                }
              tria_serial.execute_coarsening_and_refinement();
            }
        };
      const auto serial_grid_partitioner =
        [&](dealii::Triangulation<dim, dim> &tria_serial,
            const MPI_Comm                   comm,
            const unsigned int) {
          dealii::GridTools::partition_triangulation_zorder(
            dealii::Utilities::MPI::n_mpi_processes(comm), tria_serial);
        };

      const unsigned int group_size = 20;

      parallel::fullydistributed::Triangulation<dim> tria(MPI_COMM_WORLD);
      typename dealii::TriangulationDescription::Settings
        triangulation_description_setting =
          dealii::TriangulationDescription::default_setting;
      const auto description = dealii::TriangulationDescription::Utilities::
        create_description_from_triangulation_in_groups<dim, dim>(
          serial_grid_generator,
          serial_grid_partitioner,
          tria.get_communicator(),
          group_size,
          dealii::Triangulation<dim>::none,
          triangulation_description_setting);

      tria.create_triangulation(description);

      MappingFE<dim, dim> mapping(FE_SimplexP<dim, dim>(1));
      QGaussSimplex<dim>  quad(4);

      double max_aspect_ratio =
        GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);
      pcout << "Aspect ration at level " << refinement
            << " is: " << max_aspect_ratio << std::endl;
    }
}


void ball(std::uint8_t refinement_choice)
{
  dealii::ConditionalOStream pcout(
    std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << "Ball" << std::endl;
  using namespace dealii;
  constexpr unsigned int dim = 3;

  Triangulation<3, 3> tria, temp;
  GridGenerator::subdivided_cylinder(temp, 2);
  GridGenerator::convert_hypercube_to_simplex_mesh(temp, tria);
  {
    std::ofstream out("grid_ball.0.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(tria, out);
  }

  MappingFE<dim, dim> mapping(FE_SimplexP<dim, dim>(1));
  QGaussSimplex<dim>  quad(4);

  double max_aspect_ratio =
    GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);

  pcout << "Aspect ration at level 0 is: " << max_aspect_ratio << std::endl;

  for (const auto &cell : tria.active_cell_iterators())
    {
      cell->set_refine_flag(RefinementCase<dim>::isotropic_refinement);
      cell->set_refine_choice(static_cast<unsigned int>(refinement_choice));
    }
  tria.execute_coarsening_and_refinement();

  max_aspect_ratio =
    GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);

  pcout << "Aspect ration at level 1 is: " << max_aspect_ratio << std::endl;


  for (const auto &cell : tria.active_cell_iterators())
    {
      const auto parent = cell->parent();
      for (unsigned int i = 0; i < parent->n_children(); ++i)
        {
          auto child = parent->child(i);
          child->set_material_id(i);
        }
    }
  {
    std::ofstream out("grid_ball.1.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(tria, out);
  }

  for (int i = 2; i < 5; i++)
    {
      for (const auto &cell : tria.active_cell_iterators())
        {
          cell->set_refine_flag(RefinementCase<dim>::isotropic_refinement);
          cell->set_refine_choice(static_cast<unsigned int>(refinement_choice));
        }
      tria.execute_coarsening_and_refinement();
      for (const auto &cell : tria.active_cell_iterators())
        {
          const auto parent = cell->parent();
          for (unsigned int i = 0; i < parent->n_children(); ++i)
            {
              auto child = parent->child(i);
              child->set_material_id(i);
            }
        }
      std::ofstream out("grid_ball." + std::to_string(i) + ".vtk");
      GridOut       grid_out;
      grid_out.write_vtk(tria, out);

      max_aspect_ratio =
        GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);

      pcout << "Aspect ration at level " << i << " is: " << max_aspect_ratio
            << std::endl;
    }
}


void cylinder(std::uint8_t refinement_choice)
{
  dealii::ConditionalOStream pcout(
    std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);


  pcout << "Cylinder" << std::endl;
  using namespace dealii;
  constexpr unsigned int dim = 3;

  Triangulation<3, 3> tria, temp;
  GridGenerator::subdivided_cylinder(temp, 2);
  GridGenerator::convert_hypercube_to_simplex_mesh(temp, tria);
  {
    std::ofstream out("grid_cylinder.0.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(tria, out);
  }

  MappingFE<dim, dim> mapping(FE_SimplexP<dim, dim>(1));
  QGaussSimplex<dim>  quad(4);

  double max_aspect_ratio =
    GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);

  pcout << "Aspect ration at level 0 is: " << max_aspect_ratio << std::endl;

  for (const auto &cell : tria.active_cell_iterators())
    {
      cell->set_refine_flag(RefinementCase<dim>::isotropic_refinement);
      cell->set_refine_choice(static_cast<unsigned int>(refinement_choice));
    }
  tria.execute_coarsening_and_refinement();

  max_aspect_ratio =
    GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);

  pcout << "Aspect ration at level 1 is: " << max_aspect_ratio << std::endl;


  for (const auto &cell : tria.active_cell_iterators())
    {
      const auto parent = cell->parent();
      for (unsigned int i = 0; i < parent->n_children(); ++i)
        {
          auto child = parent->child(i);
          child->set_material_id(i);
        }
    }
  {
    std::ofstream out("grid_cylinder.1.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(tria, out);
  }

  for (int i = 2; i < 5; i++)
    {
      for (const auto &cell : tria.active_cell_iterators())
        {
          cell->set_refine_flag(RefinementCase<dim>::isotropic_refinement);
          cell->set_refine_choice(static_cast<unsigned int>(refinement_choice));
        }
      tria.execute_coarsening_and_refinement();
      for (const auto &cell : tria.active_cell_iterators())
        {
          const auto parent = cell->parent();
          for (unsigned int i = 0; i < parent->n_children(); ++i)
            {
              auto child = parent->child(i);
              child->set_material_id(i);
            }
        }
      std::ofstream out("grid_cylinder." + std::to_string(i) + ".vtk");
      GridOut       grid_out;
      grid_out.write_vtk(tria, out);

      max_aspect_ratio =
        GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);

      pcout << "Aspect ration at level " << i << " is: " << max_aspect_ratio
            << std::endl;
    }
}



void tet(std::uint8_t refinement_choice)
{
  dealii::ConditionalOStream pcout(
    std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);


  using namespace dealii;
  constexpr unsigned int dim = 3;
  pcout << "Tet" << std::endl;
  Triangulation<3, 3> tria;
  GridGenerator::reference_cell(tria, ReferenceCells::Tetrahedron);
  {
    std::ofstream out("grid.0.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(tria, out);
  }

  MappingFE<dim, dim> mapping(FE_SimplexP<dim, dim>(1));
  QGaussSimplex<dim>  quad(4);

  double max_aspect_ratio =
    GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);

  pcout << "Aspect ration at level 0 is: " << max_aspect_ratio << std::endl;

  tria.begin_active()->set_refine_flag(
    RefinementCase<dim>::isotropic_refinement);
  tria.begin_active()->set_refine_choice(
    static_cast<unsigned int>(refinement_choice));

  tria.execute_coarsening_and_refinement();


  max_aspect_ratio =
    GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);

  pcout << "Aspect ration at level 1 is: " << max_aspect_ratio << std::endl;


  for (const auto &cell : tria.active_cell_iterators())
    {
      const auto parent = cell->parent();
      for (unsigned int i = 0; i < parent->n_children(); ++i)
        {
          auto child = parent->child(i);
          child->set_material_id(i);
        }
    }
  {
    std::ofstream out("grid.1.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(tria, out);
  }

  for (int i = 2; i < 6; i++)
    {
      for (const auto &cell : tria.active_cell_iterators())
        {
          cell->set_refine_flag(RefinementCase<dim>::isotropic_refinement);
          cell->set_refine_choice(static_cast<unsigned int>(refinement_choice));
        }
      tria.execute_coarsening_and_refinement();
      for (const auto &cell : tria.active_cell_iterators())
        {
          const auto parent = cell->parent();
          for (unsigned int i = 0; i < parent->n_children(); ++i)
            {
              auto child = parent->child(i);
              child->set_material_id(i);
            }
        }
      std::ofstream out("grid." + std::to_string(i) + ".vtk");
      GridOut       grid_out;
      grid_out.write_vtk(tria, out);

      max_aspect_ratio =
        GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);

      pcout << "Aspect ration at level " << i << " is: " << max_aspect_ratio
            << std::endl;
    }
}

int main(int argc, char **argv)
{
  dealii::ConditionalOStream pcout(
    std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);


  int refinement_choice = 0;
  if (argc > 1)
    refinement_choice = std::atoi(argv[1]);

  for (; refinement_choice < 4; ++refinement_choice)
    {
      pcout << "Refinment choice: " << refinement_choice << std::endl;
      if (false)
        {
          tet(refinement_choice);
          cube(refinement_choice);
          cylinder(refinement_choice);
          ball(refinement_choice);
        }
      if (true)
        grid_in(refinement_choice);
    }
  return 0;
}
