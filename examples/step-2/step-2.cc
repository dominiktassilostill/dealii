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
  std::cout << "Tet degenerated" << std::endl;
  using namespace dealii;
  constexpr unsigned int dim = 3;

  Triangulation<3, 3> tria;
  {
    std::vector<Point<dim>>    vertices;
    std::vector<CellData<dim>> cells;

    vertices.push_back(Point<dim>(0.0, 0.0, 0.0));
    vertices.push_back(Point<dim>(1.0, 0.0, 0.0));
    vertices.push_back(Point<dim>(0.0, 1.0, 0.0));
    //vertices.push_back(Point<dim>(-1., -1., 0.2));
    vertices.push_back(Point<dim>(0.0, 0.0, 1.0));


    {
      CellData<dim> tet;
      tet.vertices = {0, 1, 2, 3};
      cells.push_back(tet);
    }

    tria.create_triangulation(vertices, cells, SubCellData());
  }

  MappingFE<dim, dim> mapping(FE_SimplexP<dim, dim>(1));
  QGaussSimplex<dim>  quad(4);

  for (unsigned int i = 0; i < 3; i++)
    {
      for (const auto &cell : tria.active_cell_iterators())
        {
          cell->set_refine_flag(RefinementCase<dim>::isotropic_refinement);
          cell->set_refine_choice(static_cast<unsigned int>(refinement_choice));
        }
      if (i > 0)
        {
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
        }
      std::ofstream out("obtuse_tet" + std::to_string(refinement_choice) + "_" +
                        std::to_string(i) + ".vtk");
      GridOut       grid_out;
      grid_out.write_vtk(tria, out);

      auto max_aspect_ratio =
        GridTools::compute_aspect_ratio_of_cells(mapping, tria, quad);

      auto max_aspect_ratio_scalar =
        GridTools::compute_maximum_aspect_ratio(mapping, tria, quad);

      std::cout << "Max aspect ration at level " << i << " is: " << max_aspect_ratio_scalar << std::endl;
      std::cout << "Aspect ration at level " << i << " is: ";
      for (unsigned int i = 0; i < max_aspect_ratio.size(); ++i)
        std::cout << max_aspect_ratio(i) << " ";
      std::cout << std::endl;
    }
}


int main(int argc, char **argv)
{
  int refinement_choice = 0;

  if (argc > 1)
    refinement_choice = std::atoi(argv[1]);


  for (; refinement_choice < 4; ++refinement_choice)
    {
      std::cout << "Refinment choice: " << refinement_choice << std::endl;
      cube(refinement_choice);
    }


  return 0;
}
