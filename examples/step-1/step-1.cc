#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>


#include <fstream>
#include <iostream>


using namespace dealii;

template <int dim>
void tet(std::uint8_t refinement, unsigned int n_cycles)
{
  std::cout << "Refinement case: " << int(refinement) << std::endl;


  const ReferenceCell reference_cell = ReferenceCells::Tetrahedron;
  MappingFE<dim>      mapping(FE_SimplexP<dim>(3));
  const auto          quad = reference_cell.get_gauss_type_quadrature<dim>(4);


  Triangulation<dim> triangulation;

  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      std::cout << "Cycle: " << cycle << std::endl;

      if (cycle == 0)
        {
          // GridGenerator::reference_cell(triangulation, reference_cell);
          {
            std::vector<Point<dim>>    vertices;
            std::vector<CellData<dim>> cells;

            vertices.push_back(Point<dim>(0.7, 1, 1));
            vertices.push_back(Point<dim>(0.7, -1, -1));
            vertices.push_back(Point<dim>(-1.5, 1, -1));
            vertices.push_back(Point<dim>(-1, -1, 1));

            vertices.push_back(Point<dim>(1, 1, 1));
            vertices.push_back(Point<dim>(1, -1, -1));
            vertices.push_back(Point<dim>(-1, 1, -1));
            vertices.push_back(Point<dim>(-1, -1, 1));

            {
              CellData<dim> tet;
              tet.vertices = {0, 1, 3, 2};
              // tet.vertices = {4,5,7,6};
              cells.push_back(tet);
            }

            triangulation.create_triangulation(vertices, cells, SubCellData());
          }
        }


      else
        {
          for (const auto &cell : triangulation.active_cell_iterators())
            {
              cell->set_refine_flag(RefinementCase<dim>::isotropic_refinement);
              cell->set_refine_choice(static_cast<unsigned int>(refinement));
              // if (cycle == 1)
              // cell->set_refine_choice(static_cast<unsigned int>(3));
            }

          triangulation.execute_coarsening_and_refinement();
        }

      {
        auto aspect_ratios =
          dealii::GridTools::compute_aspect_ratio_of_cells(mapping,
                                                           triangulation,
                                                           quad);
        double first_entry = aspect_ratios(0);
        std::cout << first_entry << " ";
        for (auto &ratio : aspect_ratios)
          if (std::abs(ratio - first_entry) > 1e-12)
            std::cout << ratio << " ";
        std::cout << std::endl;
        // std::cout << aspect_ratios.linfty_norm() << " " <<
        // aspect_ratios.l1_norm() << std::endl;

        for (const auto &cell : triangulation.active_cell_iterators())
          {
            // std::cout << cell->measure() << " ";
          }
        // std::cout << std::endl;
      }

      if (cycle > 0)
        for (const auto &cell : triangulation.active_cell_iterators())
          {
            auto parent = cell->parent();
            for (unsigned int i = 0; i < 8; ++i)
              {
                auto child = parent->child(i);
                child->set_material_id(i);
              }
          }

      std::ofstream out("grid_" + Utilities::int_to_string(int(refinement), 1) +
                        "_" + Utilities::int_to_string(int(cycle), 1) + ".vtk");
      GridOut       grid_out;
      grid_out.write_vtk(triangulation, out);
      // std::cout << "Grid written to grid-1.vtk" << std::endl;
    }
}



int main()
{
  const unsigned int n_cycles = 4;
  for (unsigned int i = 0; i < 0; ++i)
    {
      tet<3>(i, n_cycles);
    }
  tet<3>(0, n_cycles); // IsotropicRefinementChoice<dim>::isotropic_refinement
  tet<3>(4, n_cycles); // IsotropicRefinementChoice::cut_tet_49

  return 0;
}
