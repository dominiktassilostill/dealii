#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include<deal.II/grid/reference_cell.h>

using namespace dealii;


void grid()
{
  
  Triangulation<3> triangulation;
  
  GridGenerator::reference_cell(triangulation, ReferenceCells::Wedge);
  triangulation.refine_global(4);

}


int main()
{
  grid();
}
