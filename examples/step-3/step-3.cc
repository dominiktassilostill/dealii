
#include <fstream>
#include <iostream>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_wedge_p.h>


using namespace dealii;



int main(int argc, char **argv)
{
  const unsigned int dim = 3;

  for (unsigned int degree = 1; degree < 5; ++degree)
    {
      const FE_PyramidDGP<dim> fe(degree, false);
      const FE_PyramidDGP<dim> fe_equidistant(degree, true);

      const auto p1 = fe.get_unit_support_points();
      const auto p2 = fe_equidistant.get_unit_support_points();

      std::cout << "N dofs at degree " << degree << ": " << p1.size() << " "
                << p2.size() << std::endl;
      for (unsigned int i = 0; i < p1.size(); ++i)
        std::cout << p1[i] << "    " << p2[i] << std::endl;
      std::cout << std::endl;
    }


  return 1;
}
