// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2020 - 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


// Test the electrostatic support points of FE_SimplexP for consistency on the
// edges with Gauss-Lobatto nodes


#include <deal.II/base/types.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>

#include <deal.II/grid/reference_cell.h>

#include "../tests.h"

template <int dim>
void
test(const unsigned int degree)
{
  FE_Q<1> feq(degree);

  std::vector<double> edge_points;
  for (unsigned int i = 0; i < feq.n_dofs_per_cell(); ++i)
    if (i > 1)
      edge_points.push_back(feq.unit_support_point(i)[0]);

  const auto       reference_cell = ReferenceCells::get_simplex<dim>();
  FE_SimplexP<dim> fe(degree);
  const auto       points = fe.get_unit_support_points();

  for (const auto &p : points)
    {
      const double x = p[0];
      const double y = p[1];
      const double z = dim == 3 ? p[2] : 0.0;

      const std::array<double, 4> l{{1.0 - x - y - z, x, y, z}};

      double sum = 0;
      for (const auto b : l)
        sum += b;
      if (std::abs(1 - sum) > 1e-10)
        DEAL_II_ASSERT_UNREACHABLE();

      unsigned int n_pos_coordinates = 0;
      for (const auto b : l)
        if (std::abs(b) > 1e-10)
          ++n_pos_coordinates;

      if (n_pos_coordinates == 2)
        {
          // on edge, test if edge is correct
          deallog << "Edge ";
          for (unsigned int d = 0; d < dim; ++d)
            deallog << l[d + 1] << " ";

          bool valid = false;

          // search for the point
          for (unsigned int line = 0; line < reference_cell.n_lines(); ++line)
            {
              const unsigned int v0 =
                reference_cell.line_to_cell_vertices(line, 0);
              const unsigned int v1 =
                reference_cell.line_to_cell_vertices(line, 1);

              const Point<dim> vertex0 = reference_cell.vertex(v0);
              const Point<dim> vertex1 = reference_cell.vertex(v1);

              for (const auto distance : edge_points)
                {
                  const Point<dim> poisition =
                    vertex0 + distance * (vertex1 - vertex0);
                  if (p.distance(poisition) < 1e-10)
                    valid = true;
                }
            }

          if (valid)
            deallog << " is at a valid position" << std::endl;
          else
            {
              deallog << " is not at a valid poitions" << std::endl;
              DEAL_II_ASSERT_UNREACHABLE();
            }
        }
    }
}

int
main()
{
  initlog();

  deallog.push("2D");
  for (unsigned int i = 4; i < 8; ++i)
    test<2>(i);
  deallog.pop();

  deallog.push("3D");
  for (unsigned int i = 4; i < 8; ++i)
    test<3>(i);
  deallog.pop();
}
