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


// Test the electrostatic support points of FE_SimplexP by printing them


#include <deal.II/fe/fe_simplex_p.h>

#include "../tests.h"

template <int dim>
void
test(const unsigned int degree)
{
  const FE_SimplexP<dim>        fe(degree);
  const std::vector<Point<dim>> points = fe.get_unit_support_points();

  deallog << "degree: " << degree << std::endl;
  // go over all points
  for (const auto &p : points)
    deallog << p << " ";
  deallog << std::endl;
}


int
main()
{
  initlog();

  // test for 2D
  {
    deallog.push("2D");
    for (unsigned int i = 4; i < 4; ++i)
      test<2>(i);
    deallog.pop();
  }
  // test for 3D
  {
    deallog.push("3D");
    for (unsigned int i = 4; i < 4; ++i)
      test<3>(i);
    deallog.pop();
  }
}
