// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2020 - 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// Test PolynomialsWedge by comparing it to the old implementation


#include <deal.II/base/polynomials_wedge.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_wedge_p.h>

#include "../tests.h"



template <int dim>
void
test(const unsigned int degree)
{
  FE_WedgeP<dim> fe_wedge(degree);
  const auto     support_points = fe_wedge.get_unit_support_points();
  deallog << "N support points: " << support_points.size() << " at degree "
          << degree << std::endl;
  const auto poly = ScalarNodalPolynomialWedge(degree,
                                               fe_wedge.n_dofs_per_cell(),
                                               support_points);

  const ScalarLagrangePolynomialWedge<dim> poly_wedge(degree);

  std::vector<Point<dim>> points;
  for (const auto &p : support_points)
    points.emplace_back(p);
  // Add some random points
  points.emplace_back(Point<dim>(0.36658, 0.58775, 0.21455));
  points.emplace_back(Point<dim>(0.64464, 0.3546, 0.3246));
  points.emplace_back(Point<dim>(0.0, 0.0, 0.999));

  // Add points from the quadrature
  QGaussWedge<dim> quad(2);
  for (unsigned int i = 0; i < quad.size(); ++i)
    points.emplace_back(quad.point(i));

  for (unsigned int i = 0; i < points.size(); ++i)
    for (unsigned int j = 0; j < fe_wedge.n_dofs_per_cell(); ++j)
      {
        const auto v1 = poly.compute_value(j, points[i]);
        const auto v2 = poly_wedge.compute_value(j, points[i]);
        if (std::abs(v2 - v1) < 1e-9)
          deallog << "ok ";
        else
          deallog << "Failure by value!!! " << i << " " << j << " " << v2 - v1
                  << std::endl;


        const auto g1 = poly.compute_grad(j, points[i]);
        const auto g2 = poly_wedge.compute_grad(j, points[i]);
        for (unsigned int d = 0; d < dim; ++d)
          {
            if (std::abs((g2 - g1)[d]) < 1e-9)
              deallog << "ok ";
            else
              deallog << "Failure in gradient!!! " << i << " " << j << " "
                      << g2[d] << " " << g1[d] << " " << d << std::endl;
          }
        deallog << std::endl;
      }
  deallog << std::endl;
}


int
main()
{
  initlog();
  {
    deallog.push("3d-1");
    test<3>(1);
    deallog.pop();

    deallog.push("3d-2");
    test<3>(2);
    deallog.pop();
  }
}
