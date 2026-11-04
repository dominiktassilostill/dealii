// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2002 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

// like fe/up_and_down test but use simplex elements instead
// Restrict a vector to a coarse grid and prolongate it again,
// repeat the procedure twice and check if the reslting vectors are the same
// i.e. if prolongation and restriction are successful.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector.h>

#include <string>
#include <vector>

#include "../tests.h"

#define PRECISION 3

template <int dim>
void
check_element(const Triangulation<dim> &tr, const hp::FECollection<dim> &fe)
{
  DoFHandler<dim> dof_handler(tr);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->reference_cell() == ReferenceCells::Pyramid)
        cell->set_active_fe_index(0);
      else if (cell->reference_cell() == ReferenceCells::Tetrahedron)
        cell->set_active_fe_index(1);
      else
        DEAL_II_NOT_IMPLEMENTED();
    }

  dof_handler.distribute_dofs(fe);

  const unsigned int n_dofs_fine = dof_handler.n_dofs();
  // create a mostly arbitrary
  // function plus a trend on this
  // grid
  Vector<double> vector_fine(n_dofs_fine);
  for (unsigned int i = 0; i < n_dofs_fine; ++i)
    vector_fine[i] = 0.0;

  for (const auto &cell : tr.cell_iterators())
    for (const auto v : cell->vertex_indices())
      {
        const auto v_global   = cell->vertex_index(v);
        vector_fine(v_global) = cell->vertex(v)(0) + cell->vertex(v)(2);
      }

  for (unsigned int i = 1; i < n_dofs_fine; ++i)
    if (vector_fine[i] == 0.0)
      {
        // find next non zero entry and interpolate
        unsigned int non_zero_index = numbers::invalid_unsigned_int;
        for (unsigned int j = i + 1;
             j < n_dofs_fine && non_zero_index == numbers::invalid_unsigned_int;
             ++j)
          if (vector_fine[j] != 0.0)
            non_zero_index = j;

        if (non_zero_index != numbers::invalid_unsigned_int)
          vector_fine[i] =
            (vector_fine[i - 1] + vector_fine[non_zero_index]) * 0.5;
      }

  // restrict this function to the
  // next coarser level and
  // distribute it again to the
  // higher level
  Vector<double> x(vector_fine.size());
  Vector<double> v(fe[0].dofs_per_cell);
  for (typename DoFHandler<dim>::cell_iterator cell = dof_handler.begin();
       cell != dof_handler.end();
       ++cell)
    if (cell->has_children() && cell->child(0)->is_active())
      {
        // first make sure that what
        // we do is reasonable. for
        // this, _all_ children have
        // to be active, not only
        // some of them
        for (unsigned int c = 0;
             c < fe[0].reference_cell().n_isotropic_children();
             ++c)
          AssertThrow(cell->child(c)->is_active(), ExcInternalError());

        // then restrict and prolongate
        cell->get_interpolated_dof_values(vector_fine, v, 0);
        cell->set_dof_values_by_interpolation(v, x, 0);
      };

  // now x is a function on the fine
  // grid that is representable on
  // the coarse grid. so another
  // cycle should not alter it any
  // more:
  Vector<double> x2(x.size());
  for (typename DoFHandler<dim>::cell_iterator cell = dof_handler.begin();
       cell != dof_handler.end();
       ++cell)
    if (cell->has_children() && cell->child(0)->is_active())
      {
        cell->get_interpolated_dof_values(x, v, 0);
        cell->set_dof_values_by_interpolation(v, x2, 0);
      };

  // then check that this is so:
  x2 -= x;
  const double relative_residual = (x2.l2_norm() / x.l2_norm());

  const double threshold = 1e-6;
  deallog << ", dofs_per_cell=" << fe[0].dofs_per_cell
          << "; relative residual: " << relative_residual << " "
          << (relative_residual < threshold ? "ok" : "botched up!")
          << std::endl;

  // TODO:[WB] Why this exception with a value different from above. Output of
  // the error should be sufficient!
  //  Assert (relative_residual < threshold*x.l2_norm(), ExcInternalError());
}


template <int dim>
void
test()
{
  // make a coarse triangulation as a
  // hypercube. if in more than 1d,
  // distort it so that it is no more
  // an affine image of the
  // hypercube, to make things more
  // difficult. then refine it twice
  Triangulation<dim> tr;
  GridGenerator::reference_cell(tr, ReferenceCells::Pyramid);

  tr.refine_global(1);

  // now for a list of finite
  // elements, for which we want to
  // test. we happily waste tons of
  // memory here, but who cares...
  std::vector<hp::FECollection<dim>> fe_collections;

  for (unsigned int fe_degree = 1; fe_degree < 3; ++fe_degree)
    {
      const FE_SimplexP<dim> fe_p(fe_degree, false);
      const FE_PyramidP<dim> fe_pyramid(fe_degree, false);
      fe_collections.emplace_back(fe_pyramid, fe_p);
    }

  for (unsigned int fe_degree = 1; fe_degree < 3; ++fe_degree)
    {
      const FE_SimplexDGP<dim> fe_p(fe_degree, false);
      const FE_PyramidDGP<dim> fe_pyramid(fe_degree, false);
      fe_collections.emplace_back(fe_pyramid, fe_p);
    }

  unsigned int counter = 0;
  for (const auto &fe_collection : fe_collections)
    {
      deallog << dim << "d, uniform grid, fe #" << counter++;
      check_element(tr, fe_collection);
    }
}



int
main()
{
  std::ofstream logfile("output");
  deallog << std::setprecision(PRECISION);
  deallog << std::fixed;
  deallog.attach(logfile);

  test<3>();

  return 0;
}
