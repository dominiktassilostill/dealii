// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2009 - 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



// check that DoFs adjust_quad_dof_index_for_face_orientation gives consistent
// results for the triangular and quadrilateral faces of FE_WedgeP and
// FE_PyramidP


#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_wedge_p.h>

#include <deal.II/grid/reference_cell.h>

#include "../tests.h"


template <int dim>
void
test(const FiniteElement<dim> &fe)
{
  const FE_SimplexP<dim> fe_p(fe.degree);
  const FE_Q<dim>        fe_q(fe.degree);

  const auto reference_cell = fe.reference_cell();

  deallog << "Testing adjust_quad_dof_index_for_face_orientation for "
          << fe.get_name() << std::endl;

  for (const auto f : reference_cell.face_indices())
    for (unsigned int i = 0; i < fe.n_dofs_per_quad(f); ++i)
      for (types::geometric_orientation o = 0;
           o < reference_cell.n_face_orientations(f);
           ++o)
        {
          const unsigned int adjusted_index =
            fe.adjust_quad_dof_index_for_face_orientation(i, f, o);

          const unsigned int adjusted_index_reference =
            reference_cell.face_reference_cell(f).is_hyper_cube() ?
              fe_q.adjust_quad_dof_index_for_face_orientation(i, 0, o) :
              fe_p.adjust_quad_dof_index_for_face_orientation(i, 0, o);

          if (adjusted_index == adjusted_index_reference)
            deallog << "ok ";
          else
            deallog << "error in face, index, orientation: " << f << " " << i
                    << " " << std::to_string(o) << std::endl;
        }
  deallog << std::endl;
  deallog << std::endl;
}


int
main()
{
  initlog();

  // test FE_WedgeP
  for (unsigned int i = 1; i < 3; ++i)
    {
      const FE_WedgeP<3> fe(i);
      test<3>(fe);
    }

  // test FE_PyramidP
  for (unsigned int i = 1; i < 4; ++i)
    {
      const FE_PyramidP<3> fe(i);
      test<3>(fe);
    }

  return 0;
}
