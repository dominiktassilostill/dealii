// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2001 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// test the deformation of a circular annulus to a domain where the central
// circle is displaced


#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include "../tests.h"



int
main()
{
  const unsigned int dim = 2;
  Point<dim>         origin;
  MappingQ<dim>      mapping(2);
  Triangulation<dim> tria;
  const double       inner_radius = 1.;
  const double       outer_radius = 5.;
  GridGenerator::hyper_shell(tria, origin, inner_radius, outer_radius, 8);
  // restore compatibility with the pre-9.0 version of GridGenerator by
  // resetting manifolds
  tria.set_all_manifold_ids(numbers::flat_manifold_id);
  tria.set_all_manifold_ids_on_boundary(0);
  tria.refine_global(2);
  // We will move the boundary faces below in the laplace smoothing algorithm:
  // for now reset all manifold IDs.
  tria.set_all_manifold_ids(numbers::flat_manifold_id);

  // build up a map of vertex indices
  // of boundary vertices to the new
  // boundary points
  std::map<unsigned int, Point<dim>> new_points;

  // new center and new radius
  // of the inner circle.
  const Point<dim> n_center(0, -1);
  const double     n_radius = 0.5;

  Triangulation<dim>::cell_iterator cell = tria.begin_active(),
                                    endc = tria.end();
  Triangulation<dim>::face_iterator face;
  for (; cell != endc; ++cell)
    {
      if (cell->at_boundary())
        for (const unsigned int face_no : GeometryInfo<dim>::face_indices())
          {
            face = cell->face(face_no);
            if (face->at_boundary())
              for (unsigned int vertex_no = 0;
                   vertex_no < GeometryInfo<dim>::vertices_per_face;
                   ++vertex_no)
                {
                  const Point<dim> &v = face->vertex(vertex_no);
                  if (std::fabs(std::sqrt(v.square()) - outer_radius) < 1e-12)
                    {
                      // leave the
                      // point, where
                      // they are.
                      new_points.insert(
                        std::pair<types::global_dof_index, Point<dim>>(
                          face->vertex_index(vertex_no), v));
                      face->set_manifold_id(0);
                    }
                  else if (std::fabs(std::sqrt(v.square()) - inner_radius) <
                           1e-12)
                    {
                      // move the
                      // center of
                      // the inner
                      // circle to
                      // (-1,0) and
                      // take half
                      // the radius
                      // of the
                      // circle.
                      new_points.insert(
                        std::pair<types::global_dof_index, Point<dim>>(
                          face->vertex_index(vertex_no),
                          n_radius / inner_radius * v + n_center));
                      face->set_manifold_id(1);
                    }
                  else
                    Assert(false, ExcInternalError());
                }
          }
    }
  GridTools::copy_boundary_to_manifold_id(tria);
  SphericalManifold<dim> inner_ball(n_center);
  tria.set_manifold(1, inner_ball);
  SphericalManifold<dim> outer_ball(origin);
  tria.set_manifold(0, outer_ball);
  GridTools::laplace_transform(new_points, tria);

  std::ofstream eps_stream2("output");
  GridOut       grid_out;
  grid_out.write_eps(tria, eps_stream2, &mapping);

  tria.clear();

  return 0;
}
