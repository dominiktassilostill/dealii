// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2009 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



// Test DoFRenumbering::block_wise. For the element used here, it
// needs to produce the exact same numbering as that for
// DoFRenumber::component_wise



#include <deal.II/base/function_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/vector.h>

#include "../tests.h"



template <int dim>
std::vector<types::global_dof_index>
get_dofs(const DoFHandler<dim> &dof)
{
  std::vector<types::global_dof_index> local;
  std::vector<types::global_dof_index> global;
  for (typename DoFHandler<dim>::active_cell_iterator cell = dof.begin_active();
       cell != dof.end();
       ++cell)
    {
      local.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local);

      global.insert(global.end(), local.begin(), local.end());
    }

  return global;
}



template <int dim>
void
check_renumbering(DoFHandler<dim> &dof)
{
  // Prepare a reordering of
  // components so that each
  // component maps to its natural
  // block
  std::vector<unsigned int> order(dof.get_fe_collection().n_components());
  order[0] = 0;
  order[1] = 1;
  order[2] = 1;

  // do component-wise and save the
  // results
  DoFRenumbering::component_wise(dof, order);
  const std::vector<types::global_dof_index> vc = get_dofs(dof);

  // now do the same with blocks
  DoFRenumbering::block_wise(dof);
  const std::vector<types::global_dof_index> vb = get_dofs(dof);

  AssertThrow(vc == vb, ExcInternalError());

  deallog << "OK" << std::endl;
}


template <int dim>
void
check()
{
  Triangulation<dim> tr;
  if (dim == 2)
    GridGenerator::hyper_ball(tr, Point<dim>(), 1);
  else
    GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(1);
  tr.begin_active()->set_refine_flag();
  tr.execute_coarsening_and_refinement();
  if (dim == 1)
    tr.refine_global(2);

  DoFHandler<dim> dof(tr);
  {
    bool coin = false;
    for (typename DoFHandler<dim>::active_cell_iterator cell =
           dof.begin_active();
         cell != dof.end();
         ++cell)
      {
        cell->set_active_fe_index(coin ? 0 : 1);
        coin = !coin;
      }
  }

  // note that the following elements
  // have 3 components but 2 blocks
  FESystem<dim> e1(FE_DGQ<dim>(1), 1, FESystem<dim>(FE_DGQ<dim>(2), 2), 1);
  FESystem<dim> e2(FE_DGQ<dim>(2), 1, FESystem<dim>(FE_DGQ<dim>(1), 2), 1);

  hp::FECollection<dim> fe_collection;
  fe_collection.push_back(e1);
  fe_collection.push_back(e2);

  dof.distribute_dofs(fe_collection);
  check_renumbering(dof);
  dof.clear();
}


int
main()
{
  initlog();
  deallog << std::setprecision(2);
  deallog << std::fixed;

  deallog.push("2d");
  check<2>();
  deallog.pop();
  deallog.push("3d");
  check<3>();
  deallog.pop();
}
