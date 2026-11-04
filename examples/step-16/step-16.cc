// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2020 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


/**
 * Test prolongation with uniformly refined meshes.
 *
 * To compare with non_nested_transfer_04.
 */

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_pyramid_p.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>


using namespace dealii;

template <int dim>
class RightHandSideFunction : public Function<dim>
{
public:
  RightHandSideFunction()
    : Function<dim>(1)
  {}

  virtual double value(const Point<dim>  &p,
                       const unsigned int component = 0) const override
  {
    (void)component;
    (void)p;
    return p[0] + p[1] + p[2];
  }
};

template <typename Number = double>
void print(const LinearAlgebra::distributed::Vector<Number> &vec)
{
  for (unsigned int i = 0; i < vec.locally_owned_size(); ++i)
    std::cout << vec.local_element(i) << " ";
  std::cout << std::endl;
}


template <typename Number = double>
void print(const Vector<Number> &vec)
{
  for (unsigned int i = 0; i < vec.size(); ++i)
    std::cout << vec[i] << " ";
  std::cout << std::endl;
}


template <int dim, typename Number, typename MeshType>
void test_pyramid_prolongation_manual(
  const MeshType                  &dof_handler_fine,
  const MeshType                  &dof_handler_coarse,
  const AffineConstraints<Number> &constraint_fine,
  const AffineConstraints<Number> &constraint_coarse,
  const Function<dim, Number>     &function,
  Vector<Number>                  &vector_fine,
  Vector<Number>                  &vector_coarse)
{
  (void)constraint_fine;
  (void)constraint_coarse;

  unsigned int pyramid_index = numbers::invalid_unsigned_int;
  for (unsigned int i = 0; i < dof_handler_fine.get_fe_collection().size(); ++i)
    if (dof_handler_fine.get_fe(i).reference_cell() == ReferenceCells::Pyramid)
      pyramid_index = i;

  Assert(pyramid_index != numbers::invalid_unsigned_int, ExcInternalError());

  const auto &fe_pyramid = dof_handler_fine.get_fe(pyramid_index);

  const bool is_dg = fe_pyramid.n_dofs_per_vertex() == 0;

  std::vector<Number> weights;
  if (!is_dg)
    {
      weights.resize(dof_handler_fine.n_dofs(), 0.0);

      for (const auto &cell : dof_handler_fine.active_cell_iterators())
        {
          std::vector<types::global_dof_index> dof_indices(
            cell->get_fe().dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          for (const auto &d : dof_indices)
            weights[d] += 1;
        }

      for (unsigned int i = 0; i < weights.size(); ++i)
        if (weights[i] != 0.0)
          weights[i] = 1.0 / weights[i];
    }

  vector_coarse = 0.0;
  VectorTools::interpolate(dof_handler_coarse, function, vector_coarse);
  vector_fine = 0.0;

  const auto parent = dof_handler_fine.begin(0);
  for (unsigned int c = 0; c < parent->n_children(); ++c)
    {
      const auto &cell = parent->child(c);

      Vector<double> vector_child(cell->get_fe().dofs_per_cell);
      std::vector<types::global_dof_index> dof_indices(
        cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(dof_indices);

      fe_pyramid.get_prolongation_matrix(c).vmult(vector_child, vector_coarse);

      for (unsigned int v = 0; v < dof_indices.size(); ++v)
        {
          vector_fine[dof_indices[v]] += vector_child[v];
        }
    }
  if (weights.empty() == false)
    for (unsigned int i = 0; i < weights.size(); ++i)
      vector_fine[i] *= weights[i];

  print(vector_coarse);
  print(vector_fine);
  std::cout << std::endl;
}


template <int dim, typename Number, typename MeshType>
void test_pyramid_restriction_manual(
  const MeshType                  &dof_handler_fine,
  const MeshType                  &dof_handler_coarse,
  const AffineConstraints<Number> &constraint_fine,
  const AffineConstraints<Number> &constraint_coarse,
  const Function<dim, Number>     &function,
  Vector<Number>                  &vector_fine,
  Vector<Number>                  &vector_coarse)
{
  (void)constraint_fine;
  (void)constraint_coarse;
  (void)dof_handler_coarse;
  (void)function;

  unsigned int pyramid_index = numbers::invalid_unsigned_int;
  for (unsigned int i = 0; i < dof_handler_fine.get_fe_collection().size(); ++i)
    if (dof_handler_fine.get_fe(i).reference_cell() == ReferenceCells::Pyramid)
      pyramid_index = i;

  Assert(pyramid_index != numbers::invalid_unsigned_int, ExcInternalError());

  const auto &fe_pyramid = dof_handler_fine.get_fe(pyramid_index);

  const bool is_dg = fe_pyramid.n_dofs_per_vertex() == 0;

  std::vector<Number> weights;
  if (!is_dg)
    {
      weights.resize(dof_handler_fine.n_dofs(), 0.0);

      for (const auto &cell : dof_handler_fine.active_cell_iterators())
        {
          std::vector<types::global_dof_index> dof_indices(
            cell->get_fe().dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          for (const auto &d : dof_indices)
            weights[d] += 1;
        }

      for (unsigned int i = 0; i < weights.size(); ++i)
        if (weights[i] != 0.0)
          weights[i] = 1.0 / weights[i];
    }

  print(vector_fine);

  if (weights.empty() == false)
    for (unsigned int i = 0; i < weights.size(); ++i)
      vector_fine[i] *= weights[i];

  vector_coarse = 0.0;

  const auto parent = dof_handler_fine.begin(0);
  for (unsigned int c = 0; c < parent->n_children(); ++c)
    {
      const auto &cell = parent->child(c);

      Vector<double> vector_child(cell->get_fe().dofs_per_cell);
      Vector<double> vector_parent(fe_pyramid.dofs_per_cell);
      std::vector<types::global_dof_index> dof_indices(
        cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(dof_indices);
      for (unsigned int v = 0; v < dof_indices.size(); ++v)
        vector_child[v] = vector_fine[dof_indices[v]];

      fe_pyramid.get_prolongation_matrix(c).Tvmult_add(vector_coarse,
                                                       vector_child);
    }

  print(vector_coarse);
  std::cout << std::endl;
}

template <int dim, typename Number, typename MeshType>
void test_prolongation(
  const MGTwoLevelTransferBase<dim, LinearAlgebra::distributed::Vector<Number>>
                                             &transfer,
  const MeshType                             &dof_handler_fine,
  const MeshType                             &dof_handler_coarse,
  const AffineConstraints<Number>            &constraint_fine,
  const AffineConstraints<Number>            &constraint_coarse,
  const Function<dim, Number>                &function,
  LinearAlgebra::distributed::Vector<Number> &dst,
  LinearAlgebra::distributed::Vector<Number> &src)
{
  (void)dof_handler_fine;
  (void)constraint_coarse;
  // test prolongation
  {
    src = 0.0;
    VectorTools::interpolate(dof_handler_coarse, function, src);
    dst = 0.0;
    transfer.prolongate_and_add(dst, src);

    // transfer operator sets only non-constrained dofs -> update the rest
    // via constraint matrix
    constraint_fine.distribute(dst);

    // print vectors
    print(src);
    print(dst);
    std::cout << std::endl;

    // print full prolongation matrix
    if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1 && false)
      {
        FullMatrix<Number> prolongation_matrix(dst.size(), src.size());
        for (unsigned int i = 0; i < src.size(); ++i)
          {
            src    = 0.0;
            src[i] = 1.0;
            dst    = 0.0;

            transfer.prolongate_and_add(dst, src);

            for (unsigned int j = 0; j < dst.size(); ++j)
              prolongation_matrix[j][i] = dst[j];
          }

        prolongation_matrix.print_formatted(
          std::cout, 2, false, 5, "", 1, 1e-5);
      }
  }
}


template <int dim, typename Number>
void test_restriction(
  const MGTwoLevelTransferBase<dim, LinearAlgebra::distributed::Vector<Number>>
                                             &transfer,
  LinearAlgebra::distributed::Vector<Number> &dst,
  LinearAlgebra::distributed::Vector<Number> &src)
{
  // test restriction
  {
    src = 0.0;
    transfer.restrict_and_add(src, dst);

    // print vectors
    print(dst);
    print(src);
    std::cout << std::endl;

    // print full restriction matrix
    if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1 && false)
      {
        FullMatrix<Number> restriction_matrix(src.size(), dst.size());
        for (unsigned int i = 0; i < dst.size(); ++i)
          {
            dst    = 0.0;
            dst[i] = 1.0;
            src    = 0.0;

            transfer.restrict_and_add(src, dst);

            for (unsigned int j = 0; j < src.size(); ++j)
              restriction_matrix[j][i] = src[j];
          }

        restriction_matrix.print_formatted(std::cout, 2, false, 5, "", 1, 1e-5);
      }
  }
}



template <int dim, typename Number, typename MeshType>
void test_hex_prolongation_manual(
  const MeshType                  &dof_handler_fine,
  const MeshType                  &dof_handler_coarse,
  const AffineConstraints<Number> &constraint_fine,
  const AffineConstraints<Number> &constraint_coarse,
  const Function<dim, Number>     &function,
  Vector<Number>                  &vector_fine,
  Vector<Number>                  &vector_coarse)
{
  (void)constraint_fine;
  (void)constraint_coarse;


  const auto &feq = dof_handler_fine.get_fe(0);

  const bool is_dg = feq.n_dofs_per_vertex() == 0;

  std::vector<Number> weights;
  if (!is_dg)
    {
      weights.resize(dof_handler_fine.n_dofs(), 0.0);

      for (const auto &cell : dof_handler_fine.active_cell_iterators())
        {
          std::vector<types::global_dof_index> dof_indices(
            cell->get_fe().dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          for (const auto &d : dof_indices)
            weights[d] += 1;
        }

      for (unsigned int i = 0; i < weights.size(); ++i)
        if (weights[i] != 0.0)
          weights[i] = 1.0 / weights[i];
    }

  vector_coarse = 0.0;
  VectorTools::interpolate(dof_handler_coarse, function, vector_coarse);
  vector_fine = 0.0;

  const auto parent = dof_handler_fine.begin(0);
  for (unsigned int c = 0; c < parent->n_children(); ++c)
    {
      const auto &cell = parent->child(c);

      Vector<double> vector_child(cell->get_fe().dofs_per_cell);
      std::vector<types::global_dof_index> dof_indices(
        cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(dof_indices);

      feq.get_prolongation_matrix(c, parent->refinement_case())
        .vmult(vector_child, vector_coarse);

      for (unsigned int v = 0; v < dof_indices.size(); ++v)
        {
          vector_fine[dof_indices[v]] += vector_child[v];
        }
    }
  if (weights.empty() == false)
    for (unsigned int i = 0; i < weights.size(); ++i)
      vector_fine[i] *= weights[i];

  print(vector_coarse);
  print(vector_fine);
  std::cout << std::endl;
}


template <int dim, typename Number, typename MeshType>
void test_hex_restriction_manual(
  const MeshType                  &dof_handler_fine,
  const MeshType                  &dof_handler_coarse,
  const AffineConstraints<Number> &constraint_fine,
  const AffineConstraints<Number> &constraint_coarse,
  const Function<dim, Number>     &function,
  Vector<Number>                  &vector_fine,
  Vector<Number>                  &vector_coarse)
{
  (void)constraint_fine;
  (void)constraint_coarse;
  (void)dof_handler_coarse;
  (void)function;


  const auto &feq = dof_handler_fine.get_fe(0);

  const bool is_dg = feq.n_dofs_per_vertex() == 0;

  std::vector<Number> weights;
  if (!is_dg)
    {
      weights.resize(dof_handler_fine.n_dofs(), 0.0);

      for (const auto &cell : dof_handler_fine.active_cell_iterators())
        {
          std::vector<types::global_dof_index> dof_indices(
            cell->get_fe().dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          for (const auto &d : dof_indices)
            weights[d] += 1;
        }

      for (unsigned int i = 0; i < weights.size(); ++i)
        if (weights[i] != 0.0)
          weights[i] = 1.0 / weights[i];
    }

  print(vector_fine);

  if (weights.empty() == false)
    for (unsigned int i = 0; i < weights.size(); ++i)
      vector_fine[i] *= weights[i];

  vector_coarse = 0.0;

  const auto parent = dof_handler_fine.begin(0);
  for (unsigned int c = 0; c < parent->n_children(); ++c)
    {
      const auto &cell = parent->child(c);

      Vector<double> vector_child(cell->get_fe().dofs_per_cell);
      std::vector<types::global_dof_index> dof_indices(
        cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(dof_indices);
      for (unsigned int v = 0; v < dof_indices.size(); ++v)
        vector_child[v] = vector_fine[dof_indices[v]];

      feq.get_prolongation_matrix(c, parent->refinement_case())
        .Tvmult_add(vector_coarse, vector_child);
    }

  print(vector_coarse);
  std::cout << std::endl;
}


template <int dim, typename Number>
void do_test_pyramid(const unsigned int           fe_degree,
                     const Function<dim, Number> &function,
                     const bool                   use_dg)
{
  std::cout << "Degree " << fe_degree << std::endl;
  // create coarse grid
  Triangulation<dim> tria_coarse;
  GridGenerator::reference_cell(tria_coarse, ReferenceCells::Pyramid);

  // create fine grid
  Triangulation<dim> tria_fine;
  GridGenerator::reference_cell(tria_fine, ReferenceCells::Pyramid);
  tria_fine.refine_global(1);

  std::map<ReferenceCell<dim>, types::fe_index> ref_cell_to_fe_index;
  const auto &ref_cells = tria_fine.get_reference_cells();
  for (unsigned int i = 0; i < ref_cells.size(); ++i)
    ref_cell_to_fe_index[ref_cells[i]] = i;

  hp::FECollection<dim> fe;
  if (use_dg)
    {
      for (const auto &r : ref_cells)
        {
          if (r.is_simplex())
            fe.push_back(FE_SimplexDGP<dim>(fe_degree, false));
          else if (r.is_hyper_cube())
            fe.push_back(FE_DGQ<dim>(fe_degree));
          else if (r == ReferenceCells::Wedge)
            fe.push_back(FE_WedgeDGP<dim>(fe_degree, false));
          else if (r == ReferenceCells::Pyramid)
            fe.push_back(FE_PyramidDGP<dim>(fe_degree, false));
        }
    }
  else
    {
      for (const auto &r : ref_cells)
        {
          if (r.is_simplex())
            fe.push_back(FE_SimplexP<dim>(fe_degree, false));
          else if (r.is_hyper_cube())
            fe.push_back(FE_Q<dim>(fe_degree));
          else if (r == ReferenceCells::Wedge)
            fe.push_back(FE_WedgeP<dim>(fe_degree, false));
          else if (r == ReferenceCells::Pyramid)
            fe.push_back(FE_PyramidP<dim>(fe_degree, false));
        }
    }

  // setup dof-handlers
  DoFHandler<dim> dof_handler_fine(tria_fine);
  for (auto &cell : dof_handler_fine.active_cell_iterators())
    cell->set_active_fe_index(ref_cell_to_fe_index[cell->reference_cell()]);
  dof_handler_fine.distribute_dofs(fe);

  DoFHandler<dim> dof_handler_coarse(tria_coarse);
  for (auto &cell : dof_handler_coarse.active_cell_iterators())
    cell->set_active_fe_index(ref_cell_to_fe_index[cell->reference_cell()]);
  dof_handler_coarse.distribute_dofs(fe);

  // setup constraint matrix
  AffineConstraints<Number> constraint_coarse;
  constraint_coarse.close();

  AffineConstraints<Number> constraint_fine;
  constraint_fine.close();

  // setup transfer operator
  MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<Number>> transfer;

  transfer.reinit(dof_handler_fine,
                  dof_handler_coarse,
                  constraint_fine,
                  constraint_coarse);

  // perform prolongation
  LinearAlgebra::distributed::Vector<Number> src, dst;

  dst.reinit(dof_handler_fine.locally_owned_dofs(),
             DoFTools::extract_locally_relevant_dofs(dof_handler_fine),
             dof_handler_fine.get_triangulation().get_mpi_communicator());

  src.reinit(dof_handler_coarse.locally_owned_dofs(),
             DoFTools::extract_locally_relevant_dofs(dof_handler_coarse),
             dof_handler_coarse.get_triangulation().get_mpi_communicator());

  test_prolongation(transfer,
                    dof_handler_fine,
                    dof_handler_coarse,
                    constraint_fine,
                    constraint_coarse,
                    function,
                    dst,
                    src);

  Vector<Number> vector_fine(dof_handler_fine.n_dofs());
  Vector<Number> vector_coarse(dof_handler_coarse.n_dofs());
  test_pyramid_prolongation_manual(dof_handler_fine,
                                   dof_handler_coarse,
                                   constraint_fine,
                                   constraint_coarse,
                                   function,
                                   vector_fine,
                                   vector_coarse);

  for (unsigned int i = 0; i < vector_fine.size(); ++i)
    Assert(std::abs(vector_fine[i] - dst.local_element(i)) < 1e-12,
           ExcInternalError());

  test_restriction(transfer, dst, src);

  test_pyramid_restriction_manual(dof_handler_fine,
                                  dof_handler_coarse,
                                  constraint_fine,
                                  constraint_coarse,
                                  function,
                                  vector_fine,
                                  vector_coarse);

  for (unsigned int i = 0; i < vector_coarse.size(); ++i)
    Assert(std::abs(vector_coarse[i] - src.local_element(i)) < 1e-12,
           ExcInternalError());
}


template <int dim, typename Number>
void do_test_hex(const unsigned int           fe_degree,
                 const Function<dim, Number> &function,
                 const bool                   use_dg)
{
  std::cout << "Degree " << fe_degree << std::endl;
  // create coarse grid
  Triangulation<dim> tria_coarse;
  GridGenerator::reference_cell(tria_coarse, ReferenceCells::Tetrahedron);
  // GridGenerator::reference_cell(tria_coarse, ReferenceCells::Hexahedron);

  // create fine grid
  Triangulation<dim> tria_fine;
  // GridGenerator::reference_cell(tria_fine, ReferenceCells::Hexahedron);
  GridGenerator::reference_cell(tria_fine, ReferenceCells::Tetrahedron);
  tria_fine.refine_global(1);

  std::map<ReferenceCell<dim>, types::fe_index> ref_cell_to_fe_index;
  const auto &ref_cells = tria_fine.get_reference_cells();
  for (unsigned int i = 0; i < ref_cells.size(); ++i)
    ref_cell_to_fe_index[ref_cells[i]] = i;

  hp::FECollection<dim> fe;
  if (use_dg)
    {
      for (const auto &r : ref_cells)
        {
          if (r.is_simplex())
            fe.push_back(FE_SimplexDGP<dim>(fe_degree, false));
          else if (r.is_hyper_cube())
            fe.push_back(FE_DGQ<dim>(fe_degree));
          else if (r == ReferenceCells::Wedge)
            fe.push_back(FE_WedgeDGP<dim>(fe_degree, false));
          else if (r == ReferenceCells::Pyramid)
            fe.push_back(FE_PyramidDGP<dim>(fe_degree, false));
        }
    }
  else
    {
      for (const auto &r : ref_cells)
        {
          if (r.is_simplex())
            fe.push_back(FE_SimplexP<dim>(fe_degree, false));
          else if (r.is_hyper_cube())
            fe.push_back(FE_Q<dim>(fe_degree));
          else if (r == ReferenceCells::Wedge)
            fe.push_back(FE_WedgeP<dim>(fe_degree, false));
          else if (r == ReferenceCells::Pyramid)
            fe.push_back(FE_PyramidP<dim>(fe_degree, false));
        }
    }

  // setup dof-handlers
  DoFHandler<dim> dof_handler_fine(tria_fine);
  for (auto &cell : dof_handler_fine.active_cell_iterators())
    cell->set_active_fe_index(ref_cell_to_fe_index[cell->reference_cell()]);
  dof_handler_fine.distribute_dofs(fe);

  DoFHandler<dim> dof_handler_coarse(tria_coarse);
  for (auto &cell : dof_handler_coarse.active_cell_iterators())
    cell->set_active_fe_index(ref_cell_to_fe_index[cell->reference_cell()]);
  dof_handler_coarse.distribute_dofs(fe);

  // setup constraint matrix
  AffineConstraints<Number> constraint_coarse;
  constraint_coarse.close();

  AffineConstraints<Number> constraint_fine;
  constraint_fine.close();

  // setup transfer operator
  MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<Number>> transfer;

  transfer.reinit(dof_handler_fine,
                  dof_handler_coarse,
                  constraint_fine,
                  constraint_coarse);

  // perform prolongation
  LinearAlgebra::distributed::Vector<Number> src, dst;

  dst.reinit(dof_handler_fine.locally_owned_dofs(),
             DoFTools::extract_locally_relevant_dofs(dof_handler_fine),
             dof_handler_fine.get_triangulation().get_mpi_communicator());

  src.reinit(dof_handler_coarse.locally_owned_dofs(),
             DoFTools::extract_locally_relevant_dofs(dof_handler_coarse),
             dof_handler_coarse.get_triangulation().get_mpi_communicator());

  test_prolongation(transfer,
                    dof_handler_fine,
                    dof_handler_coarse,
                    constraint_fine,
                    constraint_coarse,
                    function,
                    dst,
                    src);

  Vector<Number> vector_fine(dof_handler_fine.n_dofs());
  Vector<Number> vector_coarse(dof_handler_coarse.n_dofs());
  test_hex_prolongation_manual(dof_handler_fine,
                               dof_handler_coarse,
                               constraint_fine,
                               constraint_coarse,
                               function,
                               vector_fine,
                               vector_coarse);

  for (unsigned int i = 0; i < vector_fine.size(); ++i)
    Assert(std::abs(vector_fine[i] - dst.local_element(i)) < 1e-12,
           ExcInternalError());

  test_restriction(transfer, dst, src);

  test_hex_restriction_manual(dof_handler_fine,
                              dof_handler_coarse,
                              constraint_fine,
                              constraint_coarse,
                              function,
                              vector_fine,
                              vector_coarse);

  for (unsigned int i = 0; i < vector_coarse.size(); ++i)
    Assert(std::abs(vector_coarse[i] - src.local_element(i)) < 1e-12,
           ExcInternalError());
}


template <int dim, typename Number>
void do_test(const unsigned int           fe_degree,
             const Function<dim, Number> &function)
{
  do_test_pyramid(fe_degree, function, false);
  // do_test_hex(fe_degree, function, true);
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  constexpr int              dim = 3;
  RightHandSideFunction<dim> fu;

  for (unsigned int i = 1; i <= 4; ++i)
    do_test<dim, double>(i, fu);
}
