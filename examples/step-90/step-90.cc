#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/conditional_ostream.h>

void test()
{
  std::cout << "Test serial" << std::endl;
  constexpr int              dim = 3;
  dealii::Triangulation<dim> tria;
  dealii::GridGenerator::reference_cell(tria, dealii::ReferenceCells::Pyramid);

  {
    std::cout
      << "Reference cells before refinement given by tria.get_reference_cells():"
      << std::endl;

    const auto &ref_cells = tria.get_reference_cells();
    for (const auto &r : ref_cells)
      std::cout << r.to_string() << ", ";
    std::cout << std::endl;

    std::cout << "Reference cells before refinement from cells:" << std::endl;
    for (const auto &cell : tria.active_cell_iterators())
      std::cout << cell->reference_cell().to_string() << ", ";
    std::cout << std::endl;
    std::cout << std::endl;
  }

  tria.refine_global(1);

  {
    std::cout
      << "Reference cells after refinement given by tria.get_reference_cells():"
      << std::endl;

    const auto &ref_cells = tria.get_reference_cells();
    for (const auto &r : ref_cells)
      std::cout << r.to_string() << ", ";
    std::cout << std::endl;

    std::cout << "Reference cells after refinement from cells:" << std::endl;
    for (const auto &cell : tria.active_cell_iterators())
      std::cout << cell->reference_cell().to_string() << ", ";
    std::cout << std::endl;
    std::cout << std::endl;
  }
}

void test_distributed()
{
  constexpr int dim = 3;

  dealii::parallel::fullydistributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  dealii::ConditionalOStream pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(
                                     tria.get_mpi_communicator()) == 0);

  pcout << "Test distributed on "
        << dealii::Utilities::MPI::n_mpi_processes(tria.get_mpi_communicator())
        << std::endl
        << std::endl;

  const unsigned int mpi_process =
    dealii::Utilities::MPI::this_mpi_process(tria.get_mpi_communicator());

  unsigned int refinements = 0;

  const auto serial_grid_generator =
    [&refinements](dealii::Triangulation<dim, dim> &tria_serial) {
      // set up triangulation
      dealii::GridGenerator::reference_cell(tria_serial,
                                            dealii::ReferenceCells::Pyramid);
      if (refinements > 0)
        tria_serial.refine_global(refinements);
    };
  const auto serial_grid_partitioner =
    [&](dealii::Triangulation<dim, dim> &tria_serial,
        const MPI_Comm                   comm,
        const unsigned int) {
      dealii::GridTools::partition_triangulation_zorder(
        dealii::Utilities::MPI::n_mpi_processes(comm), tria_serial);
    };

  const unsigned int group_size = 40;

  typename dealii::TriangulationDescription::Settings
    triangulation_description_setting =
      dealii::TriangulationDescription::default_setting;
  const auto description = dealii::TriangulationDescription::Utilities::
    create_description_from_triangulation_in_groups<dim, dim>(
      serial_grid_generator,
      serial_grid_partitioner,
      tria.get_mpi_communicator(),
      group_size,
      dealii::Triangulation<dim>::none,
      triangulation_description_setting);

  tria.clear();
  tria.create_triangulation(description);
  MPI_Barrier(tria.get_mpi_communicator());

  std::cout
    << mpi_process << ": "
    << "Reference cells before refinement given by tria.get_reference_cells()"
    << std::endl;
  MPI_Barrier(tria.get_mpi_communicator());

  {
    const auto &ref_cells = tria.get_reference_cells();
    for (const auto &r : ref_cells)
      std::cout << mpi_process << ": " << r.to_string() << std::endl;
  }
  MPI_Barrier(tria.get_mpi_communicator());

  std::cout << mpi_process << ": "
            << "Reference cells before refinement from cells:" << std::endl;
  {
    for (const auto &cell : tria.active_cell_iterators())
      if (!cell->is_artificial())
        std::cout << mpi_process << ": " << cell->reference_cell().to_string()
                  << std::endl;
  }

  std::cout << std::endl;

  tria.clear();
  refinements                    = 1;
  const auto description_refined = dealii::TriangulationDescription::Utilities::
    create_description_from_triangulation_in_groups<dim, dim>(
      serial_grid_generator,
      serial_grid_partitioner,
      tria.get_mpi_communicator(),
      group_size,
      dealii::Triangulation<dim>::none,
      triangulation_description_setting);
  tria.create_triangulation(description_refined);

  std::cout
    << mpi_process << ": "
    << "Reference cells after refinement given by tria.get_reference_cells():"
    << std::endl;
  MPI_Barrier(tria.get_mpi_communicator());

  {
    const auto &ref_cells = tria.get_reference_cells();
    for (const auto &r : ref_cells)
      std::cout << mpi_process << ": " << r.to_string() << std::endl;
  }

  std::cout << mpi_process << ": "
            << "Reference cells after refinement from cells:" << std::endl;
  {
    for (const auto &cell : tria.active_cell_iterators())
      if (!cell->is_artificial())
        std::cout << mpi_process << ": " << cell->reference_cell().to_string()
                  << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
    test();
  else
    test_distributed();

  return 0;
}
