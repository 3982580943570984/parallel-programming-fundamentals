#include <mpi.h>

#include <algorithm>
#include <limits>
#include <print>
#include <ranges>
#include <vector>

int main(int argc, char** argv)
{
  using namespace std::views;
  using std::ranges::to;

  MPI_Init(&argc, &argv);

  int size {};
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank {};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto array { iota(0) | take(size) | reverse | to<std::vector>() };

  if (rank == 0)
  {
    std::print("Изначальный массив: ");
    for (auto const& element : array) std::print("{} ", element);
    std::println();
  }

  int dimensions[] { size }, periods[] {};
  MPI_Comm communicator {};
  MPI_Cart_create(MPI_COMM_WORLD, std::size(dimensions), dimensions, periods, 0,
                  &communicator);

  int previous {}, next {};
  MPI_Cart_shift(communicator, 0, 1, &previous, &next);

  int min { std::numeric_limits<int>::max() }, max {};

  for (auto const& value : array)
  {
    rank == 0 ? max = value
              : MPI_Recv(&max, 1, MPI_INT, previous, 0, communicator,
                         MPI_STATUS_IGNORE);

    if (min > max) std::swap(min, max);

    if (next > 0) MPI_Send(&max, 1, MPI_INT, next, 0, communicator);
  }

  MPI_Gather(&min, 1, MPI_INT, array.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    std::print("Результирующий массив: ");
    for (auto const& element : array) std::print("{} ", element);
    std::println();
  }

  MPI_Finalize();
}