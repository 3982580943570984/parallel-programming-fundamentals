#include <mpi.h>

#include <algorithm>
#include <limits>
#include <print>
#include <ranges>
#include <vector>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int size {};
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank {};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto const array { std::views::iota(0) | std::views::take(size) |
                     std::views::reverse |
                     std::ranges::to<std::vector<int>>() };

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

  int min { std::numeric_limits<int>::max() }, max {}, previous {}, next {};

  MPI_Cart_shift(communicator, 0, 1, &previous, &next);

  // 0: 3 1 2 5 4
  // 1: 2147483647 3 2 5 4
  // 2: 2147483647 2147483647 3 5 4
  // 3: 2147483647 2147483647 2147483647 5 4
  // 4: 2147483647 2147483647 2147483647 2147483647 5

  for (int i {}; i < size; ++i)
  {
    rank == 0 ? max = array.at(i)
              : MPI_Recv(&max, 1, MPI_INT, previous, 0, communicator,
                         MPI_STATUS_IGNORE);

    if (min > max) std::swap(min, max);

    if (next > 0) MPI_Send(&max, 1, MPI_INT, next, 0, communicator);
  }

  std::vector<int> result {};
  result.resize(size);

  MPI_Gather(&min, 1, MPI_INT, result.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    std::print("Результат: ");
    for (auto const& element : result) std::print("{} ", element);
    std::println();
  }

  MPI_Finalize();
}