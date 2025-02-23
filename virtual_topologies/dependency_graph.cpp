#include <mpi.h>

#include <algorithm>
#include <cmath>
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

  int dimensions[] {
    static_cast<int>(std::sqrt(size)),
    static_cast<int>(std::sqrt(size)),
  },
      periods[] { 0, 0 };
  MPI_Comm communicator {};
  MPI_Cart_create(MPI_COMM_WORLD, std::size(dimensions), dimensions, periods, 0,
                  &communicator);

  int min {}, max { array.front() }, coordinates[2] {};

  MPI_Cart_coords(communicator, rank, std::size(dimensions), coordinates);

  // Код выполняется только для процессов выше или на главной диагонали сетки
  auto const& [x, y] { coordinates };
  if (x <= y)
  {
    int x_previous {}, x_next {};
    MPI_Cart_shift(communicator, 0, 1, &x_previous, &x_next);

    int y_previous {}, y_next {};
    MPI_Cart_shift(communicator, 1, 1, &y_previous, &y_next);

    //           y_previous
    //               ---
    // x_previous ---   --- x_next
    //               ---
    //              y_next

    //             24
    //          18 19
    //       12 13 14
    //    6  7  8  9
    // 0  1  2  3  4

    // Процессы на диагонали начинают с максимально возможного значения
    (x == y) ? min = std::numeric_limits<int>::max()
             : MPI_Recv(&min, 1, MPI_INT, y_previous, 0, communicator,
                        MPI_STATUS_IGNORE);

    if (x == 0 && y == 0)
      for (int i { 1 }; i < size; ++i)
        MPI_Send(&array.at(i), 1, MPI_INT, i, 0, communicator);

    if (x == 0 && y != 0)
      MPI_Recv(&max, 1, MPI_INT, 0, 0, communicator, MPI_STATUS_IGNORE);

    if (min > max) std::swap(min, max);

    MPI_Send(&min, 1, MPI_INT, (y_next < 0) ? 0 : y_next, 0, communicator);

    if (x != y && x_next > 0)
      MPI_Send(&max, 1, MPI_INT, x_next, 0, communicator);

    if (rank == 0)
    {
      std::vector<int> result {};
      result.resize(size);

      // 20 21 22 23 24
      // 15 16 17 18 19
      // 10 11 12 13 14
      // 5  6  7  8  9
      // 0  1  2  3  4

      for (int i {}, z { size - 1 }; i < size; ++i, z += size)
        MPI_Recv(&result.at(i), 1, MPI_INT, z, 0, communicator,
                 MPI_STATUS_IGNORE);

      std::print("Результат: ");
      for (auto const& element : result) std::print("{} ", element);
      std::println();
    }
  }

  MPI_Finalize();
}