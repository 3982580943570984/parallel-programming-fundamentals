#include <mpi.h>

#include <algorithm>
#include <cmath>
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

  auto array { iota(0) | take(std::sqrt(size)) | reverse | to<std::vector>() };

  if (rank == 0)
  {
    std::print("Изначальный массив: ");
    for (auto const& element : array) std::print("{} ", element);
    std::println();
  }

  int dimensions[2] {}, periods[2] {};
  MPI_Dims_create(size, std::size(dimensions), dimensions);

  MPI_Comm communicator {};
  MPI_Cart_create(MPI_COMM_WORLD, std::size(dimensions), dimensions, periods, 0,
                  &communicator);

  int coordinates[2] {};
  MPI_Cart_coords(communicator, rank, std::size(dimensions), coordinates);

  auto const& [x, y] { coordinates };

  int min { std::numeric_limits<int>::max() }, max {};

  MPI_Comm start_row {};
  MPI_Comm_split(communicator, x == 0, 0, &start_row);
  MPI_Scatter(array.data(), 1, MPI_INT, &max, 1, MPI_INT, 0, start_row);

  if (x <= y)
  {
    int x_previous {}, x_next {};
    MPI_Cart_shift(communicator, 0, 1, &x_previous, &x_next);

    int y_previous {}, y_next {};
    MPI_Cart_shift(communicator, 1, 1, &y_previous, &y_next);

    if (x != 0)
      MPI_Recv(&max, 1, MPI_INT, x_previous, 0, communicator,
               MPI_STATUS_IGNORE);

    if (x != y)
      MPI_Recv(&min, 1, MPI_INT, y_previous, 0, communicator,
               MPI_STATUS_IGNORE);

    if (min > max) std::swap(min, max);

    if (x_next > 0) MPI_Send(&max, 1, MPI_INT, x_next, 0, communicator);

    MPI_Send(&min, 1, MPI_INT, std::max(y_next, 0), 0, communicator);
  }

  if (rank == 0)
  {
    auto const sources {
      iota(dimensions[0] - 1, size) | stride(dimensions[0]) | to<std::vector>(),
    };

    for (auto const& [value, source] : zip(array, sources))
      MPI_Recv(&value, 1, MPI_INT, source, 0, communicator, MPI_STATUS_IGNORE);

    std::print("Результирующий массив: ");
    for (auto const& element : array) std::print("{} ", element);
    std::println();
  }

  MPI_Finalize();
}