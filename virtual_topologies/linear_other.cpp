#include <mpi.h>

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

  std::vector<int> matrix(size * size), vector(size), column(size);

  if (rank == 0)
  {
    matrix = iota(0) | take(size * size) | to<std::vector>();

    vector = iota(0) | take(size) | to<std::vector>();

    std::println("Исходная матрица:");
    for (auto const& row : matrix | chunk(size))
    {
      for (auto const& value : row) std::print("{} ", value);
      std::println();
    }

    std::println("Исходный вектор:");
    for (auto const& value : vector) std::print("{} ", value);
    std::println();
  }

  MPI_Scatter(matrix.data(), size, MPI_INT, column.data(), column.size(),
              MPI_INT, 0, MPI_COMM_WORLD);

  int vector_element {};
  MPI_Scatter(vector.data(), 1, MPI_INT, &vector_element, 1, MPI_INT, 0,
              MPI_COMM_WORLD);

  auto const multiplication {
    column | transform([&](auto const& column_element) {
      return column_element * vector_element;
    }) | to<std::vector>(),
  };

  MPI_Comm communicator {};
  int dimensions[] { size }, periods[] { 0 };
  MPI_Cart_create(MPI_COMM_WORLD, 1, dimensions, periods, 0, &communicator);

  int previous_rank {}, next_rank {};
  MPI_Cart_shift(communicator, 0, 1, &previous_rank, &next_rank);

  std::vector<int> results(size);

  for (int i {}; i < size; ++i)
  {
    int previous {};

    if (rank != 0)
      MPI_Recv(&previous, 1, MPI_INT, previous_rank, 0, communicator,
               MPI_STATUS_IGNORE);

    int result { previous + multiplication.at(i) };

    if (next_rank > 0)
      MPI_Send(&result, 1, MPI_INT, next_rank, 0, communicator);

    if (rank == 4) results.at(i) = result;
  }

  if (rank == 4)
  {
    std::print("Результат: ");
    for (auto const& result : results) std::print("{} ", result);
    std::println();
  }

  MPI_Finalize();
}