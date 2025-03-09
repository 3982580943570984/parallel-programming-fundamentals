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
    matrix = iota(0) | take(size * size) | to<std::vector<int>>();

    vector = iota(0) | take(size) | to<std::vector<int>>();

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

  std::vector<double> transposed_matrix(matrix.size());

  if (rank == 0)
    for (int i {}; i < size; ++i)
      for (int j {}; j < size; ++j)
        transposed_matrix[j * size + i] = matrix[i * size + j];

  MPI_Scatter(transposed_matrix.data(), size, MPI_INT, column.data(),
              column.size(), MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Comm communicator {};
  int dimensions[] { size }, periods[] { 0 };
  MPI_Cart_create(MPI_COMM_WORLD, 1, dimensions, periods, 0, &communicator);

  int previous_rank {}, next_rank {};
  MPI_Cart_shift(communicator, 0, 1, &previous_rank, &next_rank);

  int result {};

  for (int i {}; i < size; ++i)
  {
    int element {};

    rank == 0 ? element = vector[i]
              : MPI_Recv(&element, 1, MPI_INT, previous_rank, 0, communicator,
                         MPI_STATUS_IGNORE);

    result += column[i] * element;

    if (next_rank > 0)
      MPI_Send(&element, 1, MPI_INT, next_rank, 0, communicator);
  }

  std::vector<double> results(size);
  MPI_Gather(&result, 1, MPI_INT, results.data(), 1, MPI_INT, 0, communicator);

  if (rank == 0)
  {
    std::print("Результат: ");
    for (auto const& result : results) std::print("{} ", result);
    std::println();
  }

  MPI_Finalize();
}