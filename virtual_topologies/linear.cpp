#include <mpi.h>

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

  std::vector<double> matrix(size * size), vector(size), column(size);

  if (rank == 0)
  {
    matrix = std::views::iota(0) | std::views::take(size * size) |
             std::ranges::to<std::vector<double>>();

    vector = std::views::iota(0) | std::views::take(size) |
             std::ranges::to<std::vector<double>>();
  }

  std::vector<double> transposed_matrix(matrix.size());

  for (int i {}; i < size; ++i)
    for (int j {}; j < size; ++j)
      transposed_matrix[j * size + i] = matrix[i * size + j];

  MPI_Scatter(transposed_matrix.data(), size, MPI_DOUBLE, column.data(),
              column.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Comm communicator {};
  int dimensions[] { size }, periods[] { 0 };
  MPI_Cart_create(MPI_COMM_WORLD, 1, dimensions, periods, 0, &communicator);

  int previous_rank {}, next_rank {};
  MPI_Cart_shift(communicator, 0, 1, &previous_rank, &next_rank);

  double result {};

  for (int i {}; i < size; ++i)
  {
    double element {};

    rank == 0 ? element = vector[i]
              : MPI_Recv(&element, 1, MPI_DOUBLE, previous_rank, 0,
                         communicator, MPI_STATUS_IGNORE);

    result += column[i] * element;

    if (next_rank > 0)
      MPI_Send(&element, 1, MPI_DOUBLE, next_rank, 0, communicator);
  }

  std::vector<double> results(size);
  MPI_Gather(&result, 1, MPI_DOUBLE, results.data(), 1, MPI_DOUBLE, 0,
             communicator);

  if (rank == 0)
  {
    std::print("Result (Linear): ");

    for (double result : results) std::print("{} ", result);

    std::println();
  }

  MPI_Finalize();
}