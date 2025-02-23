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

  MPI_Scatter(matrix.data(), size, MPI_DOUBLE, column.data(), column.size(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double vector_element {};
  MPI_Scatter(vector.data(), 1, MPI_DOUBLE, &vector_element, 1, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  auto const multiplication {
    column | std::views::transform([&](auto const& column_element) {
      return column_element * vector_element;
    }) |
    std::ranges::to<std::vector<double>>()
  };

  MPI_Comm communicator {};
  int dimensions[] { size }, periods[] { 1 };
  MPI_Cart_create(MPI_COMM_WORLD, 1, dimensions, periods, 0, &communicator);

  int previous_rank {}, next_rank {};
  MPI_Cart_shift(communicator, 0, 1, &previous_rank, &next_rank);

  std::vector<double> results(size);

  for (int i {}; i < size; ++i)
  {
    double previous {};

    if (rank != 0)
      MPI_Recv(&previous, 1, MPI_DOUBLE, previous_rank, 0, communicator,
               MPI_STATUS_IGNORE);

    double result { previous + multiplication.at(i) };

    MPI_Send(&result, 1, MPI_DOUBLE, next_rank, 0, communicator);

    if (rank == 0)
      MPI_Recv(&results.at(i), 1, MPI_DOUBLE, previous_rank, 0, communicator,
               MPI_STATUS_IGNORE);
  }

  if (rank == 0)
  {
    std::print("Result (Ring): ");

    for (double result : results) std::print("{} ", result);

    std::println();
  }

  MPI_Finalize();
}