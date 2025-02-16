#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <print>
#include <ranges>
#include <vector>

constexpr auto stages { 3 };
constexpr auto cars { 5 };

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank {};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size {};
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Comm cars_and_arbiter {};
  MPI_Comm_split(MPI_COMM_WORLD, rank != 0, rank, &cars_and_arbiter);

  if (rank == 0)
  {
    std::vector<int> results(size);
    std::vector<int> points(size);

    for (int stage { 0 }; stage < stages; ++stage)
    {
      int start_signal {};
      MPI_Bcast(&start_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);

      std::println("Арбитр: Этап {} - рассылка сигнала старта", stage);

      MPI_Gather(MPI_IN_PLACE, 0, nullptr, results.data(), 1, MPI_INT, 0,
                 MPI_COMM_WORLD);

      std::println("Результаты этапа {}", stage);
      for (int car { 1 };
           auto const &[result, point] :
           std::ranges::zip_view { results, points } | std::views::drop(1))
      {
        std::println("Арбитр: Машина {}, результат {}", ++car, result);
        point += result;
      }

      std::println("Арбитр: Этап {} - все результаты получены", stage);
    }

    std::vector<std::pair<int, int>> final_results;
    for (int car { 1 }; car < size; ++car)
      final_results.emplace_back(points[car], car);

    std::ranges::sort(final_results);

    std::println("Итоговые результаты:");
    for (int place { 1 }; auto const &[score, car] : final_results)
      std::println("Место {}: Машина {} (Всего очков: {})", place++, car,
                   score);
  }
  else
  {
    std::srand(std::time(nullptr) + rank);

    for (int stage { 0 }; stage < stages; ++stage)
    {
      int start_signal {};
      MPI_Bcast(&start_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);

      std::println("Машина {}: Этап {} - сигнал старта получен", rank, stage);

      auto const time { std::rand() % 10 + 1 };
      sleep((int)time);

      MPI_Gather(&time, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);

      std::println("Машина {}: Этап {} - результат отправлен арбитру", rank,
                   stage);

      MPI_Barrier(cars_and_arbiter);
    }
  }

  MPI_Comm_free(&cars_and_arbiter);

  MPI_Finalize();
}