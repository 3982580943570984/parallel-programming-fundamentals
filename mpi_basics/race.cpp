#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <print>
#include <ranges>
#include <thread>
#include <vector>

namespace
{
constexpr auto stages { 3 };
constexpr auto cars { 5 };
constexpr auto finish_line { 100 };
}  // namespace

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank {};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size {};
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Comm cars_communicator {};
  MPI_Comm_split(MPI_COMM_WORLD, rank != 0, rank, &cars_communicator);

  std::srand(time(nullptr) ^ (getpid() << 16));

  if (rank == 0)
  {
    for (int stage { 0 }; stage != stages; ++stage)
    {
      std::vector<int> progresses(size);
      auto const progresses_view { progresses | std::views::drop(1) };

      int signal {};
      MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);

      bool all_finished {};
      while (!all_finished)
      {
        MPI_Bcast(&signal, 1, MPI_INT, 0, cars_communicator);
        MPI_Gather(MPI_IN_PLACE, 0, nullptr, progresses.data(), 1, MPI_INT, 0,
                   cars_communicator);

        for (int car { 0 }; auto &progress : progresses_view)
          std::println("Арбитр: Автомобиль {}, этап {}, результат {}", ++car,
                       stage, progress);

        all_finished = std::ranges::all_of(
            progresses_view,
            [](auto const &value) { return value == finish_line; });

        if (all_finished) break;

        auto const sleep_time { (std::rand() % 200 + 100) * 1000 };
        usleep(sleep_time);
      }

      std::println("Арбитр: Этап {} - все результаты получены", stage);
    }
  }
  else
  {
    std::atomic_int progress {};

    for (int stage { 0 }; stage != stages; ++stage)
    {
      int signal {};
      MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);

      std::jthread jthread { [&] {
        while (true)
        {
          int signal {};
          MPI_Bcast(&signal, 1, MPI_INT, 0, cars_communicator);
          MPI_Gather(&progress, 1, MPI_INT, MPI_IN_PLACE, 0, nullptr, 0,
                     cars_communicator);
        }
      } };

      progress = 0;
      while (progress < finish_line)
      {
        progress += std::rand() % 10 + 1;

        if (progress > finish_line) progress = finish_line;

        usleep((std::rand() % 200 + 100) * 1000);
      }
    }
  }

  MPI_Comm_free(&cars_communicator);

  MPI_Finalize();
}