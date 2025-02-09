#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <print>

namespace
{
constexpr auto number_of_stages { 3 };
constexpr auto number_of_cars { 5 };
constexpr auto finish_line { 100 };
constexpr auto track_length { 50 };

std::atomic_bool start_flag {};
std::atomic_bool next_flag {};
}  // namespace

template <typename T>
class MemoryMapping
{
 public:
  explicit MemoryMapping()
  {
    const auto raw_memory {
      mmap(nullptr, sizeof(T), PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_ANONYMOUS, -1, 0),
    };

    if (raw_memory == MAP_FAILED)
    {
      std::perror("mmap failed");
      std::exit(1);
    }

    object = new (raw_memory) T;
  }

  ~MemoryMapping() { munmap(object, sizeof(T)); }

  [[nodiscard]] T* get() const { return object; }

  T* operator->() const { return object; }

  T& operator*() const { return *object; }

 private:
  T* object {};
};

struct Race
{
  int current_stage {};
  int finish_order_counter {};
  std::mutex mutex {};
};

class Car
{
 public:
  void start(Race* race)
  {
    signal(SIGUSR1, [](int) { start_flag = true; });
    signal(SIGUSR2, [](int) { next_flag = true; });

    srand(time(nullptr) ^ (getpid() << 16));

    for (auto stage { 1 }; stage <= number_of_stages; ++stage)
    {
      while (!start_flag) pause();
      start_flag = false;

      while (progress < finish_line)
      {
        const auto step { rand() % 10 + 1 };
        progress += step;

        if (progress > finish_line) progress = finish_line;

        const auto sleep_time { (rand() % 200 + 100) * 1000 };
        usleep(sleep_time);
      }

      {
        std::lock_guard guard { race->mutex };
        ++race->finish_order_counter;
        order = race->finish_order_counter;
      }

      finished = true;

      if (stage == number_of_stages) continue;

      while (!next_flag) pause();
      next_flag = false;
    }
  };

 public:
  int progress {};
  int order {};
  int points {};
  bool finished {};
};

class Arbiter
{
 public:
  auto prepare()
  {
    for (auto i { 0 }; i < number_of_cars; ++i)
    {
      const auto process { fork() };

      if (process < 0)
      {
        std::perror("fork failed");
        std::exit(1);
      }

      if (process == 0)
      {
        if (i == 0) process_group = getpid();
        setpgid(0, process_group);

        cars->at(i).start(race.get());
        std::exit(0);
      }

      processes.at(i) = process;

      if (i == 0) process_group = process;
      setpgid(process, process_group);
    }
  }

  auto start()
  {
    for (auto stage { 1 }; stage <= number_of_stages; ++stage)
    {
      race->current_stage = stage;
      race->finish_order_counter = 0;

      for (auto& car : *cars)
        car = {
          .progress = 0,
          .order = 0,
          .points = car.points,
          .finished = false,
        };

      std::println("Preparing Stage {}", stage);
      sleep(1);

      kill(-process_group, SIGUSR1);

      auto finished_count { 0 };
      while (finished_count < cars->size())
      {
        display_progress();

        finished_count = 0;

        for (const auto& car : *cars)
          if (car.finished) ++finished_count;

        usleep(200000);
      }

      display_points();

      if (stage == number_of_stages)
      {
        std::println("Race finished");
        continue;
      }

      std::println("\nPress Enter to start the next stage...");
      std::cin.ignore();

      kill(-process_group, SIGUSR2);
    }

    display_results();

    for (const auto& process : processes) waitpid(process, nullptr, 0);
  }

 private:
  void display_progress() const
  {
    std::print("\033[2J\033[1;1H");
    std::print("Stage {} Progress:\n", race->current_stage);

    for (auto i { 0 }; const auto& car : *cars)
    {
      int progress = car.progress;
      int bar = (progress * track_length) / finish_line;

      std::string progress_bar {};
      progress_bar.reserve(track_length);

      for (int j = 0; j < track_length; ++j)
      {
        (j < bar)    ? progress_bar.push_back('=')
        : (j == bar) ? progress_bar.push_back('>')
                     : progress_bar.push_back('.');
      }

      std::print("Car {} : [{}] {} / {}", (++i), progress_bar, progress,
                 finish_line);

      if (car.finished)
        std::println(" (Finished, order: {})", car.order);
      else
        std::println();
    }
  }

  void display_points() const
  {
    for (auto& car : *cars) car.points += car.order;

    std::array<int, number_of_cars> order_indices {};
    for (int i = 0; i < number_of_cars; i++) order_indices[i] = i;

    std::ranges::sort(order_indices, [this](int a, int b) {
      return cars->at(a).order < cars->at(b).order;
    });

    std::print("\nStage {} Results:\n", race->current_stage);
    std::print("Rank\tCar\tFinish Order\tStage Points\tTotal Points\n");

    for (int i = 0; i < number_of_cars; i++)
    {
      const auto idx { order_indices[i] };
      const auto car { cars->at(idx) };

      int stage_points = cars->at(idx).order;

      std::print("{}\tCar {}\t{}\t\t{}\t\t{}\n", i + 1, idx + 1,
                 cars->at(idx).order, stage_points, cars->at(idx).points);
    }
  }

  void display_results() const
  {
    std::array<int, number_of_cars> order_indices {};
    for (int i = 0; i < number_of_cars; i++) order_indices[i] = i;

    std::ranges::sort(order_indices, [this](int a, int b) {
      return cars->at(a).points < cars->at(b).points;
    });

    std::print("\nFinal Race Results:\n");
    std::print("Rank\tCar\tTotal Points\n");

    for (int i = 0; i < number_of_cars; i++)
    {
      const auto idx { order_indices[i] };
      std::print("{}\tCar {}\t{}\n", i + 1, idx + 1, cars->at(idx).points);
    }
  }

 private:
  pid_t process_group {};
  std::array<pid_t, number_of_cars> processes {};

  MemoryMapping<Race> race {};
  MemoryMapping<std::array<Car, number_of_cars>> cars {};
};

// TODO: use modern random facilities

int main(int argc, char** argv)
{
  Arbiter arbiter {};
  arbiter.prepare();
  arbiter.start();
}