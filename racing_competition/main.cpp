#include <sys/msg.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <print>
#include <random>
#include <ranges>

namespace
{
constexpr auto number_of_stages { 3 };
constexpr auto cars_number { 5 };
constexpr auto finish_line { 100 };
constexpr auto track_length { 50 };

std::atomic_bool start_flag {};
std::atomic_bool next_flag {};
}  // namespace

struct ProgressMessage
{
  long mtype { 1 };
  int id {};
  int progress {};
  int finished {};
};

class Car
{
 public:
  void start(int id, int queue)
  {
    signal(SIGUSR1, [](int) { start_flag = true; });
    signal(SIGUSR2, [](int) { next_flag = true; });

    std::mt19937 generator(std::random_device {}());
    std::uniform_int_distribution<> step_dist(1, 10);
    std::uniform_int_distribution<> sleep_dist(100, 300);

    for (int stage { 1 }; stage <= number_of_stages; ++stage)
    {
      while (!start_flag) pause();
      start_flag = false;

      progress = 0;
      finished = false;
      order = 0;

      while (progress < finish_line)
      {
        progress = std::min(progress + step_dist(generator), finish_line);

        ProgressMessage message {
          .id = id,
          .progress = progress,
        };
        msgsnd(queue, &message, sizeof(message) - sizeof(long), 0);

        usleep(sleep_dist(generator) * 1000);
      }

      ProgressMessage message {
        .id = id,
        .progress = progress,
        .finished = 1,
      };
      msgsnd(queue, &message, sizeof(message) - sizeof(long), 0);

      if (stage == number_of_stages) continue;

      while (!next_flag) pause();
      next_flag = false;
    }
  }

  int progress {};
  int order {};
  int points {};
  bool finished {};
};

class Arbiter
{
 public:
  ~Arbiter() { msgctl(progress_queue, IPC_RMID, nullptr); }

  void prepare()
  {
    for (auto i { 0 }; i < cars_number; ++i)
    {
      auto const process { fork() };

      if (process == 0)
      {
        if (i == 0) process_group = getpid();
        setpgid(0, process_group);

        cars.at(i).start(i, progress_queue);
        std::exit(0);
      }

      processes.at(i) = process;

      if (i == 0) process_group = process;
      setpgid(process, process_group);
    }
  }

  void start()
  {
    for (int stage { 1 }; stage <= number_of_stages; ++stage)
    {
      current_stage = stage;
      finish_order_counter = 0;

      for (auto& car : cars)
        car = {
          .progress = 0,
          .order = 0,
          .points = car.points,
          .finished = false,
        };

      std::println("Подготовка этапа {}", stage);
      sleep(1);

      kill(-process_group, SIGUSR1);

      unsigned int finished_count {};
      while (finished_count < cars.size())
      {
        ProgressMessage message {};

        while (msgrcv(progress_queue, &message, sizeof(message) - sizeof(long),
                      0, IPC_NOWAIT) > 0)
        {
          auto& car { cars.at(message.id) };

          car.progress = message.progress;

          if (message.finished && !car.finished)
          {
            car.finished = true;
            car.order = ++finish_order_counter;
            car.points += car.order;
          }
        }

        finished_count = std::ranges::count_if(cars, &Car::finished);

        display_progress();

        usleep(200000);
      }

      display_points();

      if (stage == number_of_stages)
      {
        std::println("Гонка завершена");
        continue;
      }

      std::println("\nНажмите Enter для начала следующего этапа...");
      std::cin.ignore();

      kill(-process_group, SIGUSR2);
    }

    display_results();

    for (auto const& process : processes) waitpid(process, nullptr, 0);
  }

 private:
  void display_progress() const
  {
    std::print("\033[2J\033[1;1H");
    std::println("Прогресс этапа {}:", current_stage);

    for (int i {}; auto const& car : cars)
    {
      std::string bar(track_length, '.');

      int pos { (car.progress * track_length) / finish_line };

      std::fill_n(bar.begin(), std::min(pos, track_length), '=');

      if (pos < track_length) bar.at(pos) = '>';

      std::print("Машина {} : [{}] {} / {}", (++i), bar, car.progress,
                 finish_line);

      if (car.finished)
        std::println(" (Финишировала с местом: {})", car.order);
      else
        std::println();
    }
  }

  void display_points() const
  {
    std::println("\nРезультаты этапа:", current_stage);

    std::array<Car const*, cars_number> orders {};

    for (auto [ordered_car, car] : std::ranges::zip_view { orders, cars })
      ordered_car = &car;

    std::ranges::sort(orders, std::ranges::less {}, &Car::order);

    for (int i {}; auto order : orders)
      std::println("Место {}: Машина {} (Очки: {})", ++i,
                   std::distance(cars.begin(), order) + 1, order->points);
  }

  void display_results() const
  {
    std::println("\nИтоговые результаты:");

    std::array<Car const*, cars_number> scores {};

    for (auto [score, car] : std::ranges::zip_view { scores, cars })
      score = &car;

    std::ranges::sort(scores, std::ranges::less {}, &Car::points);

    for (int i {}; auto car : scores)
      std::println("Место {}: Машина {} (Всего очков: {})", ++i,
                   std::distance(cars.data(), car) + 1, car->points);
  }

  std::array<pid_t, cars_number> processes {};
  pid_t process_group {};

  int progress_queue { msgget(IPC_PRIVATE, IPC_CREAT | 0666) };

  int current_stage {};
  int finish_order_counter {};
  std::array<Car, cars_number> cars {};
};

int main()
{
  Arbiter arbiter {};
  arbiter.prepare();
  arbiter.start();
}