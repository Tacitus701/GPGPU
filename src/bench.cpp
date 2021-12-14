#include "cpu.hpp"
#include <vector>
#include <benchmark/benchmark.h>

void BM_Rendering_cpu(benchmark::State& st)
{
  for (auto _ : st)
    detect_barcode("../collective_database/PXL_1.png");

  st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}


BENCHMARK(BM_Rendering_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_MAIN();
