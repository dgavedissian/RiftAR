#include "Common.h"
#include "Timer.h"

Timer::Timer(bool run)
{
    if (run)
        reset();
}

void Timer::reset()
{
    mStart = high_resolution_clock::now();
}

float Timer::elapsed() const
{
    Timer::nanoseconds duration = std::chrono::duration_cast<Timer::nanoseconds>(
        Timer::high_resolution_clock::now() - mStart);
    return (float)duration.count() * 1e-6f;
}
