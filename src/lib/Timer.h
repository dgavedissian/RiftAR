#pragma once

#include <chrono>

class Timer
{
    typedef std::chrono::high_resolution_clock high_resolution_clock;
    typedef std::chrono::nanoseconds nanoseconds;

public:
    Timer(bool run = false);

    void reset();
    float elapsed() const;

    template <typename T, typename Traits>
    friend std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, const Timer& timer)
    {
        return out << timer.elapsed() << "ms";
    }

private:
    high_resolution_clock::time_point mStart;

};