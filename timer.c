#include <stdio.h>
#include <stddef.h>
#include <sys/time.h>                // for gettimeofday()
struct timeval t1, t2;

int timerOn()
{
    gettimeofday(&t1, NULL);
}

double timerOff()
{
    // stop timer
    gettimeofday(&t2, NULL);

    // compute and print the elapsed time in millisec
    double elapsedTime;
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    return elapsedTime;
}
