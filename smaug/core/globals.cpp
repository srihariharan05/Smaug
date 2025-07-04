#include "smaug/core/globals.h"

namespace smaug {
bool runningInSimulation;
bool fastForwardMode = true;
int numAcceleratorsAvailable;
ThreadPool* threadPool = nullptr;
bool useSystolicArrayWhenAvailable;
long int stat_num_of_tiles =0;

}  // namespace smaug
