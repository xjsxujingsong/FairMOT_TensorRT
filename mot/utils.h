#ifndef _UTILS_H_
#define _UTILS_H_
#ifdef WIN32
#include<ctime>
#include <io.h>
#else
#include <stdint.h>
#include <cstring>
#include <climits>
#include <float.h>
#include <math.h>
#endif
#if defined WIN32 || defined _WIN32
#include <winsock2.h>
#include <time.h>
#else
#include <sys/time.h>
#endif
#include <time.h>

#include <stdio.h>
#if defined WIN32 || defined _WIN32
#include <conio.h>        // For _kbhit() on Windows
#include <direct.h>        // For mkdir(path) on Windows
#define snprintf sprintf_s    // Visual Studio on Windows comes with sprintf_s() instead of snprintf()
#else
#include <stdio.h>        // For getchar() on Linux
#include <termios.h>    // For kbhit() on Linux
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>    // For mkdir(path, options) on Linux
#endif
#include <vector>
#include <string>
#include <iostream>            // for printing streams in C++
#include <map>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


#include "Config.h"
using namespace std;

// These functions will print using the LOG() function, using the same format as printf(). If you want it to be printed using a different
// function (such as for Android logcat output), then define LOG as your output function, otherwise it will use printf() by default.
#ifndef LOG
#ifdef __ANDROID__
// For Android debug logging to logcat:
#include <android/log.h>
#define LOG(fmt, args...) (__android_log_print(ANDROID_LOG_INFO, "........", fmt, ## args))
#else
// For stdout debug logging, with a new-line character on the end:
#ifndef _MSC_VER
// Compiles on GCC but not MSVC:
#define LOG(fmt, args...) do {printf(fmt, ## args); printf("\n"); fflush(stdout);} while (0)
//        #define LOG printf
#else
#define LOG printf
#endif
#endif
#endif

// Allow default args in C++ code, but explicit args in C code.
#ifdef __cplusplus
#define DEFAULT(val) = val
#else
#define DEFAULT(val)
#endif

//#define __ANDROID__
#ifdef __ANDROID__
#include "net.h"
#include "benchmark.h"
#endif

//------------------------------------------------------------------------------
// Timer functions
//------------------------------------------------------------------------------
#define  _USE_OPENCV_
#ifndef _USE_OPENCV_

#ifdef _MSC_VER
# if _MSC_VER >= 1700
#  pragma warning(disable:4447) // Disable warning 'main' signature found without threading model
# endif
#endif

#if defined WIN32 || defined _WIN32 || defined WINCE
#ifndef _WIN32_WINNT           // This is needed for the declaration of TryEnterCriticalSection in winbase.h with Visual Studio 2005 (and older?)
#define _WIN32_WINNT 0x0400  // http://msdn.microsoft.com/en-us/library/ms686857(VS.85).aspx
#endif
#include <windows.h>
#if (_WIN32_WINNT >= 0x0602)
#include <synchapi.h>
#endif

#endif

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

int64 getTickCount(void);
double getTickFrequency(void);

int64  cvGetTickCount(void);
double cvGetTickFrequency(void);


int  cvRound(double value);

#endif
// Record the execution time of some code, in milliseconds. By Shervin Emami, 27th May 2012.
// eg:
//  DECLARE_TIMING(myTimer);
//    ...
//  START_TIMING(myTimer);
//    printf("A slow calc = %f\n", pow(1.2, 3.4) );
//  STOP_TIMING(myTimer);
//  AVERAGE_TIMING(myTimer);
#define DECLARE_TIMING(s)           int64 timeStart_##s; int64 timeDiff_##s; int64 timeTally_##s = 0; int64 countTally_##s = 0; double timeMin_##s = DBL_MAX; double timeMax_##s = 0; int64 timeEnd_##s;
#define START_TIMING(s)             timeStart_##s = cv::getTickCount()
#define STOP_TIMING(s)              do {    timeEnd_##s = cv::getTickCount(); timeDiff_##s = (timeEnd_##s - timeStart_##s); timeTally_##s += timeDiff_##s; countTally_##s++; timeMin_##s = MIN(timeMin_##s, timeDiff_##s); timeMax_##s = MAX(timeMax_##s, timeDiff_##s);    } while (0)
#define GET_TIMING(s)               (double)(1000  * ( (double)timeDiff_##s / (double)cv::getTickFrequency() ))
#define GET_MIN_TIMING(s)           (double)(1000 * ( (double)timeMin_##s / (double)cv::getTickFrequency() ))
#define GET_MAX_TIMING(s)           (double)(1000 * ( (double)timeMax_##s / (double)cv::getTickFrequency() ))
#define GET_AVERAGE_TIMING(s)       (double)(countTally_##s ? 1000 * ( (double)timeTally_##s / ((double)countTally_##s * cv::getTickFrequency()) ) : 0)
#define GET_TOTAL_TIMING(s)         (double)(1000 * ( (double)timeTally_##s / ((double)cv::getTickFrequency()) ))
#define GET_TIMING_COUNT(s)         (int)(countTally_##s)
#define CLEAR_AVERAGE_TIMING(s)     do {    timeTally_##s = 0; countTally_##s = 0;     } while (0)
#define SHOW_TIMING(s, msg)         LOG("%s time:\t %dms\t (ave=%dms min=%dms max=%dms, across %d runs). FPS:%.2f\n", msg, cvRound(GET_TIMING(s)), cvRound(GET_AVERAGE_TIMING(s)), cvRound(GET_MIN_TIMING(s)), cvRound(GET_MAX_TIMING(s)), GET_TIMING_COUNT(s),  1000.0/cvRound(GET_AVERAGE_TIMING(s)))
#define SHOW_TIMING2(s, msg, times)         LOG("%s time:\t %dms\t (ave=%dms min=%dms max=%dms, across %d runs). FPS:%.2f\n", msg, cvRound(GET_TIMING(s)), cvRound(GET_AVERAGE_TIMING(s)), cvRound(GET_MIN_TIMING(s)), cvRound(GET_MAX_TIMING(s)), GET_TIMING_COUNT(s),  times*1000.0/cvRound(GET_AVERAGE_TIMING(s)))
#define SHOW_TOTAL_TIMING(s, msg)   LOG("%s total:\t %dms\t (ave=%dms min=%dms max=%dms, across %d runs).", msg, cvRound(GET_TOTAL_TIMING(s)), cvRound(GET_AVERAGE_TIMING(s)), cvRound(GET_MIN_TIMING(s)), cvRound(GET_MAX_TIMING(s)), GET_TIMING_COUNT(s) )
#define AVERAGE_TIMING(s)           SHOW_TIMING(s, #s)
#define TOTAL_TIMING(s)             do {    SHOW_TOTAL_TIMING(s, #s); CLEAR_AVERAGE_TIMING(s);     } while (0)


// Convert a float number to an int by rounding to nearest int using a certain method.
// Replace the code with the fastest method detected by 'testTiming_FloatConversion()'.
inline int roundFloat(float f);

//http://stackoverflow.com/questions/8065413/stdlexical-cast-is-there-such-a-thing
template <typename T>
T lexical_cast(const std::string& str)
{
	T var;
	std::istringstream iss;
	iss.str(str);
	iss >> var;
	// deal with any error bits that may have been set on the stream
	return var;
}
typedef std::map<std::string, std::string> configuration_map;

template <typename T> T get(configuration_map const& config, std::string const& name, T default_value = T())
{
	configuration_map::const_iterator it = config.find(name);
	if (it == config.end())
		return default_value;
	return lexical_cast<T>(it->second);
}
template <typename T, typename U> void set_from(T& target, configuration_map const& config, std::string const& name, U default_value)
{
	target = get<T>(config, name, default_value);
}
template <typename T> void set_from(T& target, configuration_map const& config, std::string const& name)
{
	target = get<T>(config, name);
}

template <typename T>
static void clear_2d_vector(T &t)
{
	for (int i = 0, size = t.size(); i < size; i++) t[i].clear();
	t.clear();
}

template <typename T>
static void clear_3d_vector(T &t)
{
	for (int i = 0, size = t.size(); i < size; i++)
	{
		for (int j = 0, size2 = t[i].size(); j < size2; j++) t[i][j].clear();
		t[i].clear();
	}
	t.clear();
}

template<class TI>
float log2(TI n)
{
	// log(n)/log(2) is log2.  
	return log(n) / log(2.0f);
}
template<class TI>
TI round(TI r)
{
	return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}

void read_filelist(const std::string& listfilename, std::vector<std::string>& vec_list_filename);


#endif
