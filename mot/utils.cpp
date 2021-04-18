#include <chrono>
#include "utils.h"

#ifdef WIN32
#include <io.h>
#endif


std::string get_timestamp()
{
	const auto now = std::chrono::system_clock::now();
	auto sec_utc = std::chrono::system_clock::to_time_t(now);
	struct tm* ptm = localtime(&sec_utc);
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

	char tmp[30] = { 0 };
	sprintf(tmp, "%4d-%02d-%02d %02d:%02d:%02d:%03d",
		(int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
		(int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec, ms);
	std::string filename = tmp;
	return filename;
}

void read_filelist(const std::string& listfilename, std::vector<std::string>& vec_list_filename)
{
	std::ifstream listfile;
	int filenum = 0;
	listfile.open(listfilename.c_str());
	listfile >> filenum; listfile.ignore(65536, '\n');
	std::string stringbuf;
	for (int i = 0; i < filenum; i++)
	{
	    listfile >> stringbuf;
		vec_list_filename.push_back(stringbuf);
		stringbuf.clear();
	}
	listfile.close();
}
