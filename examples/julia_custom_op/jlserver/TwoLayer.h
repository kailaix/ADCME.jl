#include "io_julia.h"
#include <unistd.h>
#include <string>
#include <iostream>
#include <fstream>  



bool FileExists( const std::string &Filename )
{
    return access( Filename.c_str(), 0 ) == 0;
}

void forward(double *g, double *y, const double *x, int n){
  static int jobid = 1;
  jobid += 1;
  char filename[1024];char  ofilename[1024];char  ofilename2[1024];
  char signal[1024];char  osignal[1024];
  sprintf(filename, "input.txt");
  sprintf(signal, "isignal.txt");
  sprintf(ofilename, "output.txt");
  sprintf(ofilename2, "output2.txt");
  sprintf(osignal, "osignal.txt");

  saveArray<double>(x, n, string(filename));
  std::ofstream outfile( (string(signal)) ); outfile.close();


  while(true){
    if (FileExists( (string(ofilename)) )&& FileExists( (string(osignal)))){
      loadArray<double>(y, n, (string(ofilename)));
      loadArray<double>(g, n*n, (string(ofilename2)));
      remove(ofilename); remove(ofilename2); remove(osignal);
      break;
    }
  } 
}