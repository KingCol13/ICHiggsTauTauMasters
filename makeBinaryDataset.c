/*
Compile:
g++ makeBinaryDataset.c -I /home/kingsley/anaconda3/envs/htt/include -L /home/kingsley/anaconda3/envs/htt/lib -lCore -lRIO -lTree -lTreePlayer -std=c++17 -o makeBinaryDataset


*/

#include <iostream>
#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

int main()
{
	auto myFile = TFile::Open("MVAFILE_AllHiggs_tt.root");
	if(!myFile || myFile->IsZombie())
	{
		std::cerr << "File didn't load correctly." << std::endl;
		return -1;
	}
	TTreeReader myReader("ntuple", myFile);
	TTreeReaderValue<double> my_pt_1(myReader, "pt_1");
	
	for(unsigned int i=0; i<10; i++)
	{
		myReader.Next();
		std::cout << *my_pt_1 << std::endl;
	}
	
	return 0;
}
