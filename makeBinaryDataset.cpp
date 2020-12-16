/*
Compile:
g++ makeBinaryDataset.cpp -I /home/kingsley/anaconda3/envs/htt/include -L /home/kingsley/anaconda3/envs/htt/lib -lCore -lRIO -lTree -lTreePlayer -std=c++17 -o makeBinaryDataset

modify path after -I and -L for root Include and lib directories accordingly
linker libs:
Core
RIO
Tree
TreePlayer

TODO: 
-check keys are in file: https://root.cern/manual/storing_root_objects/#finding-tkey-objects
-split argv into int variables and float variables
-output to binary

*/

#include <iostream>
#include <fstream>
#include <vector>

#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

int main(int argc, char *argv[])
{
	if(argc<2)
	{
		std::cerr << "Usage: makeBinaryDataset [ntupleKey1] [ntupleKey2] ..." << std::endl;
		return -1;
	}	
	
	// Open file and make sure it isn't borked
	auto myFile = TFile::Open("MVAFILE_AllHiggs_tt.root");
	if(!myFile || myFile->IsZombie())
	{
		std::cerr << "File didn't load correctly." << std::endl;
		return -1;
	}
	// Make the reader object to iterate over
	TTreeReader reader("ntuple", myFile);
	
	// Make sure 
	
	std::vector<TTreeReaderValue<double>> readerValueVec;
	for(unsigned int i=1; i<argc; i++)
	{
		readerValueVec.emplace_back(reader, argv[i]);
	}
	
	for(unsigned int i=0; i<10; i++)
	{
		reader.Next();
		// loop through values to output
		for(unsigned int j=0; j<argc-1; j++)
		{
			 std::cout << *readerValueVec[j] << ", ";
		}
		std::cout << std::endl;
	}
	
	return 0;
}
