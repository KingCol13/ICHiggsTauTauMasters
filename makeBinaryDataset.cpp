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
-apply selections
-output to binary

*/

#include <iostream>
#include <fstream>
#include <vector>

#include "TFile.h"
#include "TKey.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

int main(int argc, char *argv[])
{
	if(argc<2)
	{
		std::cerr << "Usage: makeBinaryDataset [ntupleKey1] [ntupleKey2] ..." << std::endl;
		return -1;
	}
	
	int tau_dm_1 = 1;
	int tau_dm_2 = 1;
	
	// Open file and make sure it isn't borked
	auto inFile = TFile::Open("MVAFILE_AllHiggs_tt.root");
	if(!inFile || inFile->IsZombie())
	{
		std::cerr << "File didn't load correctly." << std::endl;
		return -1;
	}
	
	/*
	TODO: check argv keys are in NTuple
	// Make sure all keys are in file
	TIter next(inFile->GetListOfKeys());
	TKey *key;
	while((key)=(TKey*)next())
	{
		std::cout << "Key: " << key->GetName() << " has objects of class: " << key->GetSeekKey() << std::endl;
	}
	*/
	
	// Make the reader object to iterate over
	TTreeReader reader("ntuple", inFile);
	
	// Make output value readers bound to the reader
	std::vector<TTreeReaderValue<double>> readerValueVec;
	for(unsigned int i=1; i<argc; i++)
	{
		readerValueVec.emplace_back(reader, argv[i]);
	}
	
	// Make selector value readers bound to the reader
	// Tau decay mode selectors:
	TTreeReaderValue<int> mva_dm_1(reader, "mva_dm_1");
	TTreeReaderValue<int> mva_dm_2(reader, "mva_dm_1");
	TTreeReaderValue<int> hps_dm_1(reader, "tau_decay_mode_1");
	TTreeReaderValue<int> hps_dm_2(reader, "tau_decay_mode_2");
	// Neutrino momenta selectors:
	TTreeReaderValue<double> gen_nu_p_1(reader, "gen_nu_p_1");
	TTreeReaderValue<double> gen_nu_p_2(reader, "gen_nu_p_2");
	
	// Binary file stream for output
	std::fstream outFile;
	outFile.open("recordData.dat", std::ios::binary);
	
	//for(unsigned int i=0; i<10; i++)
	while(reader.Next())
	{
		reader.Next();
		// Selection conditions
		if( 
			(*mva_dm_1 == tau_dm_1) &&
			(*mva_dm_2 == tau_dm_2) &&
			(*hps_dm_1 == tau_dm_1) &&
			(*hps_dm_2 == tau_dm_2) &&
			(*gen_nu_p_1 > -9000)   &&
			(*gen_nu_p_2 > -9000)
		)
		{
			// loop through values to output
			for(unsigned int j=0; j<argc-1; j++)
			{
				float val = *readerValueVec[j];
				outFile.write( (char *) &val, sizeof(float) );
			}
		}
	}
	
	std::cout << "Bytes per entry: " << sizeof(float)*readerValueVec.size() << std::endl;
	outFile.close();
	inFile->Close();
	
	return 0;
}
