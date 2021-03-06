/*
Compile:
g++ makeBinaryDataset.cpp -I /home/kingsley/anaconda3/envs/htt/include -L /home/kingsley/anaconda3/envs/htt/lib -lCore -lRIO -lTree -lTreePlayer -std=c++17 -o makeBinaryDataset

modify path after -I and -L for root Include and lib directories accordingly
linker libs:
Core
RIO
Tree
TreePlayer

Note metcov variables are treated specially (they are leaf lists) and so will be the first variables of each entry in the binary dataset

TODO: 
-check keys are in file: https://root.cern/manual/storing_root_objects/#finding-tkey-objects
-get tau decay mode ints from command line

*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>

#include "TFile.h"
#include "TKey.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

int main(int argc, char *argv[])
{
	if(argc<4)
	{
		std::cerr << "Usage: makeBinaryDataset [tau_decay_mode_1] [tau_decay_mode_2] [ntupleKey1] [ntupleKey2] ..." << std::endl;
		return -1;
	}
	
	int tau_dm_1 = std::stoi(argv[1]);
	int tau_dm_2 = std::stoi(argv[2]);
	
	std::cout << "Using tau_dm_1 = " << tau_dm_1 << std::endl;
	std::cout << "Using tau_dm_2 = " << tau_dm_2 << std::endl;
	
	// Open file and make sure it isn't borked
	auto inFile = TFile::Open("MVAFILE_AllHiggs_tt.root");
	if(!inFile || inFile->IsZombie())
	{
		std::cerr << "File didn't load correctly." << std::endl;
		return -1;
	}
	
	// Make the reader object to iterate over
	TTreeReader reader("ntuple", inFile);

	// Make output value readers bound to the reader
	std::vector<TTreeReaderValue<double>> readerValueVec;
	std::vector<TTreeReaderValue<float>> metValueVec;
	
	TTree *myTree = (TTree *) inFile->Get("ntuple");
	// Check key is valid then bind a reader
	for(int i=3; i<argc; i++)
	{
		bool metVal = false;
		// Tolerate metCov instead of metcov
		if( (strncmp(argv[i], "metcov", 6)==0) || (strncmp(argv[i], "metCov", 6)==0) )
		{
			argv[i][3] = 'C';
			metVal = true;
		}
		
		// Check the key is valid
		if(myTree->GetLeaf(argv[i])==nullptr)
		{
			std::cerr << "Key \"" << argv[i] << "\" is not a valid leaf, exiting." << std::endl;
			return -1;
		}
		
		// If it was a met variable then deal with leaflist nonsense
		if( metVal )
		{
			std::string metName = "metcov";
			metName+=argv[i][6];
			metName+=argv[i][7];
			metName+=".metCov";
			metName+=argv[i][6];
			metName+=argv[i][7];
			std::cout << "Binding: " << metName << std::endl;
			metValueVec.emplace_back(reader, metName.c_str());
		}
		else // Other variables are simple
		{
			readerValueVec.emplace_back(reader, argv[i]);
		}
	}
	
	const int numNormalVariables = readerValueVec.size();
	const int numMetVariables = metValueVec.size();
	
	// Make selector value readers bound to the reader
	// Tau decay mode selectors:
	TTreeReaderValue<int> mva_dm_1(reader, "mva_dm_1");
	TTreeReaderValue<int> mva_dm_2(reader, "mva_dm_2");
	TTreeReaderValue<int> hps_dm_1(reader, "tau_decay_mode_1");
	TTreeReaderValue<int> hps_dm_2(reader, "tau_decay_mode_2");
	// Neutrino momenta selectors:
	TTreeReaderValue<double> gen_nu_p_1(reader, "gen_nu_p_1");
	TTreeReaderValue<double> gen_nu_p_2(reader, "gen_nu_p_2");
	
	// Binary file stream for output
	std::ofstream outFile;
	outFile.open("recordData.dat", std::ios::binary);
	
	// Counter for how many entries meet selection criteria
	int numEntriesSelected = 0;
	int numEntriesSeen = 0;
	
	//for(unsigned int i=0; i<10; i++)
	std::cout << "Beginning write." << std::endl;
	while(reader.Next())
	{
		// Selection conditions
		if( 
			(*mva_dm_1 == tau_dm_1) &&
			(*mva_dm_2 == tau_dm_2) &&
			(*hps_dm_1 == tau_dm_1) &&
			(*hps_dm_2 == tau_dm_2) &&
			(*gen_nu_p_1 > -4000)   &&
			(*gen_nu_p_2 > -4000)
		)
		{
			// Increment entry counter
			numEntriesSelected++;
			
			// loop through met variables to output
			for(int j=0; j<numMetVariables; j++)
			{
				float val = *metValueVec[j];
				if(std::isnan(val))
				{
					std::cout << "Setting NaN at ntuple entry: " << numEntriesSeen << ", key: " << argv[j+3] << " to 0." << std::endl;
					val = 0;
				}
				outFile.write( (char *) &val, sizeof(float) );
			}
			
			// loop through normal variables to output
			for(int j=0; j<numNormalVariables; j++)
			{
				float val = *readerValueVec[j];
				if(std::isnan(val))
				{
					std::cout << "Setting NaN at ntuple entry: " << numEntriesSeen << ", key: " << argv[j+3] << " to 0." << std::endl;
					val = 0;
				}
				outFile.write( (char *) &val, sizeof(float) );
			}
		}
		numEntriesSeen++;
	}
	
	std::cout << "Number of entries selected: " << numEntriesSelected << std::endl;
	std::cout << "Total entries seen: " << numEntriesSeen << std::endl;
	std::cout << "Bytes per entry: " << sizeof(float)*(numMetVariables+numNormalVariables) << std::endl;
	outFile.close();
	inFile->Close();
	
	return 0;
}
