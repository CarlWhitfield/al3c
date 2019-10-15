#define LINE_LEN    1024
#define __STDC_LIMIT_MACROS

#include <iostream>
#include <sstream>
#include <vector>
#include <string.h>
//#include <unistd.h>
#include <process.h>
#include <mpi.h>
#include <signal.h>
#include <cmath> 
#include "rapidxml/rapidxml.hpp"
#include "../include/al3c.hpp"
#include "../include/externs_typedefs.hpp"
#include "../include/mpi_check.hpp"
#include "../include/signal.hpp"
#include "../include/SMC.hpp"
#include "../include/u01.hpp"
#include <boost/multiprecision/random.hpp>

using namespace std;

boost::random::mt19937_64 rng;

struct param_t 
{
    float a;
};

struct param_summary_t 
{
    float a_var;
};

void user_t::prior() 
{
    param->a = u01()*20 - 10; // Unif[-10,10]
};

float user_t::prior_density() 
{
    if (-10<=param->a && param->a<=10)
        return 1;
    else
        return 0;

    return 1;
}

void user_t::perturb() 
{
    param->a+=(u01()-0.5f)*sqrt(2*param_summary->a_var*12);
}

float user_t::perturb_density(param_t *old_param) 
{
	if ( fabs(param->a - old_param->a) > sqrt(2*param_summary->a_var*12)/2.f ) 
		return 0.f;

    return 1.f;
}

void user_t::simulate() 
{
	//toy example
	boost::random::normal_distribution<float> nd1(param->a, 1);
	boost::random::normal_distribution<float> nd2(param->a, 0.1);
	float test = u01();
	if(test >= 0.5)
		S[0][0] = nd1(rng);
	else
		S[0][0] = nd2(rng);
}

float user_t::distance() 
{
	float r=0;

	for (uint n=0;n<N;n++)
	{
		for (uint d=0;d<D;d++)
		{
			r+=(S[n][d]-O[n][d])*(S[n][d]-O[n][d]);
		}
	}

	return sqrt(r);
}

void user_summary_t::summarize()
{
    float m1=0,m2=0;

    for (uint a=0;a<A;a++) {
        m1+=params[a]->a;
        m2+=params[a]->a*params[a]->a;
    } m1/=(float)A; m2/=(float)A;
    summary->a_var=m2-m1*m1;
}

void user_t::print(ofstream& output, bool header) 
{
	if(header)
		output << "Theta, distance\n";

	output << param->a << ", " << distance() << '\n';
}

uint np=0, NP=0, SIGNUM=0;

int main (int argc, char *argv[] ) {

    //configuration file...
    if (argc!=2) {
        std::cerr<<"Error! Run with "<<argv[0]<<" <XML configuration>"\
            <<std::endl;
        exit(EXIT_FAILURE);
    }

    std::ifstream xmlfile (argv[1]);
    if (!xmlfile) {
        std::cerr<<"Error! Could not open XML file '"<<argv[1]<<"'"<<std::endl;
        exit(EXIT_FAILURE);
    }

    //register signal handler
    signal(SIGINT, signal_callback_handler);

    //get MPI running...
    MPI_Init(NULL,NULL);

	int nph, NPh;
    MPI_Comm_rank(MPI_COMM_WORLD, &nph);
	MPI_Comm_size(MPI_COMM_WORLD, &NPh);
	np = nph;
	NP = NPh;

    std::vector<char> buffer((std::istreambuf_iterator<char>(xmlfile)), \
            std::istreambuf_iterator<char>());
    buffer.push_back('\0');
    rapidxml::xml_document<> config;    // character type defaults to char
    config.parse<0>(&buffer[0]);    // 0 means default parse flags


    if (!config.first_node("MPI")->first_node("NP")) {
        std::cerr<<"Error! Could not find required <MPI><NP></NP></MPI> in"\
            " XML file"<<std::endl;
        exit(EXIT_FAILURE);
    }

    ////if not already running in MPI, this will invoke it for us...
	//uncomment when using MPI
    if (atoi(config.first_node("MPI")->first_node("NP")->value())!=(int)NP && \
            NP==1) {

        int returncode;

        char **args=new char*[4+argc];
        args[0]=_strdup("mpiexec");
        args[1]=_strdup("-n");
        args[2]=_strdup(config.first_node("MPI")->first_node("NP")->value());

        for (int i=0;i<argc;i++) {
            args[3+i]=argv[i];
        } args[3+argc]=NULL;

        //if (check_mpirun())
        returncode=_execvp("mpiexec",args);
       // else
         //   returncode=_execvp("bin/mpirun",args);

        if (returncode!=0)
            std::cerr<<"Warning: mpirun exited with code '"<<returncode<<\
                "'"<<std::endl;

        for (int i=0;i<3;i++)
            free(args[i]);
        delete [] args;

        exit(returncode);
    }

    print_cpu_info();
	u01();
    //initialize our ABC routine
    SMC_t SMC(&config);

    //begin the loop
    SMC.loop();

    delete [] rnd_array;
    // gracefully quit
    MPI_Finalize();

    exit(EXIT_SUCCESS);

}
