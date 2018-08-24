//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>              //For FILE, popen
#include <string.h>             //For strncpy
#include <unistd.h>             //For getcwd
#include "arts.h"
#include "artsConfig.h"         //For struct artsConfig
#include "artsGlobals.h"        //For artsGloablMessageTable
#include "artsRemoteLauncher.h" //For struct artsLauncher

#define DPRINTF(...)
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

void artsRemoteLauncherSSHStartupProcesses( struct artsRemoteLauncher * launcher )
{
    unsigned int argc = launcher->argc;
    char ** argv = launcher->argv;
    struct artsConfig * config = launcher->config;
    unsigned int killMode = launcher->killStuckProcesses;
        
    FILE ** sshExecutions = NULL;
    int i,j,k;
    int startNode = config->myRank;
    
    char cwd[1024];
    int cwdLength;
    getcwd(cwd, sizeof(cwd));
    cwdLength = strlen(cwd);
    
    sshExecutions = artsMalloc( sizeof( FILE *  ) * config->tableLength-1 );
    launcher->launcherMemory = sshExecutions;
    DPRINTF("%s\n", argv[0]);
    char command[4096];
    char directory[4096];
    DPRINTF("%d \n", config->tableLength);
    pid_t child;
    for(k=startNode+1; k< config->tableLength+startNode; k++ )
    {
        i=k%config->tableLength;
        unsigned int finalLength=0;
        unsigned int len = strlen(config->table[i].ipAddress);

        if(killMode)
        {
            strncpy(command+finalLength, "\"\"pkill ", 8);
            finalLength+=8;
            

            if(k==startNode+1)
            {
                len = strlen(argv[0]);
                char* lastSlash;
                for(j=0; j<len; j++)
                    if(argv[0][j]=='/')
                        lastSlash=argv[0]+j;

                *lastSlash = '\0';
            }
                
            len = strlen(argv[0]);
            int lastLen = len;
            len = strlen(argv[0]+len+1);
            len = (len>15) ? 15 : len;
            strncpy(command+finalLength, argv[0]+lastLen+1, len);
            finalLength+=len;
        }
        else
        {
            strncpy(command+finalLength, "\"\"cd ", 5);
            finalLength+=5;
            strncpy(command+finalLength, cwd, cwdLength);
            finalLength+=cwdLength;
            strncpy(command+finalLength, ";", 1);
            finalLength+=1;
            
            for(j=0; j<argc; j++)
            {
                *(command+finalLength++)=' ';
                len = strlen( argv[j] );
                strncpy(command+finalLength, argv[j], len);
                finalLength+=len;
            }
        }

        strncpy(command+finalLength, "\"\"\0", 3);
        finalLength+=3;

        child = fork();

        if(child == 0)
        {
            execlp("ssh", "-f", config->table[i].ipAddress, command, (char *)NULL); 
            //execlp("ssh", "-f", config->table[i].ipAddress, "cd runLocal; /home/land350/intel/test/intel/inspector_xe_2016.1.1.435552/bin64/inspxe-cl -c mi3 /home/land350/new/dtcp/test/artsFib 20", (char *)NULL); 
        }
    }
    if(killMode)
        exit(0);
}


void artsRemoteLauncherSSHCleanupProcesses( struct artsRemoteLauncher * launcher )
{
}

