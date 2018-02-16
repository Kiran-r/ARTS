#include "hive.h"
#include "hiveMalloc.h"
#include "hiveConfig.h"
#include "hiveGlobals.h"
#include "hiveRemoteLauncher.h"
#include "string.h"
#include "stdlib.h"
#include "unistd.h"
#include <ctype.h>

#define DPRINTF(...)
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

struct hiveConfigVariable * hiveConfigFindVariable( struct hiveConfigVariable ** head, char * string  )
{
    struct hiveConfigVariable * found = NULL;
    struct hiveConfigVariable * last = NULL;
    struct hiveConfigVariable * next = *head;
    
    while(next != NULL)
    {
        if(strcmp(string, next->variable) == 0)
        {
            found = next;
            break;
        }
        last = next;
        next = next->next;
    }
    
    char * overide = getenv(string);

    //if(found)
    {
        if(overide)
        {
            unsigned int size = strlen(overide);
            struct hiveConfigVariable * newVar = hiveMalloc(sizeof(struct hiveConfigVariable) + size);
            
            newVar->size = size;
            strcpy(newVar->variable, string);
            strcpy(newVar->value, overide);
            
            if(last)
                last->next = newVar;
            else
            {
                newVar->next = *head;
                *head = newVar;
            }

            if(next)
            {
                newVar->next = next->next;
                hiveFree(next);
            }
            return newVar;
        }
        
    }
    
    return found;
}
char * hiveConfigFindVariableChar( struct hiveConfigVariable * head, char * string  )
{
    struct hiveConfigVariable * found = NULL;
    char * overide = getenv(string);
    
    if(overide)
        return overide;

    while(head != NULL)
    {
        if(strcmp(string, head->variable) == 0)
        {
            found = head;
            break;
        }
        head = head->next;
    }
    
    if(found)
        return found->value;

    return NULL;
}

unsigned int hiveConfigGetVariable( FILE * config, char * lookForMe  )
{
    char * line;
    size_t len=0;
    ssize_t read;
    char * var;
    char * val;
    int size;
    struct hiveConfigVariable * cVar;
    struct hiveConfigVariable * head, * next=NULL;

    while ((read = getline(&line, &len, config)) != -1)
    {
        //printf("Retrieved line of length %zu :\n", read);
        //printf("%s", line);

        var = strtok(line, "=");
        val = strtok(NULL, "=");

        if( strcmp(lookForMe, var) == 0 )
        {
            if(val==NULL)
                return 4;
            size= strlen(val);

            if(val[size-1]=='\n')
                val[size-1]='\0';
            
            
            return strtol(val, NULL, 10); 

            //cVar = hiveMalloc( sizeof( struct hiveConfigVariable ) + size );


            //cVar->size = size;

            //strncpy(cVar->variable, var, 255);
            //strcpy(cVar->value, val);
            //cVar->next=NULL;

            //if(next!=NULL)
            //    next->next= cVar;
            //else
            //    head=cVar;
            //next=cVar;
        }
    }

    return 4;
}

void removeWhiteSpaces(char * str)
{
    char *write = str, *read = str;
    do {
           if (*read != ' ')
                      *write++ = *read;
    } while (*read++);
}

struct hiveConfigVariable * hiveConfigGetVariables( FILE * config )
{
    char * line = NULL;
    size_t len=0;
    ssize_t read;
    char * var;
    char * val;
    int size;
    struct hiveConfigVariable * cVar;
    struct hiveConfigVariable * head, * next=NULL;

    while ((read = getline(&line, &len, config)) != -1)
    {
        //printf("Retrieved line of length %zu :\n", read);
        //printf("%s", line);

        var = strtok(line, "=");
        val = strtok(NULL, "=");

        if( val != NULL )
        {
            size= strlen(val);

            if(val[size-1]=='\n')
                val[size-1]='\0';

            cVar = hiveMalloc( sizeof( struct hiveConfigVariable ) + size );


            cVar->size = size;

            strncpy(cVar->variable, var, 255);
            strcpy(cVar->value, val);
            
            removeWhiteSpaces(cVar->variable);
            removeWhiteSpaces(cVar->value);

            cVar->next=NULL;

            if(next!=NULL)
                next->next= cVar;
            else
                head=cVar;
            next=cVar;
        }
    }

    return head;
}


char * hiveConfigMakeNewVar( char * var  )
{
    char * newVar;
    unsigned int size;
    size = strlen(var);
    newVar = hiveMalloc( size+1 );
    strncpy(newVar, var, size);
    newVar[size] = '\0';
    DPRINTF("%s l\n", newVar);
    return var;
}

unsigned int hiveConfigGetValue(char * start, char * stop)
{
    int i, value, size = stop - start;

    for(i=0; i<size; i++)
    {
        if(isdigit(start[i]))
        {
            if(*stop==':')
            {
                *stop='\0';

                value = strtol(start+i,NULL, 10);
                *stop = ':';
            }
            else
                value = strtol(start+i,NULL, 10);
            break;
        }
    }

    return value;
}

char * hiveConfigGetNodeName(char * start, char * stop)
{
    int i, value, size = stop - start;

    char * name;

    for(i=0; i<size; i++)
    {
        if(isdigit(start[i]))
        {

            name = hiveMalloc( (stop-start) );

            strncpy(name, start, stop-start);

            //name[i] = '\0';

            break;
        }
    }

    return name;
}
char * hiveConfigGetHostname( char * name, unsigned int value )
{
    unsigned int length = strlen(name);
    unsigned int i, j;
    unsigned int digits=1;
    unsigned int temp = value;
    unsigned int stop;
    char * outName = hiveMalloc(length);


    while(temp>9)
    {
        temp /= 10;

        digits++;
    }


    temp = value;

    for(i=0; i<length; i++)
    {
        if(isdigit(name[i]))
        {
            stop = i;
            while( stop < length  )
            {
                if(!isdigit(name[stop]))
                {
                    break;
                }
                stop++;
            }

            for(j=stop-1; j>(stop-1) - digits; j--)
            {
                //name[j]= itoa( value%10 );
                //sprintf(name+j,"%d",value%10);
                name[j] = ((int)'0')+value%10;
                value /=10;
            }

            for(j=(stop-1) - digits; j>=i; j--)
            {
                name[j]='0';
            }


            break;
        }
    }

    strncpy(outName, name, length);

    return outName;
}

char * hiveConfigGetSlurmHostname( char * name, char * digitSample, unsigned int value, bool ib, char * prefix, char * suffix )
{
    unsigned int length = strlen(name);
    unsigned int digitLength = strlen(digitSample);
    unsigned int i, j;
    unsigned int suffixLength = 0;
    unsigned int prefixLength = 0;
    unsigned int nameLength;
    
    if(suffix!= NULL)
        suffixLength = strlen(suffix);
    
    if(prefix!=NULL)
        prefixLength = strlen(prefix);

    //if(ib)
    //    nameLength = length + digitLength + 4;
    //else
    nameLength = length + digitLength+1+prefixLength+suffixLength;
   
    //RINTF("%d\n", nameLength);


    char * outName = hiveMalloc(nameLength);

    if(prefix!=NULL)
    {
        strncpy(outName, prefix, prefixLength );
        strncpy(outName+prefixLength, name, length );
    }
    else
        strncpy(outName, name, length );

    for(i=digitLength; i>0; i--)
    {
        outName[prefixLength+length+i-1]= ((int)'0')+value%10;
        value /=10;
    }
    
    //if(ib)
    //    strncpy(outName+digitLength+length,"-ib\0",4);
    //else
    if(suffix!= NULL)
    {
        strncpy(outName+prefixLength+digitLength+length,suffix,suffixLength);
        strncpy(outName+prefixLength+digitLength+length+suffixLength,"\0",1);
    }
    else
        strncpy(outName+prefixLength+digitLength+length,"\0",1);

    //PRINTF("%s\n",outName);

    return outName;
}

unsigned int hiveConfigCountNodes( char * nodeList )
{
    unsigned int i,j, length = strlen(nodeList);

    unsigned int rangeCount=0;
    unsigned int nodes = 0;

    for(i=0; i< length; i++)
        if(nodeList[i] == ',')
            nodes++;

    nodes++;

    char * begin, * end, * search;
    begin = nodeList;

    for(i=0; i< length; i++)
    {
        if(nodeList[i] == ',')
            begin = nodeList + i;
        else if(nodeList[i] == ':')
        {

            nodes--;
            search = nodeList+i;
            end = nodeList+length;

            for(j=0; j < end - search; j++)
            {
                if(nodeList[i+j] == ',')
                {
                    end = nodeList+i+j;
                }
            }

            unsigned int front = hiveConfigGetValue(begin,nodeList+i);

            unsigned int back = hiveConfigGetValue(nodeList+i+1,end);

            if(front>back)
                nodes+=(front-back)+1;
            else
                nodes+=(back-front)+1;

        }
    }

    return nodes;
}

unsigned int hiveConfigCountSlurmNodes( char * nodeList )
{
    unsigned int i,j, length = strlen(nodeList);

    unsigned int rangeCount=0;
    unsigned int nodes = 0;

    for(i=0; i< length; i++)
        if(nodeList[i] == ',')
            nodes++;

    nodes++;

    char * begin, * end, * search;
    begin = nodeList;

    for(i=0; i< length; i++)
    {
        if(nodeList[i] == ',')
            begin = nodeList + i+1;
        else if(nodeList[i] == '-')
        {

            nodes--;
            search = nodeList+i;
            end = nodeList+length;

            for(j=0; j < end - search; j++)
            {
                if(nodeList[i+j] == ',')
                {
                    end = nodeList+i+j;
                }
            }

            *(nodeList+i) = '\0'; 
            //PRINTF("aas %s\n", begin);
            unsigned int front =strtol(begin,NULL, 10);
            *(nodeList+i) = '-'; 
    
            *(end) = '\0'; 
            //PRINTF("aas2 %s\n", nodeList+i+1);
            unsigned int back = strtol(nodeList+i+1,NULL,10);
            *(end) = ','; 

            if(front>back)
                nodes+=(front-back)+1;
            else
                nodes+=(back-front)+1;

        }
    }
    //PRINTF("kldkld %d\n",nodes);
    return nodes;
}

void hiveConfigCreateRoutingTable( struct hiveConfig ** config, char* nodeList )
{
    unsigned int nodeCount;

    struct hiveConfigTable * table;

    unsigned int currentNode = 0, strLength;

    char * nodeBegin;
    char * nodeEnd, * temp, * nodeNext, *next;

    unsigned int start;
    unsigned int stop;
    unsigned int direction, listLength;

    nodeBegin = nodeList;
    listLength  =strlen(nodeList);

    bool done = false;
    nodeBegin = strtok(nodeBegin, "[");
    next  = strtok(NULL, "[");
    
    unsigned int suffixLength = 0;
    unsigned int prefixLength = 0;
    unsigned int totalLength = 0;
    
    char* prefix = (*config)->prefix;
    char* suffix = (*config)->suffix;

    if(suffix!= NULL)
        suffixLength = strlen(suffix);
    
    if(prefix!=NULL)
        prefixLength = strlen(prefix);
    //PRINTF("kkkkkk Here %d %s\n", (*config)->masterBoot, next);

    if(next != NULL || !(*config)->masterBoot )
    {
        char * name = nodeBegin;
        if(next != NULL)
        {
            DPRINTF("kkk Here %d %s\n", nodeCount, nodeBegin);
            nodeCount = hiveConfigCountSlurmNodes( next );
            nodeBegin = nodeBegin+strlen(nodeBegin)+1;
            (*config)->tableLength = nodeCount;

            table = hiveMalloc( sizeof( struct hiveConfigTable ) * nodeCount );
        }
        else
        {
            //Single Node
            DPRINTF("Here %d %s\n", nodeCount, nodeBegin);
            nodeCount = hiveConfigCountSlurmNodes( nodeBegin );
            table = hiveMalloc( sizeof( struct hiveConfigTable ) * nodeCount );
            (*config)->tableLength = nodeCount;
            
            table[currentNode].rank = currentNode;
            
            strLength = strlen(nodeBegin);
            totalLength = strLength+1+prefixLength+suffixLength;

            temp = hiveMalloc(totalLength);

            if(prefix!=NULL)
                strncpy(temp, prefix, prefixLength);
            
            strncpy(temp+prefixLength, nodeBegin, strLength);
            
            if(suffix!=NULL)
                strncpy(temp+prefixLength+strLength, suffix, suffixLength);
            
            strncpy(temp+totalLength-1, "\0", 1);

            table[currentNode].ipAddress = temp;
            DPRINTF("Here %s\n", temp );
            currentNode++;
        }
        //PRINTF("Here %d %s %s\n", nodeCount, nodeBegin, next);
        //PRINTF("%s\n", name);
        //PRINTF("%s\n", nodeBegin);
        
        if(next != NULL)
        {
        do
        {
            nodeBegin = strtok(nodeBegin, ",");
            next = nodeBegin+strlen(nodeBegin)+1;
            DPRINTF("%s\n", nodeBegin);
            if ( nodeBegin != NULL )
            {
                nodeBegin = strtok(nodeBegin, "-");
                nodeEnd = strtok(NULL, "-");
                
                if(nodeEnd!= NULL)
                {
                    if(nodeEnd[strlen(nodeEnd)-1]==']')
                    {
                        nodeEnd[strlen(nodeEnd)-1]='\0';
                        done = true;
                    }
                    DPRINTF("%s\n", nodeBegin);
                    DPRINTF("%s\n", nodeEnd);

                    start = strtol(nodeBegin,NULL, 10);
                    stop = strtol(nodeEnd,NULL, 10);
                    
                    if(start < stop)
                        direction = 1;
                    else
                        direction = -1;
                    
                    while(start != stop+1)
                    {
                        table[currentNode].rank = currentNode;
                        
                        //PRINTF("%s\n", name);
                        table[currentNode].ipAddress = hiveConfigGetSlurmHostname( name, nodeBegin, start, (*config)->ibNames, (*config)->prefix, (*config)->suffix);
                        DPRINTF("%s\n", table[currentNode].ipAddress );
                        start += direction;
                        currentNode++;

                    }
                }
                else
                {
                    if(nodeBegin[strlen(nodeBegin)-1]==']')
                    {
                        nodeBegin[strlen(nodeBegin)-1]='\0';
                        done = true;
                    }
                    DPRINTF("cc %s\n", nodeBegin);
                    table[currentNode].rank = currentNode;

                    unsigned int nameLength = strlen(name);

                    strLength = strlen(nodeBegin)+nameLength;
                    totalLength = strLength+1+prefixLength+suffixLength;
                    temp = hiveMalloc(totalLength);
                    if(prefix!=NULL)
                        strncpy(temp, prefix, prefixLength);
                    
                    //PRINTF("totalLength %d %d %d %s %s\n", totalLength, prefixLength+nameLength, strLength, name, nodeBegin);

                    strncpy(temp+prefixLength, name, nameLength);
                    
                    strncpy(temp+prefixLength+nameLength, nodeBegin, strlen(nodeBegin) );
                    
                    if(suffix!=NULL)
                        strncpy(temp+prefixLength+strLength, suffix, suffixLength);
                   
                    strncpy(temp+totalLength-1, "\0", 1);


                    DPRINTF("%s\n", temp);
                    table[currentNode].ipAddress = temp;
                    currentNode++;
                }

            }

            nodeBegin = next;

        }while(!done);
        }

        //exit(0);

    }
    else
    {
        nodeCount = hiveConfigCountNodes( nodeList );
        (*config)->tableLength = nodeCount;
        table = hiveMalloc( sizeof( struct hiveConfigTable ) * nodeCount );
        nodeBegin = nodeList;
    do
    {
        //exit(0);
        nodeBegin = strtok(nodeBegin, ",");
        
        next= nodeBegin+strlen(nodeBegin)+1;
        //nodeNext = strtok(NULL, ",");

        if ( nodeBegin != NULL )
        {
            nodeBegin = strtok(nodeBegin, ":");
            nodeEnd = strtok(NULL, ":");

            if(nodeBegin[strlen(nodeBegin)-1]=='\n')
                nodeBegin[strlen(nodeBegin)-1]='\0';


            if(nodeEnd != NULL)
            {
                start = hiveConfigGetValue(nodeBegin, nodeEnd-1);
                stop = hiveConfigGetValue(nodeEnd, nodeEnd+strlen(nodeEnd));

                char * name = hiveConfigGetNodeName(nodeBegin, nodeEnd);

                //if(nodeBegin < nodeEnd)
                if(start < stop)
                    direction = 1;
                else
                    direction = -1;

                strLength = strlen(nodeBegin);
                while(start != stop+1)
                {
                    table[currentNode].rank = currentNode;

                    table[currentNode].ipAddress = hiveConfigGetHostname( name, start );
                    start += direction;
                    currentNode++;

                }
            }
            else
            {
                table[currentNode].rank = currentNode;

                strLength = strlen(nodeBegin);
                
                temp = hiveMalloc(strLength+1);


                strncpy(temp, nodeBegin, strLength+1);
                
                DPRINTF("%s a\n", temp);

                table[currentNode].ipAddress = temp;
                currentNode++;
            }
            //nodeBegin = nodeEnd;
        }
        nodeBegin = next;
    }
    while(nodeBegin < nodeList+listLength);

    }

    //if(((*config)-> nodes != 0 || (*config)->nodes <= nodeCount) && (*config)->masterBoot )
    if(((*config)-> nodes != 0 || (*config)->nodes <= nodeCount))
        (*config)->tableLength = (*config)-> nodes;
    
    (*config)->table = table;
}

unsigned int hiveConfigGetNumberOfThreads(char * location)
{
    FILE * configFile = NULL;
    if(location == NULL)
        configFile = fopen( "hive.cfg", "r" );
    else
        configFile = fopen( location, "r" );

    if(configFile == NULL)
    {
        return 4;
    }

    return hiveConfigGetVariable( configFile, "threads");
}



struct hiveConfig * hiveConfigLoad( int argc, char ** argv, char * location )
{
    FILE * configFile = NULL;
    struct hiveConfig * config;
    struct hiveConfigVariable * configVariables;
    struct hiveConfigVariable * foundVariable;
    char * foundVariableChar;

    char * end = NULL;

    config = hiveMalloc( sizeof( struct hiveConfig ) );

    if(location == NULL)
    {
        location = getenv("hiveConfig");
        if(location)
            configFile = fopen( location, "r" );
        else    
            configFile = fopen( "hive.cfg", "r" );
    }
    else
        configFile = fopen( location, "r" );

    if(configFile == NULL)
    {
        PRINTF("No Config file found (./hive.cfg).\n");
        configVariables = NULL;
    }
    else
        configVariables = hiveConfigGetVariables( configFile );

    char *isSlurm = getenv("SLURM_NNODES");
    foundVariable = hiveConfigFindVariable(&configVariables, "launcher");
    if (isSlurm) {
      // ONCE_PRINTF("Reading nodes for slurm...\n");
        config->launcher = hiveConfigMakeNewVar("slurm");
    } else if (strncmp(foundVariable->value, "local", 5) == 0) {
        config->launcher = hiveConfigMakeNewVar("local");
    } else
        config->launcher = hiveConfigMakeNewVar("ssh");

    char * killSet = getenv("killMode");
    if(killSet == NULL)
    {
        if( (foundVariable = hiveConfigFindVariable(&configVariables,"killMode")) != NULL)
        {
            config->killMode = strtol( foundVariable->value, &end , 10);

            if(config->killMode )
            {
                ONCE_PRINTF("Killmode set: Attempting to kill remote proccesses.\n");
            }
        } 
        else
        {
            config->killMode = 0;
        }
    }
    else
    {
        config->killMode = strtol( killSet, &end , 10);
    }
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables,"coreDump")) != NULL)
    {
        config->coreDump = strtol( foundVariable->value, &end , 10);
    }
    else
    {
        config->coreDump = 0;
    }
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables,"pinStride")) != NULL)
    {
        config->pinStride = strtol( foundVariable->value, &end , 10);
    }
    else
    {
        config->pinStride = 1;
    }
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables,"printTopology")) != NULL)
    {
        config->printTopology = strtol( foundVariable->value, &end , 10);
    }
    else
    {
        config->printTopology = 0;
    }
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables,"threads")) != NULL)
        config->threadCount = strtol( foundVariable->value, &end , 10);
    else
    {
        ONCE_PRINTF("Defaulting to 4 threads\n");
        config->threadCount = 4;
    }
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables,"osThreads")) != NULL)
        config->osThreadCount = strtol( foundVariable->value, &end , 10);
    else
    {
        config->osThreadCount = 0;
    }
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables,"ports")) != NULL)
        config->ports = strtol( foundVariable->value, &end , 10);
    else if (strncmp(config->launcher, "local", 5) != 0) 
    {
        ONCE_PRINTF("Defaulting to 1 connection per node\n");
        config->ports = 1;
    }
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables,"outgoing")) != NULL)
        config->senderCount = strtol( foundVariable->value, &end , 10);
    else if (strncmp(config->launcher, "local", 5) != 0) 
    {
        ONCE_PRINTF("Defaulting to 1 sender\n");
        config->senderCount = 1;
    }
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables,"incoming")) != NULL)
        config->recieverCount = strtol( foundVariable->value, &end , 10);
    else if (strncmp(config->launcher, "local", 5) != 0) 
    {
        ONCE_PRINTF("Defaulting to 1 reciever\n");
        config->recieverCount = 1;
    }
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables,"sockets")) != NULL)
        config->socketCount = strtol( foundVariable->value, &end , 10);
    else
    {
        //ONCE_PRINTF("Defaulting to 1 sockets\n");
        config->socketCount = 1;
    }

    if( (foundVariable = hiveConfigFindVariable(&configVariables, "netInterface")) != NULL)
    {
        config->netInterface = hiveConfigMakeNewVar( foundVariable->value );
        
        if(config->netInterface[0] == 'i')
            config->ibNames=true;
        else
            config->ibNames=false;
    }
    else
    {
        //ONCE_PRINTF("No network interface given: defaulting to eth0\n");

        config->netInterface = NULL;//hiveConfigMakeNewVar( "eth0" );
        config->ibNames=false;
    }

    if( (foundVariable = hiveConfigFindVariable(&configVariables, "protocol")) != NULL)
        config->protocol = hiveConfigMakeNewVar( foundVariable->value );
    else
    {
        //ONCE_PRINTF("No protocol given: defaulting to tcp\n");

        config->protocol = hiveConfigMakeNewVar( "tcp" );
    }

    if( (foundVariable = hiveConfigFindVariable(&configVariables, "masterNode")) != NULL)
        config->masterNode = hiveConfigMakeNewVar( foundVariable->value );
    else if (strncmp(config->launcher, "local", 5) != 0) 
    {
      if (strncmp(config->launcher, "slurm", 5) != 0)
        ONCE_PRINTF("No master given: defaulting to first node in node list\n");

      config->masterNode = NULL;
    }
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables, "prefix")) != NULL)
        config->prefix = hiveConfigMakeNewVar( foundVariable->value );
    else
        config->prefix = NULL;
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables, "suffix")) != NULL)
        config->suffix = hiveConfigMakeNewVar( foundVariable->value );
    else
        config->suffix = NULL;
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables, "introspectiveConf")) != NULL)
        config->introspectiveConf = hiveConfigMakeNewVar( foundVariable->value );
    else
        config->introspectiveConf = NULL;
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables, "introspectiveFolder")) != NULL)
        config->introspectiveFolder = hiveConfigMakeNewVar( foundVariable->value );
    else
        config->introspectiveFolder = NULL;
    
    if( (foundVariableChar = hiveConfigFindVariableChar(configVariables,"introspectiveTraceLevel")) != NULL)
        config->introspectiveTraceLevel = strtol( foundVariableChar, &end , 10);
    else
        config->introspectiveTraceLevel = 1;
    
    if( (foundVariableChar = hiveConfigFindVariableChar(configVariables,"introspectiveStartPoint")) != NULL)
        config->introspectiveStartPoint = strtol( foundVariableChar, &end , 10);
    else
        config->introspectiveStartPoint = 1;

    if( (foundVariable = hiveConfigFindVariable(&configVariables, "counterFolder")) != NULL)
        config->counterFolder = hiveConfigMakeNewVar( foundVariable->value );
    else
        config->counterFolder = NULL;
    
    if( (foundVariableChar = hiveConfigFindVariableChar(configVariables,"counterStartPoint")) != NULL)
        config->counterStartPoint = strtol( foundVariableChar, &end , 10);
    else
        config->counterStartPoint = 1;
    
    if( (foundVariableChar = hiveConfigFindVariableChar(configVariables,"printNodeStats")) != NULL)
        config->printNodeStats = strtol( foundVariableChar, &end , 10);
    else
        config->printNodeStats = 0;
    
    if( (foundVariableChar = hiveConfigFindVariableChar(configVariables,"scheduler")) != NULL)
        config->scheduler = strtol( foundVariableChar, &end , 10);
    else
        config->scheduler = 0;
    
    //WARNING: Slurm Launcher Set!  
    if(strncmp(config->launcher, "slurm", 5 )==0)
    {
      ONCE_PRINTF("Using Slurm\n");

      config->masterBoot = false;
      char *threadsTemp = getenv("SLURM_CPUS_PER_TASK");
      if (threadsTemp != NULL)
        config->threadCount = strtol(threadsTemp, &end, 10);

      char *slurmNodes;

      slurmNodes = getenv("SLURM_NNODES");

      config->nodes = strtol(slurmNodes, &end, 10);

      char *nodeList = getenv("SLURM_STEP_NODELIST");
      DPRINTF("nodes: %s\n", nodeList);

      hiveConfigCreateRoutingTable(&config, nodeList);

      // if(config->masterNode == NULL)
      {
        unsigned int length = strlen(config->table[0].ipAddress) + 1;
        config->masterNode = hiveMalloc(sizeof(char) * length);

        strncpy(config->masterNode, config->table[0].ipAddress, length);
        }
        int i;
        for(i=0; i<config->tableLength; i++)
        {
            config->table[i].rank=i;
            if(strcmp(config->masterNode, config->table[i].ipAddress)==0)
            {
                DPRINTF("%d %s\n", i, config->table[i].ipAddress);
                config->masterRank=i;
            }
        }
    }
    else if(strncmp(config->launcher, "ssh", 5 )==0)
    {
      config->launcherData =
          hiveRemoteLauncherCreate(argc, argv, config, config->killMode,
                                   hiveRemoteLauncherSSHStartupProcesses,
                                   hiveRemoteLauncherSSHCleanupProcesses);
      config->masterBoot = true;

      if ((foundVariable =
               hiveConfigFindVariable(&configVariables, "nodeCount")) != NULL)
        config->nodes = strtol(foundVariable->value, &end, 10);
      else {
        config->nodes = 1;
        }
        char * nodeList=0;
        if( (foundVariable = hiveConfigFindVariable(&configVariables, "nodes")) != NULL)
        {
            nodeList = foundVariable->value;
        }
        else
        {
            ONCE_PRINTF("No nodes given: defaulting to 1 node\n");
        
            nodeList = hiveMalloc(sizeof(char)*strlen("localhost\0"));
            
            strncpy(nodeList, "localhost\0", strlen("localhost\0")+1 );
        }
            DPRINTF("nodes: %s\n", nodeList);
            hiveConfigCreateRoutingTable( &config, nodeList );

            if(config->masterNode == NULL)
            {
                unsigned int length = strlen(config->table[0].ipAddress)+1;
                config->masterNode = hiveMalloc(sizeof(char)*length);

                strncpy( config->masterNode, config->table[0].ipAddress, length );
            }
            int i;
            for(i=0; i<config->tableLength; i++)
            {
                config->table[i].rank=i;
                if(strcmp(config->masterNode, config->table[i].ipAddress)==0)
                {
                    DPRINTF("Here %d\n", config->tableLength);
                    config->masterRank=i;
                }
            }
    }
    else if(strncmp(config->launcher, "local",5) == 0)
    {
        ONCE_PRINTF("Running in Local Mode.\n");
        config->masterBoot = false;
        config->masterNode = NULL;
        // OS Threads
        char * threadsOS = getenv("OS_THREAD_COUNT");
        if(threadsOS != NULL)
            config->osThreadCount = strtol(threadsOS, &end, 10);
        else if(!config->osThreadCount)
            config->osThreadCount = 0; // Default to single thread.
        // OS Threads
        char *threadsUSER = getenv("USER_THREAD_COUNT");
        if (threadsUSER != NULL)
            config->threadCount = strtol(threadsUSER, &end, 10);
        else if (!config->threadCount)
            config->threadCount = 4; // Default to single thread.
        config->nodes = 1;
        config->tableLength = 1; // for GUID
        config->masterRank = 0;    
    }
    else
    {
        ONCE_PRINTF("Unknown launcher: %s\n", config->launcher);
        exit(1);
    }

    if ((foundVariableChar = hiveConfigFindVariableChar(
             configVariables, "remoteWorkStealing")) != NULL) {
        config->remoteWorkStealing = strtol(foundVariableChar, &end, 10);
    } 
    else if (strncmp(config->launcher, "local", 5) != 0) 
    {
        config->remoteWorkStealing = 0;
        ONCE_PRINTF("Defaulting to no remote work stealing\n");
    }

    if( (foundVariable = hiveConfigFindVariable(&configVariables,"stackSize")) != NULL)
        config->stackSize = strtoull( foundVariable->value, &end , 10);
    else
    {
        config->stackSize = 0;
    }
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables,"workerInitDequeSize")) != NULL)
        config->dequeSize = strtol( foundVariable->value, &end , 10);
    else
    {
        ONCE_PRINTF("Defaulting the worker queue length to 4096\n");
        config->dequeSize = 4096;
    }

    if( (foundVariable = hiveConfigFindVariable(&configVariables,"port")) != NULL)
        config->port = strtol( foundVariable->value, &end , 10);
    else if (strncmp(config->launcher, "local", 5) != 0) 
    {
        ONCE_PRINTF("Defaulting port to %d\n", 75563);
        config->port = 75563;
    }
    if( (foundVariable = hiveConfigFindVariable(&configVariables,"routeTableSize")) != NULL)
        config->routeTableSize = strtol( foundVariable->value, &end , 10);
    else {
        ONCE_PRINTF("Defaulting routing table size to 2^20\n");
        config->routeTableSize = 20;
    }

    int routeTableEntries = 1;

    if (config->launcher != NULL) { //&&
        //(strncmp(config->launcher, "local", 5) != 0)) {
      for (int i = 0; i < config->routeTableSize; i++)
        routeTableEntries *= 2;
      config->routeTableEntries = routeTableEntries;
    }
    
    if( (foundVariable = hiveConfigFindVariable(&configVariables,"pin")) != NULL)
        config->pinThreads = strtol( foundVariable->value, &end , 10);
    else
    {
        config->pinThreads = 1; // default thread pinning 
    }
    
    DPRINTF("Config Parsed\n");

    return config;
}

void hiveConfigDestroy( void * config )
{
    hiveFree( config );
}

