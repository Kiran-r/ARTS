#include <unistd.h>
#include <signal.h>
#include <execinfo.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void hiveDebugPrintStack()
{
        void *array[10];
        size_t size;

        size = backtrace(array, 10);
        backtrace_symbols_fd(array, size, STDOUT_FILENO);
}

void hiveDebugGenerateSegFault()
{
        raise(SIGSEGV);
}

char * getBackTrace(unsigned int skip)
{
    void *array[10];
    size_t size = backtrace(array, 10);
    char ** funct = backtrace_symbols(array, size);
    unsigned int length = 0;
    for(unsigned int i=skip; i<size; i++)
    {
        unsigned int flag = 0;
        for(unsigned int j=0; j<strlen(funct[i]); j++)
        {
            if(!flag)
            {   
                if(funct[i][j] == '(')
                    flag = 1;
            }
            else
            {
                if(funct[i][j] == ')')
                {
                    flag = 0;
                }
                else
                    length++;
            }
        }
        length++;
    }
    
    char * buffer = malloc(sizeof(char)*length);
    unsigned int temp = 0;
    for(unsigned int i=skip; i<size; i++)
    {
        unsigned int flag = 0;
        for(unsigned int j=0; j<strlen(funct[i]); j++)
        {
            if(!flag)
            {   
                if(funct[i][j] == '(')
                    flag = 1;
            }
            else
            {
                if(funct[i][j] == ')')
                {
                    flag = 0;
                }
                else
                {
                    buffer[temp++] = funct[i][j];
                }
            }
        }
        buffer[temp++] = '\n';
    }
    buffer[length-1] = '\0';
    free(funct);
    return buffer;
}