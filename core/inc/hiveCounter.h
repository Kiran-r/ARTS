#ifndef HIVECOUNTER_H
#define	HIVECOUNTER_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "hiveTimer.h"
    
#ifdef JUSTCOUNT
#define COUNTERTIMESTAMP 0
#elif defined(COUNT) || defined(MODELCOUNT)
#define COUNTERTIMESTAMP hiveGetTimeStamp()
#else
#define COUNTERTIMESTAMP 0
#endif
    
#ifdef MODELCOUNT
    
#define COUNT_edtCounter(x) x
#define COUNT_sleepCounter(x) x
#define COUNT_totalCounter(x) x
#define COUNT_signalEdtCounter(x) x
#define COUNT_signalEventCounter(x) x 
#define COUNT_mallocMemory(x) x
#define COUNT_callocMemory(x) x
#define COUNT_freeMemory(x) x
#define COUNT_edtFree(x) x
#define COUNT_emptyTime(x) x


#define CAT(a, ...) PRIMITIVE_CAT(a, __VA_ARGS__)
#define PRIMITIVE_CAT(a, ...) a ## __VA_ARGS__
    
#define IIF(c) PRIMITIVE_CAT(IIF_, c)
#define IIF_0(t, ...) __VA_ARGS__
#define IIF_1(t, ...) t  
    
#define COMPL(b) PRIMITIVE_CAT(COMPL_, b)
#define COMPL_0 1
#define COMPL_1 0
    
#define BITAND(x) PRIMITIVE_CAT(BITAND_, x)
#define BITAND_0(y) 0
#define BITAND_1(y) y
    
#define CHECK_N(x, n, ...) n
#define CHECK(...) CHECK_N(__VA_ARGS__, 0,)
#define PROBE(x) x, 1
    
#define NOT(x) CHECK(PRIMITIVE_CAT(NOT_, x))
#define NOT_0 PROBE(~)
    
#define BOOL(x) COMPL(NOT(x))
#define IF(c) IIF(BOOL(c))
#define EAT(...)
  
#define IS_PAREN(x) CHECK(IS_PAREN_PROBE x)
#define IS_PAREN_PROBE(...) PROBE(~)

#define COUNTER_ON(x) IS_PAREN( CAT(COUNT_, x) (()) )
    
#define INITCOUNTERLIST(threadId, nodeId, folder, startPoint) hiveInitCounterList(threadId, nodeId, folder, startPoint)
#define HIVESTARTCOUNTING(startPoint) hiveStartCounters(startPoint)
#define HIVECOUNTERSON() hiveCountersOn()
#define HIVECOUNTERSOFF() hiveEndCounters()
#define HIVECREATECOUNTER(counter) hiveCreateCounter(threadId, nodeId, counterName)
#define HIVEGETCOUNTER(counter) hiveGetCounter(counter)
    
#define HIVECOUNTERINCREMENT(counter)         IF(COUNTER_ON(counter)) ( hiveCounterIncrement,         EAT ) (hiveGetCounter(counter))
#define HIVECOUNTERINCREMENTBY(counter, num)  IF(COUNTER_ON(counter)) ( hiveCounterIncrementBy,       EAT ) (hiveGetCounter(counter), num)
#define HIVECOUNTERTIMERSTART(counter)        IF(COUNTER_ON(counter)) ( hiveCounterTimerStart,        EAT ) (hiveGetCounter(counter))
#define HIVECOUNTERTIMERENDINCREMENT(counter) IF(COUNTER_ON(counter)) ( hiveCounterTimerEndIncrement, EAT ) (hiveGetCounter(counter)) 
#define HIVECOUNTERTIMERENDINCREMENTBY(counter, num) IF(COUNTER_ON(counter)) ( hiveCounterTimerEndIncrementBy, EAT ) (hiveGetCounter(counter), num) 
#define HIVECOUNTERADDTIME(counter, time)     IF(COUNTER_ON(counter)) ( hiveCounterAddTime,           EAT ) (hiveGetCounter(counter), time)
#define HIVECOUNTERSETENDTOTIME(counter)      IF(COUNTER_ON(counter)) ( hiveCounterSetEndTime,        EAT ) (hiveGetCounter(counter), hiveExtGetTimeStamp())
#define HIVECOUNTERADDENDTIME(counter)        IF(COUNTER_ON(counter)) ( hiveCounterAddEndTime,        EAT ) (hiveGetCounter(counter))
#define HIVECOUNTERNONEMPTY(counter)          IF(COUNTER_ON(counter)) ( hiveCounterNonEmtpy,          EAT ) (hiveGetCounter(counter))

#define INTERNAL_HIVEEDTCOUNTERTIMERSTART(counter) hiveCounterTimerStart(hiveGetCounter((hiveThreadInfo.currentEdtGuid) ? (counter ## On) : (counter)))
#define HIVEEDTCOUNTERTIMERSTART(counter) IF(COUNTER_ON(counter)) ( INTERNAL_HIVEEDTCOUNTERTIMERSTART, EAT)(counter)
    
#define INTERNAL_HIVEEDTCOUNTERTIMERENDINCREMENT(counter) hiveCounterTimerEndIncrement(hiveGetCounter((hiveThreadInfo.currentEdtGuid) ? (counter ## On) : (counter)))    
#define HIVEEDTCOUNTERTIMERENDINCREMENT(counter) IF(COUNTER_ON(counter)) ( INTERNAL_HIVEEDTCOUNTERTIMERENDINCREMENT, EAT)(counter)

#define HIVERESETCOUNTER(counter)
#define HIVECOUNTERTIMERENDOVERWRITE(counter)
#define HIVECOUNTERPRINT(counter, stream)
#define HIVECOUNTERSETSTARTTIME(counter, start)
#define HIVECOUNTERSETENDTIME(counter, end)
#define HIVECOUNTERGETSTARTTIME(counter)
#define HIVECOUNTERGETENDTIME(counter)
#define HIVEUSERGETCOUNTER(counter)
#define HIVEEDTCOUNTERINCREMENT(counter)
#define HIVEEDTCOUNTERINCREMENTBY(counter, num)
#define HIVEUSERCOUNTERINCREMENT(counter)
#define HIVEUSERCOUNTERINCREMENTBY(counter, num)
#define HIVEUSERCOUNTERTIMERSTART(counter) hiveCounterTimerStart(hiveUserGetCounter(counter, #counter))
#define HIVEUSERCOUNTERTIMERENDINCREMENT(counter) hiveCounterTimerEndIncrement(hiveUserGetCounter(counter, #counter))
#define USERCOUNTERS(first, ...) enum __userCounters{ first=lastCounter, __VA_ARGS__}
#define USERCOUNTERINIT(counter) hiveUserGetCounter(counter, #counter)

#elif COUNT    
#define INITCOUNTERLIST(threadId, nodeId, folder, startPoint) hiveInitCounterList(threadId, nodeId, folder, startPoint)
#define HIVESTARTCOUNTING(startPoint) hiveStartCounters(startPoint)
#define HIVECOUNTERSON() hiveCountersOn()
#define HIVECOUNTERSOFF() hiveEndCounters()
#define HIVECREATECOUNTER(counter) hiveCreateCounter(threadId, nodeId, counterName)
#define HIVEGETCOUNTER(counter) hiveGetCounter(counter)
#define HIVERESETCOUNTER(counter) hiveResetCounter(hiveGetCounter(counter))
#define HIVECOUNTERINCREMENT(counter) hiveCounterIncrement(hiveGetCounter(counter))
#define HIVECOUNTERINCREMENTBY(counter, num) hiveCounterIncrementBy(hiveGetCounter(counter), num)
#define HIVECOUNTERTIMERSTART(counter) hiveCounterTimerStart(hiveGetCounter(counter))
#define HIVECOUNTERTIMERENDINCREMENT(counter) hiveCounterTimerEndIncrement(hiveGetCounter(counter))
#define HIVECOUNTERTIMERENDINCREMENTBY(counter, num) hiveCounterTimerEndIncrementBy(hiveGetCounter(counter), num)
#define HIVECOUNTERTIMERENDOVERWRITE(counter) hiveCounterTimerEndOverwrite(hiveGetCounter(counter))
#define HIVECOUNTERPRINT(counter, stream) hiveCounterPrint(hiveGetCounter(counter), stream)
#define HIVECOUNTERSETSTARTTIME(counter, start) hiveCounterSetStartTime (hiveGetCounter(counter), start)
#define HIVECOUNTERSETENDTIME(counter, end) hiveCounterSetEndTime(hiveGetCounter(counter), end)
#define HIVECOUNTERGETSTARTTIME(counter) hiveCounterGetStartTime(hiveGetCounter(counter))
#define HIVECOUNTERGETENDTIME(counter) hiveCounterGetEndTime(hiveGetCounter(counter))  
#define HIVECOUNTERADDTIME(counter, time) hiveCounterAddTime(hiveGetCounter(counter), time)
#define HIVEUSERGETCOUNTER(counter) hiveUserGetCounter(counter, #counter)
#define HIVECOUNTERSETENDTOTIME(counter) hiveCounterSetEndTime(hiveGetCounter(counter), hiveExtGetTimeStamp())
#define HIVECOUNTERADDENDTIME(counter) hiveCounterAddEndTime(hiveGetCounter(counter))
#define HIVECOUNTERNONEMPTY(counter) hiveCounterNonEmtpy(hiveGetCounter(counter))
    
#define HIVEEDTCOUNTERINCREMENT(counter) hiveCounterIncrement(hiveGetCounter((hiveThreadInfo.currentEdtGuid) ? (counter ## On) : (counter)))
#define HIVEEDTCOUNTERINCREMENTBY(counter, num) hiveCounterIncrementBy(hiveGetCounter((hiveThreadInfo.currentEdtGuid) ? (counter ## On) : (counter)), num)
#define HIVEEDTCOUNTERTIMERSTART(counter) hiveCounterTimerStart(hiveGetCounter((hiveThreadInfo.currentEdtGuid) ? (counter ## On) : (counter)))
#define HIVEEDTCOUNTERTIMERENDINCREMENT(counter) hiveCounterTimerEndIncrement(hiveGetCounter((hiveThreadInfo.currentEdtGuid) ? (counter ## On) : (counter)))    
    
#define HIVEUSERCOUNTERINCREMENT(counter) hiveCounterIncrement(hiveUserGetCounter(counter, #counter))
#define HIVEUSERCOUNTERINCREMENTBY(counter, num) hiveCounterIncrementBy(hiveUserGetCounter(counter, #counter), num)
#define HIVEUSERCOUNTERTIMERSTART(counter) hiveCounterTimerStart(hiveUserGetCounter(counter, #counter))
#define HIVEUSERCOUNTERTIMERENDINCREMENT(counter) hiveCounterTimerEndIncrement(hiveUserGetCounter(counter, #counter))
#define USERCOUNTERINIT(counter) hiveUserGetCounter(counter, #counter)
#define USERCOUNTERS(first, ...) enum __userCounters{ first=lastCounter, __VA_ARGS__}

#else

#define INITCOUNTERLIST(threadId, nodeId, folder, startPoint)
#define HIVESTARTCOUNTING(startPoint)
//This seems backwards but it is for hiveMallocFOA
#define HIVECOUNTERSON() 1
#define HIVECOUNTERSOFF()
#define HIVECREATECOUNTER(counter)
#define HIVEGETCOUNTER(counter)
#define HIVERESETCOUNTER(counter)
#define HIVECOUNTERINCREMENT(counter)
#define HIVECOUNTERINCREMENTBY(counter, num)
#define HIVECOUNTERTIMERSTART(counter)
#define HIVECOUNTERTIMERENDINCREMENT(counter)
#define HIVECOUNTERTIMERENDINCREMENTBY(counter, num)
#define HIVECOUNTERTIMERENDOVERWRITE(counter)
#define HIVECOUNTERPRINT(counter, stream)
#define HIVECOUNTERSETSTARTTIME(counter, start)
#define HIVECOUNTERSETENDTIME(counter, end)
#define HIVECOUNTERGETSTARTTIME(counter)
#define HIVECOUNTERGETENDTIME(counter)  
#define HIVECOUNTERADDTIME(counter, time)
#define HIVEUSERGETCOUNTER(counter)
#define HIVEEDTCOUNTERINCREMENT(counter)
#define HIVEEDTCOUNTERINCREMENTBY(counter, num)
#define HIVEEDTCOUNTERTIMERSTART(counter)
#define HIVEEDTCOUNTERTIMERENDINCREMENT(counter)
#define HIVEUSERCOUNTERINCREMENT(counter)
#define HIVEUSERCOUNTERINCREMENTBY(counter, num)
#define HIVEUSERCOUNTERTIMERSTART(counter)
#define HIVEUSERCOUNTERTIMERENDINCREMENT(counter)
#define USERCOUNTERINIT(counter)
#define USERCOUNTERS(first, ...)
#define HIVECOUNTERSETENDTOTIME(counter)
#define HIVECOUNTERADDENDTIME(counter)
#define HIVECOUNTERNONEMPTY(counter)

#endif
    
#include "hiveArrayList.h"
    
#define COUNTERNAMES const char * const __counterName[] = { \
"edtCounter", \
"sleepCounter", \
"totalCounter", \
"signalEventCounter", \
"signalEventCounterOn", \
"signalEdtCounter", \
"signalEdtCounterOn", \
"edtCreateCounter", \
"edtCreateCounterOn", \
"eventCreateCounter", \
"eventCreateCounterOn", \
"dbCreateCounter", \
"dbCreateCounterOn", \
"mallocMemory", \
"mallocMemoryOn", \
"callocMemory", \
"callocMemoryOn", \
"freeMemory", \
"freeMemoryOn", \
"guidAllocCounter", \
"guidAllocCounterOn", \
"guidLookupCounter", \
"guidLookupCounterOn", \
"getDbCounter", \
"getDbCounterOn", \
"putDbCounter", \
"putDbCounterOn", \
"contextSwitch", \
"yield" \
}
    
#define GETCOUNTERNAME(x) __counterName[x] 
#define COUNTERARRAYBLOCKSIZE 128
#define COUNTERPREFIXSIZE 1024
#define COUNTERPREFIX "counters"
#define FIRSTCOUNTER edtCounter
#define LASTCOUNTER lastCounter
        
    enum hiveCounterType { 
        edtCounter=0, 
        sleepCounter, 
        totalCounter,
        signalEventCounter,
        signalEventCounterOn,
        signalEdtCounter,
        signalEdtCounterOn,
        edtCreateCounter,
        edtCreateCounterOn,
        eventCreateCounter,
        eventCreateCounterOn,
        dbCreateCounter,
        dbCreateCounterOn,
        mallocMemory,
        mallocMemoryOn,
        callocMemory,
        callocMemoryOn,
        freeMemory,
        freeMemoryOn,
        guidAllocCounter,
        guidAllocCounterOn,
        guidLookupCounter,
        guidLookupCounterOn,
        getDbCounter,
        getDbCounterOn,
        putDbCounter,
        putDbCounterOn,
        contextSwitch,
        yield,
        lastCounter
    };
    typedef enum hiveCounterType hiveCounterType;
    
    typedef struct {
        unsigned int threadId;
        unsigned int nodeId;
        const char * name;
        uint64_t count;
        uint64_t totalTime;
        uint64_t startTime;
        uint64_t endTime;
    } hiveCounter;
    
    void hiveInitCounterList(unsigned int threadId, unsigned int nodeId, char * folder, unsigned int startPoint);
    void hiveStartCounters(unsigned int startPoint);
    void hiveEndCounters();
    unsigned int hiveCountersOn();
    hiveCounter * hiveCreateCounter(unsigned int threadId, unsigned int nodeId, const char * counterName);
    hiveCounter * hiveGetCounter(hiveCounterType counter);
    hiveCounter * hiveUserGetCounter(unsigned int index, char * name);
    void hiveResetCounter(hiveCounter * counter);
    void hiveCounterIncrement(hiveCounter * counter);
    void hiveCounterIncrementBy(hiveCounter * counter, uint64_t num);
    void hiveCounterTimerStart(hiveCounter * counter);
    void hiveCounterTimerEndIncrement(hiveCounter * counter);
    void hiveCounterTimerEndIncrementBy(hiveCounter * counter, uint64_t num);
    void hiveCounterTimerEndOverwrite(hiveCounter * counter);
    void hiveCounterPrint(hiveCounter * counter, FILE * stream);
    void hiveCounterSetStartTime(hiveCounter * counter, uint64_t start);
    void hiveCounterSetEndTime(hiveCounter * counter, uint64_t end);
    void hiveCounterAddEndTime(hiveCounter * counter);
    void hiveCounterNonEmtpy(hiveCounter * counter);
    uint64_t hiveCounterGetStartTime(hiveCounter * counter);
    uint64_t hiveCounterGetEndTime(hiveCounter * counter);   
    void hiveCounterAddTime(hiveCounter * counter, uint64_t time);
    void hiveWriteCountersToFile(unsigned int threadId, unsigned int nodeId);
#ifdef __cplusplus
}
#endif

#endif	/* HIVECOUNTER_H */

