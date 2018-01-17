#ifndef HIVE_TERMINATION_DETECTION_H
#define  HIVE_TERMINATION_DETECTION_H
#ifdef __cplusplus
extern "C" {
#endif

void incrementActiveCount(unsigned int n);
void incrementFinishedCount(unsigned int n);
void hiveDetectTermination(hiveGuid_t finishGuid, unsigned int slot);
void initializeTerminationDetection(hiveGuid_t kickoffTerminationGuid);

#ifdef __cplusplus
}
#endif
#endif
